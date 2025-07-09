import argparse
import logging
import multiprocessing
from datetime import UTC, datetime
from functools import partial
from typing import Any, Literal

import fsspec
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import (
    Optimizer,
    OptimizerLRSchedulerConfig,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForMaskedLM

CADUCEUS_PAD_TOKEN_ID = 4

logger = logging.getLogger(__name__)


def _filter_non_specific_nucleotides_and_batch(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = input_ids != CADUCEUS_PAD_TOKEN_ID
    # 2d mask is going to collapses some dimensions
    student_logits = student_logits[mask]
    teacher_logits = teacher_logits[mask]
    input_ids = input_ids[mask]

    assert student_logits.ndim == 2
    assert teacher_logits.ndim == 2
    assert input_ids.ndim == 1

    return student_logits, teacher_logits, input_ids


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    temperature: float = 4.0,
    alpha: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate combined distillation loss (KL divergence with temperature scaling + cross-entropy with hard targets).

    Args:
        student_logits: Raw logits from student model [B, T, V]
        teacher_logits: Raw logits from teacher model [B, T, V]
        input_ids: Ground truth token ids (hard targets) [B, T]
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss (1-alpha for hard loss)

    Returns:
        Weighted sum of distillation and hard target loss
    """
    assert temperature > 0, "Temperature must be positive"
    assert 0 <= alpha <= 1, "Alpha must be in [0, 1]"

    # NOTE: we could easily make this function work on any number of dimensions, but for simplicity and to make it
    # fail quick if the shapes are wrong, we assume 3d input [B, T, V] for logits and [B, T] for input_ids.
    assert (
        student_logits.ndim == 3
    ), f"Expected student_logits to be 3D, got {student_logits.ndim}D"
    assert (
        teacher_logits.ndim == 3
    ), f"Expected teacher_logits to be 3D, got {teacher_logits.ndim}D"
    assert input_ids.ndim == 2, f"Expected input_ids to be 2D, got {input_ids.ndim}D"

    student_logits, teacher_logits, input_ids = (
        _filter_non_specific_nucleotides_and_batch(
            student_logits, teacher_logits, input_ids
        )
    )

    assert (
        input_ids.size(0) > 0
    ), "Input IDs must not be empty after filtering non-specific nucleotides"

    # Soft loss (distillation)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # NOTE: `batchmean` is required per pytorch docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
    soft_loss = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="batchmean",
        log_target=True,
    )

    # NOTE: re T^2 scaling, from `Distilling the Knowledge in a Neural Network`:
    # > Since the magnitudes of the gradients produced by the soft targets scale as 1/T 2 it is important
    # > to multiply them by T^2 when using both hard and soft targets. This ensures that the relative
    # > contributions of the hard and soft targets remain roughly unchanged if the temperature used for
    # > distillation is changed while experimenting with meta-parameters.
    if alpha != 1.0:
        soft_loss *= temperature**2

    # Hard loss (cross-entropy)
    hard_loss = F.cross_entropy(student_logits, input_ids)

    soft_loss_contrib = alpha * soft_loss
    hard_loss_contrib = (1 - alpha) * hard_loss
    total_loss = soft_loss_contrib + hard_loss_contrib

    return total_loss, soft_loss_contrib, hard_loss_contrib


EXAMPLE_T = tuple[torch.Tensor, torch.Tensor]


class DistillationDataset(Dataset[EXAMPLE_T]):
    def __init__(
        self,
        zarr_path: str,
    ) -> None:
        self.zarr_path = zarr_path
        self.ds: xr.Dataset | None = None

        with xr.open_zarr(self.zarr_path, chunks=None) as temp_ds:
            total_samples = len(temp_ds.sample)

            self._len = total_samples

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> EXAMPLE_T:
        if self.ds is None:
            self.ds = xr.open_zarr(self.zarr_path, chunks=None)

        sample = self.ds.isel(sample=idx)
        input_ids = torch.tensor(sample.input_ids.values, dtype=torch.long)

        if input_ids.eq(CADUCEUS_PAD_TOKEN_ID).all():
            raise ValueError(
                f"Sample {idx} contains only PAD tokens. This is not allowed."
            )

        teacher_logits = torch.tensor(sample.logits.values, dtype=torch.float32)
        return input_ids, teacher_logits


class CallbackWithExplicitSchedule(L.Callback):
    def __init__(self, schedule: set[int]) -> None:
        self.schedule = schedule
        # NOTE: when gradient accumulation is > 1, the global step updates every `accumulate_grad_batches` steps,
        # to compute validation once per global step, we need to track the last global step
        self._last_global_step: int = -1


class ValidationScheduler(CallbackWithExplicitSchedule):
    def __init__(self, schedule: set[int]) -> None:
        super().__init__(schedule)
        self.trainer: L.Trainer | None = None

    def on_fit_start(self, trainer: L.Trainer, *args: Any, **kwargs: Any) -> None:
        self.trainer = trainer
        self.trainer.fit_loop.epoch_loop._should_check_val_fx = partial(self.step)  # type: ignore[method-assign]

    def step(self, *args: Any, **kwargs: Any) -> bool:
        assert self.trainer is not None
        # NOTE: this method is called after the global_step is incremented
        last_step = self.trainer.global_step - 1
        should_run = last_step in self.schedule and self._last_global_step < last_step
        self._last_global_step = last_step
        return should_run


class GradientNormLogger(CallbackWithExplicitSchedule):
    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:
        cur_step = trainer.global_step
        should_log = cur_step in self.schedule and self._last_global_step < cur_step
        self._last_global_step = cur_step

        if not should_log:
            return

        lr = optimizer.param_groups[0]["lr"]

        # Full list of layers available here: https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16?show_file_info=model.safetensors
        significant_param_suffixes = [
            # in_proj.weight: This is the input projection matrix (i.e. entry point of the mamba block).
            "mamba_fwd.in_proj.weight",
            # out_proj.weight: The output projection matrix (i.e. final transformation within the mixer).
            "mamba_fwd.out_proj.weight",
            # x_proj.weight & dt_proj.weight: project the input x to dynamically generate the SSM parameters
            "mamba_fwd.x_proj.weight",
            "mamba_fwd.dt_proj.weight",
            # Forward (mamba_fwd) and reverse (mamba_rev) passes are tracked separately (weights are independent)
            "mamba_rev.x_proj.weight",
            "mamba_rev.dt_proj.weight",
        ]

        for name, param in pl_module.named_parameters():
            if param.grad is None:
                continue

            metric_prefix = None
            is_significant_layer_param = any(
                name.endswith(suffix) for suffix in significant_param_suffixes
            )
            # The first layer of the network
            is_embedding = "word_embeddings.embedding.weight" in name

            if is_significant_layer_param:
                try:
                    parts = name.split(".")
                    layer_idx_pos = parts.index("layers") + 1
                    layer_idx = int(parts[layer_idx_pos])
                    clean_param_name = ".".join(parts[layer_idx_pos + 1 :]).replace(
                        "mixer.submodule.", ""
                    )
                    metric_prefix = f"diag/L{layer_idx}/{clean_param_name}"
                except (ValueError, IndexError):
                    continue
            elif is_embedding:
                metric_prefix = "diag/embed"

            if metric_prefix:
                grad_norm = param.grad.norm(2)
                data_std = param.data.std()
                update_ratio = torch.tensor(
                    0.0, device=param.device
                )  # Default to 0 if data_std is 0
                if data_std > 0:
                    update_ratio = lr * param.grad.std() / (data_std + 1e-8)

                pl_module.log_dict(
                    {
                        f"{metric_prefix}/grad_norm": grad_norm,
                        f"{metric_prefix}/update_ratio": update_ratio,
                    }
                )


class StudentCaduceus(L.LightningModule):
    student: AutoModelForMaskedLM
    temperature: float
    lr: float
    alpha: float
    cosine_anneal: bool

    def __init__(
        self,
        teacher_model_name: str = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        lr: float = 1e-3,
        temperature: float = 4.0,
        alpha: float = 0.8,
        cosine_anneal: bool = False,
    ) -> None:
        super().__init__()
        self.cosine_anneal = cosine_anneal
        self.save_hyperparameters()

        # Create student config (half depth and width)
        student_config = AutoConfig.from_pretrained(
            teacher_model_name,
            trust_remote_code=True,
            d_model=128,  # Half of teacher's 256
            n_layer=8,  # Half of teacher's 16
        )

        # Verify student config parameters
        assert (
            student_config.d_model == 128
        ), f"Expected d_model=128, got {student_config.d_model}"
        assert (
            student_config.n_layer == 8
        ), f"Expected n_layer=8, got {student_config.n_layer}"

        self.student = AutoModelForMaskedLM.from_config(
            student_config, trust_remote_code=True
        )
        self.temperature = temperature
        self.lr = lr
        self.alpha = alpha

    def forward(
        self, input_ids: torch.Tensor, output_hidden_states: bool = False
    ) -> Any:
        return self.student(input_ids, output_hidden_states=output_hidden_states)

    def training_step(self, batch: EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        input_ids, teacher_logits = batch

        outputs = self.student(input_ids)
        student_logits = outputs.logits

        # Calculate combined loss
        loss, soft_loss, hard_loss = distillation_loss(
            student_logits,
            teacher_logits,
            input_ids,
            temperature=self.temperature,
            alpha=self.alpha,
        )

        self.log("train/loss/total", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                "train/loss/soft": soft_loss,
                "train/loss/hard": hard_loss,
            },
            on_step=True,
            on_epoch=True,
        )
        return loss

    def _base_validation_step(
        self,
        batch: EXAMPLE_T,
        batch_idx: int,
        is_final_val: bool,
    ) -> torch.Tensor:
        input_ids, teacher_logits = batch

        outputs = self.student(input_ids)
        student_logits = outputs.logits

        loss_eval, soft_loss_eval, hard_loss_eval = distillation_loss(
            student_logits,
            teacher_logits,
            input_ids,
            # The `temp` hyper-parameter should not affect the eval scoring
            # Also, temp should be set to 1.0 after the distillation is complete (per Hinton)
            temperature=1.0,
            # The `alpha` hyper-parameter should not affect the eval scoring
            # alpha=1.0 means that we only consider the soft targets
            alpha=1.0,
        )

        # Evaluation loss with training hyperparameters for comparison
        (
            loss_train_temp,
            soft_loss_train_temp,
            hard_loss_train_temp,
        ) = distillation_loss(
            student_logits,
            teacher_logits,
            input_ids,
            temperature=self.temperature,
            alpha=self.alpha,
        )

        # Use a different metric prefix for the final, post-fit validation run.
        prefix = "val" if not is_final_val else "final_val"

        self.log(
            f"{prefix}/loss/total", loss_eval, prog_bar=not is_final_val, sync_dist=True
        )

        metrics_to_log = {
            f"{prefix}/loss/soft": soft_loss_eval,
            f"{prefix}/loss/hard": hard_loss_eval,
            f"{prefix}/loss_train_temp/total": loss_train_temp,
            f"{prefix}/loss_train_temp/soft": soft_loss_train_temp,
            f"{prefix}/loss_train_temp/hard": hard_loss_train_temp,
        }
        self.log_dict(metrics_to_log, sync_dist=True)
        return loss_eval

    def validation_step(self, batch: EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        return self._base_validation_step(batch, batch_idx, is_final_val=False)

    def test_step(self, batch: EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        """
        Test step is used for final validation after training.
        It uses the same logic as validation_step but with final_val prefix.
        """
        return self._base_validation_step(batch, batch_idx, is_final_val=True)

    def configure_optimizers(
        self,
    ) -> Optimizer | OptimizerLRSchedulerConfig:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.cosine_anneal:
            assert isinstance(self.trainer.estimated_stepping_batches, int)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.trainer.estimated_stepping_batches
                    ),
                    "interval": "step",
                },
            }
        return optimizer


def main() -> None:
    L.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser(
        description="Distill Caduceus model using soft labels"
    )
    parser.add_argument(
        "zarr_path_train",
        type=str,
        help="Path to the train split Zarr store with soft labels",
    )
    parser.add_argument(
        "zarr_path_val",
        type=str,
        help="Path to the validation split Zarr store with soft labels",
    )
    parser.add_argument("--max_epoch", type=int, default=1, help="Trainer max epochs")
    parser.add_argument(
        "--max_train_batches",
        type=int,
        # NOTE: this is based on the scaling laws. The teacher was trained on 400k batches, student is about
        # 15% of the teacher --> 60k batches.
        default=60_000,
        help="Limit train batches per epoch (defaults to all)",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=128,
        help="Limit validation batches during training (defaults to 128)",
    )
    parser.add_argument(
        "--max_final_val_batches",
        type=int,
        default=1024,
        help="Limit final validation batches (defaults to 1024).",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=50,
        help="Used to compute validation schedule, validation schedule is logarithmic with this interval",
    )
    parser.add_argument(
        "--val_log_interval_sampling",
        action="store_true",
        help="If true validation will log linearly in log space",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--temperature", type=float, default=4.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Weight for distillation soft targets loss (1 - alpha for hard targets loss)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="caduceus_distill",
        help="W&B project name",
    )
    parser.add_argument(
        "--run_name_suffix",
        type=str,
        default="",
        help="Optional suffix to append to run name",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over (default: 1, no accumulation)",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--cosine_anneal",
        action="store_true",
        help="Use cosine annealing for learning rate scheduling",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default="cadu-distill",
        help="GCS bucket name (e.g. for checkpoints)",
    )

    args = parser.parse_args()
    gcs_bucket = args.gcs_bucket

    # Initialize datasets
    train_dataset = DistillationDataset(args.zarr_path_train)
    val_dataset = DistillationDataset(args.zarr_path_val)

    # Create data loaders
    train_loader, val_loader, test_loader = [
        DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=True if args.num_workers > 0 else False,
            pin_memory=True,
        )
        for ds in [train_dataset, val_dataset, val_dataset]
    ]

    # Initialize model
    model = StudentCaduceus(
        lr=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
        cosine_anneal=args.cosine_anneal,
    )

    # Setup logger
    wandb_logger: WandbLogger | None = None

    datetime_str = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    run_name_parts = [datetime_str]
    if args.run_name_suffix:
        run_name_parts.append(args.run_name_suffix)
    full_run_name = "_".join(run_name_parts)
    logger.info(f"Run name: {full_run_name}")

    if not args.no_wandb:
        wandb_logger = WandbLogger(project=args.project_name, name=full_run_name)

    try:
        from gcsfs import GCSFileSystem

        fs: GCSFileSystem = fsspec.filesystem("gs")
        assert len(fs.info(f"gs://{gcs_bucket}")) > 0
        checkpoint_dirpath = f"gs://{gcs_bucket}/checkpoints/{full_run_name}/"
    except Exception:
        logger.exception(
            "Failed to probe GCS, will use local filesystem for checkpoints."
        )
        checkpoint_dirpath = f"checkpoints/{full_run_name}/"

    logger.info(f"Checkpoint directory: {checkpoint_dirpath}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename="student-caduceus__epoch={epoch:02d}__val_loss_total={val/loss/total:.3f}__step={step}",
        # NOTE: because the metric contains slashes
        auto_insert_metric_name=False,
        monitor="val/loss/total",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Dynamically set precision based on hardware support.
    precision: Literal["bf16-mixed", "16-mixed", "32-true"]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():  # type: ignore[no-untyped-call]
        precision = "bf16-mixed"
        print("INFO: GPU supports bfloat16. Using 'bf16-mixed' precision.")
    elif torch.cuda.is_available():
        precision = "16-mixed"
        print(
            "INFO: GPU does not support bfloat16. Falling back to '16-mixed' precision."
        )
    else:
        precision = "32-true"
        print("INFO: No GPU available. Using '32-true' precision on CPU.")

    # Compute the validation schedule
    if args.val_log_interval_sampling:
        max_train_batches = args.max_train_batches
        accumulate_grad_batches = args.accumulate_grad_batches
        val_check_interval = args.val_check_interval
        max_global_step = int(np.ceil(max_train_batches / accumulate_grad_batches))
        val_schedule = set(
            np.logspace(
                0,
                np.log10(max_global_step - 1),
                num=max_global_step // val_check_interval,
                dtype=int,
            )
        )
    else:
        # Linear schedule
        max_train_batches = args.max_train_batches
        accumulate_grad_batches = args.accumulate_grad_batches
        val_check_interval = args.val_check_interval
        max_global_step = int(np.ceil(max_train_batches / accumulate_grad_batches))
        val_schedule = set(range(0, max_global_step, val_check_interval))

    logger.info(f"Validation schedule: {sorted(val_schedule)}")
    validation_scheduler = ValidationScheduler(val_schedule)
    gradient_norm_logger = GradientNormLogger(val_schedule)

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epoch,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            validation_scheduler,
            gradient_norm_logger,
        ],
        accelerator="auto",
        devices="auto",
        limit_train_batches=max_train_batches,
        # Validation settings
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=1,
        limit_val_batches=args.max_val_batches,
        limit_test_batches=args.max_final_val_batches,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    if wandb_logger is not None:
        wandb_logger.log_hyperparams(
            {
                "batch_size": args.batch_size,
                "accumulate_grad_batches": accumulate_grad_batches,
                "learning_rate": args.lr,
                "temperature": args.temperature,
                "alpha": args.alpha,
                "max_epochs": args.max_epoch,
                "max_train_batches": max_train_batches,
                "max_val_batches": args.max_val_batches,
                "max_final_val_batches": args.max_final_val_batches,
                "num_workers": args.num_workers,
                "val_check_interval": val_check_interval,
                "val_log_interval_sampling": args.val_log_interval_sampling,
                "precision": precision,
                "cosine_anneal": args.cosine_anneal,
            }
        )

    # Train model
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set start method to 'spawn' to prevent fork-safety issues with
    # multi-threaded libraries (e.g. zarr) in worker processes.
    multiprocessing.set_start_method("spawn")
    main()
