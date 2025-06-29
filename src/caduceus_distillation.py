import argparse
import logging
import multiprocessing
from datetime import UTC, datetime
from typing import Any, Literal

import lightning as L
import torch
import torch.nn.functional as F
import xarray as xr
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
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


class StudentCaduceus(L.LightningModule):
    student: AutoModelForMaskedLM
    temperature: float
    lr: float
    alpha: float

    def __init__(
        self,
        teacher_model_name: str = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        lr: float = 1e-3,
        temperature: float = 4.0,
        alpha: float = 0.8,
    ) -> None:
        super().__init__()
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

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self.student(input_ids)

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
        self.log("train/loss/soft", soft_loss, on_step=True, on_epoch=True)
        self.log("train/loss/hard", hard_loss, on_step=True, on_epoch=True)
        return loss

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.global_step % 100 == 0:
            lr = optimizer.param_groups[0]["lr"]
            for name, param in self.student.named_parameters():
                if param.grad is not None:
                    # L2 Norm of Gradients
                    grad_norm = param.grad.norm(2)
                    self.log(
                        f"diagnostics/train/grad_norm/{name}",
                        grad_norm,
                        on_step=True,
                        on_epoch=False,
                    )

                    # Update-to-Data Ratio
                    data_std = param.data.std()
                    if data_std > 0:
                        update_ratio = lr * param.grad.std() / (data_std + 1e-8)
                        self.log(
                            f"diagnostics/train/update_ratio/{name}",
                            update_ratio,
                            on_step=True,
                            on_epoch=False,
                        )

    def validation_step(self, batch: EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        input_ids, teacher_logits = batch

        outputs = self.student(input_ids)
        student_logits = outputs.logits

        # Calculate combined loss
        loss, soft_loss, hard_loss = distillation_loss(
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

        self.log(
            "val/loss/total",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val/loss/soft", soft_loss, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "val/loss/hard", hard_loss, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


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
        default=None,
        help="Limit train batches per epoch (defaults to all)",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=10,
        help="Limit validation batches during training (defaults to 10). Full dataset is used for final validation.",
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
        "--train_ratio", type=float, default=0.9, help="Train/validation split ratio"
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

    args = parser.parse_args()

    # Initialize datasets
    train_dataset = DistillationDataset(args.zarr_path_train)
    val_dataset = DistillationDataset(args.zarr_path_val)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True,
    )

    # Initialize model
    model = StudentCaduceus(lr=args.lr, temperature=args.temperature, alpha=args.alpha)

    # Setup logger
    logger: WandbLogger | None = None
    if not args.no_wandb:
        datetime_str = datetime.now(UTC).strftime("%Y%m%d_%H%M")

        run_name_parts = [datetime_str]
        if args.run_name_suffix:
            run_name_parts.append(args.run_name_suffix)

        full_run_name = "_".join(run_name_parts)
        logger = WandbLogger(project=args.project_name, name=full_run_name)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="student-caduceus-{epoch:02d}-{val_loss:.2f}",
        monitor="val/loss/total",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

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

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epoch,
        precision=precision,
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
        limit_train_batches=args.max_train_batches,
        # Validation settings
        val_check_interval=128,
        check_val_every_n_epoch=1,
        limit_val_batches=args.max_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)
    # TODO: is there a prettier way to tell the trainer to ignore `limit_val_batches`?
    trainer.limit_val_batches = 1.0
    trainer.validate(model, val_loader)


if __name__ == "__main__":
    # Set start method to 'spawn' to prevent fork-safety issues with
    # multi-threaded libraries (e.g. zarr) in worker processes.
    multiprocessing.set_start_method("spawn")
    main()
