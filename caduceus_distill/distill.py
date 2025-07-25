import logging
import multiprocessing
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal

import fsspec
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import typer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import (
    Optimizer,
    OptimizerLRSchedulerConfig,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from caduceus_distill.data.hg38_dataset import HG38_EXAMPLE_T, HG38Dataset
from caduceus_distill.utils.utils import get_root_path, setup_basic_logging

CADUCEUS_PAD_TOKEN_ID = 4

logger = logging.getLogger(__name__)


def _filter_non_specific_nucleotides_and_batch(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function filters out the examples where the label is the non-specific nucleotide (PAD token).
    """
    mask = input_ids != CADUCEUS_PAD_TOKEN_ID
    # 2d mask is going to collapses some dimensions
    student_logits = student_logits[mask]
    teacher_logits = teacher_logits[mask]
    input_ids = input_ids[mask]
    student_emb = student_emb[mask]
    teacher_emb = teacher_emb[mask]

    assert student_logits.ndim == 2
    assert teacher_logits.ndim == 2
    assert student_emb.ndim == 2
    assert teacher_emb.ndim == 2
    assert input_ids.ndim == 1

    return student_logits, teacher_logits, student_emb, teacher_emb, input_ids


def distillation_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float,
    alpha_soft: float,
    alpha_sim: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate combined distillation loss.

    Parameters:
        student_logits: Raw logits from student model [B, T, V]
        teacher_logits: Raw logits from teacher model [B, T, V]
        student_emb: Hidden states from student model [B, T, D]
        teacher_emb: Hidden states from teacher model [B, T, D]
        input_ids: Ground truth token ids (hard targets) [B, T]
        temperature: Temperature for softening probability distributions
        alpha_soft: Weight for soft targets loss (KL divergence)
        alpha_sim: Weight for hidden state similarity loss (cosine embedding loss)
    """
    assert temperature > 0, "Temperature must be positive"
    assert 0 <= alpha_soft <= 1, "alpha_soft must be in [0, 1]"
    alpha_hard = 1.0 - alpha_soft

    # NOTE: we could easily make this function work on any number of dimensions, but for simplicity and to make it
    # fail quick if the shapes are wrong, we assume 3d input [B, T, V] for logits and [B, T] for input_ids.
    assert (
        student_logits.ndim == 3
    ), f"Expected student_logits to be 3D, got {student_logits.ndim}D"

    assert (
        teacher_logits.ndim == 3
    ), f"Expected teacher_logits to be 3D, got {teacher_logits.ndim}D"
    assert input_ids.ndim == 2, f"Expected input_ids to be 2D, got {input_ids.ndim}D"

    # Expect B, T, D
    assert (
        student_emb.ndim == 3
    ), f"Expected student_emb to be 3D, got {student_emb.ndim}D"
    assert (
        teacher_emb.ndim == 3
    ), f"Expected teacher_emb to be 3D, got {teacher_emb.ndim}D"

    student_logits, teacher_logits, student_emb, teacher_emb, input_ids = (
        _filter_non_specific_nucleotides_and_batch(
            student_logits, teacher_logits, student_emb, teacher_emb, input_ids
        )
    )

    assert (
        input_ids.size(0) > 0
    ), "Input IDs must not be empty after filtering non-specific nucleotides"

    # TODO: can we depend on the Caduceus tokennizer and get these indexes from there?
    # Useful classes are: A, C, G, T and N (except N is converted to PAD token)
    useful_class_idx = [CADUCEUS_PAD_TOKEN_ID, 10, 9, 8, 7]
    # mask out the V dimension to only consider useful classes
    valid_cls_mask = torch.full(
        (student_logits.size(-1),), device=student_logits.device, fill_value=False
    )
    valid_cls_mask[useful_class_idx] = True
    # NOTE: this impact the cross_entropy below, but not the KL divergence since we nuke
    # the logits for non-useful classes there.
    student_logits = student_logits.masked_fill(~valid_cls_mask, float("-inf"))

    # Soft loss (distillation)
    teacher_log_probs = F.log_softmax(
        teacher_logits[:, valid_cls_mask] / temperature, dim=-1
    )
    student_log_probs = F.log_softmax(
        student_logits[:, valid_cls_mask] / temperature, dim=-1
    )

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
    if alpha_soft != 1.0:
        soft_loss *= temperature**2

    # Hard loss (cross-entropy)
    hard_loss = F.cross_entropy(student_logits, input_ids)

    soft_loss_contrib = alpha_soft * soft_loss
    hard_loss_contrib = alpha_hard * hard_loss

    hidden_state_sim = F.cosine_embedding_loss(
        student_emb,
        teacher_emb,
        torch.ones(
            student_emb.size(0),
            device=student_emb.device,
        ),
        reduction="mean",
    )
    hidden_state_sim = alpha_sim * hidden_state_sim

    total_loss = soft_loss_contrib + hard_loss_contrib + hidden_state_sim
    return total_loss, soft_loss_contrib, hard_loss_contrib, hidden_state_sim


class DistillationDataset(Dataset[HG38_EXAMPLE_T]):
    def __init__(
        self,
        bed_file: str,
        fasta_file: str,
        tokenizer: PreTrainedTokenizerBase,
        seq_length: int = 2**17,
        split: str = "train",
        skip_batches: set[int] | None = None,
        start_idx: int = 0,
        device: str = "cpu",
    ) -> None:
        self.hg38_ds: HG38Dataset | None = None
        self.split = split
        self.bed_file = bed_file
        self.fasta_file = fasta_file
        self.seq_length = seq_length
        self.device = device
        self.tokenizer = tokenizer

        self.skip_batches = skip_batches if skip_batches is not None else set()
        self._in_memory_cache: dict[int, HG38_EXAMPLE_T] = {}
        self.start_idx = start_idx

        self._len = len(self.__maybe_open_ds())

    def __maybe_open_ds(self) -> HG38Dataset:
        if self.hg38_ds is None:
            hg38_ds = HG38Dataset(
                split=self.split,
                bed_file=self.bed_file,
                fasta_file=self.fasta_file,
                seq_length=self.seq_length,
                tokenizer=self.tokenizer,
            )
        else:
            hg38_ds = self.hg38_ds
        return hg38_ds

    def __len__(self) -> int:
        return self._len

    def warmup(self, n: int) -> None:
        """
        Pre-load the first `n` samples into memory to speed up fetching, this is useful for
        intermediate validation with limited number of batches.
        """
        for i in range(n):
            self._in_memory_cache[i] = self[i]

    def __getitem__(self, idx: int) -> HG38_EXAMPLE_T:
        self.hg38_ds = self.__maybe_open_ds()
        idx = idx + self.start_idx

        if idx in self._in_memory_cache:
            return self._in_memory_cache[idx]

        # TODO: better doc, is there an idiomatic way to skip a batch?
        if idx in self.skip_batches:
            # NOTE: to keep things easy and safe - for now just use the 1st batch
            new_idx = np.random.randint(0, len(self))
            logger.debug(f"Using batch {new_idx} instead of {idx}")
            idx = new_idx

        input_ids, chr_name, start, end = self.hg38_ds[idx]

        if input_ids.eq(CADUCEUS_PAD_TOKEN_ID).all():
            raise ValueError(
                f"Sample {idx} contains only PAD tokens. This is not allowed."
            )

        return input_ids, chr_name, start, end

    def __getstate__(self) -> dict[str, Any]:
        r = super().__getstate__()
        assert isinstance(r, dict)
        # NOTE: never pickle the dataset
        r["hg38_ds"] = None
        return r

    # TODO: add `load_state_dict` and `state_dict` methods to support pickling/checkpointing


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
                weight_norm = param.data.norm(2)
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
                        f"{metric_prefix}/weight_norm": weight_norm,
                    }
                )


class StudentCaduceus(L.LightningModule):
    student: AutoModelForMaskedLM
    temperature: float
    lr: float
    alpha_soft: float
    alpha_sim: float
    cosine_anneal: bool

    def __init__(
        self,
        *,
        teacher_model_name: str,
        lr: float = 1e-3,
        temperature: float = 4.0,
        alpha_soft: float = 0.8,
        alpha_sim: float = 0.5,
        cosine_anneal: bool = False,
        preload_teacher: bool = False,
        # NOTE: default to half of the teacher's depth and width
        n_layer: int = 8,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.cosine_anneal = cosine_anneal
        self.n_layer = n_layer
        self.d_model = d_model
        self.save_hyperparameters()

        # Create student config (half depth and width)
        student_config = AutoConfig.from_pretrained(
            teacher_model_name,
            trust_remote_code=True,
            d_model=self.d_model,
            n_layer=self.n_layer,
        )

        # Verify student config parameters
        assert (
            student_config.d_model == self.d_model
        ), f"Expected d_model={self.d_model}, got {student_config.d_model}"
        assert (
            student_config.n_layer == self.n_layer
        ), f"Expected n_layer={self.n_layer}, got {student_config.n_layer}"

        self.student = AutoModelForMaskedLM.from_config(
            student_config, trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name, trust_remote_code=True
        )
        if preload_teacher:
            self.teacher = AutoModelForMaskedLM.from_pretrained(
                teacher_model_name, trust_remote_code=True
            )
        else:
            self.teacher = None

        self.temperature = temperature
        self.lr = lr
        self.alpha_soft = alpha_soft
        self.alpha_sim = alpha_sim

    def forward(
        self, input_ids: torch.Tensor, output_hidden_states: bool = False
    ) -> Any:
        return self.student(input_ids, output_hidden_states=output_hidden_states)

    def training_step(self, batch: HG38_EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        input_ids, chr_names, starts, ends = batch
        logger.debug(f"Train {batch_idx=}, {chr_names=}, {starts=}, {ends=}")

        with torch.no_grad():
            outputs = self.teacher(input_ids.to(self.device), output_hidden_states=True)
            teacher_logits = outputs.logits
            teacher_emb = outputs.hidden_states[-1]

        outputs = self.student(input_ids, output_hidden_states=True)
        student_logits = outputs.logits
        student_emb = outputs.hidden_states[-1]

        # Calculate combined loss
        loss, soft_loss, hard_loss, hidden_state_sim_loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_emb=student_emb,
            teacher_emb=teacher_emb,
            input_ids=input_ids,
            temperature=self.temperature,
            alpha_soft=self.alpha_soft,
            alpha_sim=self.alpha_sim,
        )

        self.log("train/loss/total", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(
            {
                "train/loss/soft": soft_loss,
                "train/loss/hard": hard_loss,
                "train/loss/hidden_state_sim": hidden_state_sim_loss,
            },
            on_step=True,
            on_epoch=True,
        )
        return loss

    def _base_validation_step(
        self,
        batch: HG38_EXAMPLE_T,
        batch_idx: int,
        is_final_val: bool,
    ) -> torch.Tensor:
        input_ids, chr_names, starts, ends = batch
        logger.debug(f"Validation {batch_idx=}, {chr_names=}, {starts=}, {ends=}")

        outputs = self.student(input_ids, output_hidden_states=True)
        student_logits = outputs.logits
        student_emb = outputs.hidden_states[-1]
        with torch.no_grad():
            outputs = self.teacher(input_ids.to(self.device), output_hidden_states=True)
            teacher_logits = outputs.logits
            teacher_emb = outputs.hidden_states[-1]

        loss_eval, soft_loss_eval, hard_loss_eval, hidden_state_sim_eval = (
            distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_emb=student_emb,
                teacher_emb=teacher_emb,
                input_ids=input_ids,
                # The `temp` hyper-parameter should not affect the eval scoring
                # Also, temp should be set to 1.0 after the distillation is complete (per Hinton)
                temperature=1.0,
                # The `alpha` hyper-parameter should not affect the eval scoring
                # alpha=1.0 means that we only consider the soft targets
                alpha_soft=1.0,
                alpha_sim=0.0,
            )
        )

        # Evaluation loss with training hyperparameters for comparison
        (
            loss_train_temp,
            soft_loss_train_temp,
            hard_loss_train_temp,
            hidden_state_sim_train_temp,
        ) = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_emb=student_emb,
            teacher_emb=teacher_emb,
            input_ids=input_ids,
            temperature=self.temperature,
            alpha_soft=self.alpha_soft,
            alpha_sim=self.alpha_sim,
        )

        # Use a different metric prefix for the final, post-fit validation run.
        prefix = "val" if not is_final_val else "final_val"

        self.log(
            f"{prefix}/loss/total", loss_eval, prog_bar=not is_final_val, sync_dist=True
        )

        metrics_to_log = {
            f"{prefix}/loss/soft": soft_loss_eval,
            f"{prefix}/loss/hard": hard_loss_eval,
            f"{prefix}/loss/hidden_state_sim": hidden_state_sim_eval,
            f"{prefix}/loss_train_temp/total": loss_train_temp,
            f"{prefix}/loss_train_temp/soft": soft_loss_train_temp,
            f"{prefix}/loss_train_temp/hard": hard_loss_train_temp,
            f"{prefix}/loss_train_temp/hidden_state_sim": hidden_state_sim_train_temp,
        }
        self.log_dict(metrics_to_log, sync_dist=True)
        return loss_eval

    def validation_step(self, batch: HG38_EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        return self._base_validation_step(batch, batch_idx, is_final_val=False)

    def _run_nt_evaluation(self) -> None:
        from caduceus_distill.nt_eval import create_features, run_modeling

        assert (
            not self.training
        ), "The model is expected to be in eval mode for NT evaluation"

        features = create_features(
            model=self,
            tokenizer=self.tokenizer,
            task_group="eric_relevant",
            chunk_size=100,
            sample_limit=10_000,
        )

        results = run_modeling(features, gbrt_only=True)
        results = results.query("split == 'test' & metric in ('roc_auc', 'f1')")

        metrics_to_log: dict[str, float] = {}
        for r in results.itertuples():
            metrics_to_log[f"nt_eval/{r.task_name}/{r.metric}"] = (
                r.value  # type:ignore[assignment]
            )

        self.log_dict(metrics_to_log)

    def on_validation_epoch_start(self) -> None:
        return self._run_nt_evaluation()

    def on_test_epoch_start(self) -> None:
        return self._run_nt_evaluation()

    def test_step(self, batch: HG38_EXAMPLE_T, batch_idx: int) -> torch.Tensor:
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


def main(
    bed_file: Annotated[
        str | None,
        typer.Argument(
            help="Path to the hg38 BED file. Default to data in `data/hg38/`."
        ),
    ] = None,
    fasta_file: Annotated[
        str | None,
        typer.Argument(
            help="Path to the hg38 FASTA file. Default to data in `data/hg38/`."
        ),
    ] = None,
    seq_length: Annotated[
        int,
        typer.Option(
            help="Nucleotide sequence length",
        ),
    ] = 2
    ** 17,
    n_layer: Annotated[
        int, typer.Option(help="Number of layers in the student model", min=1)
    ] = 8,
    d_model: Annotated[
        int, typer.Option(help="Model dimension (d_model) in the student model", min=1)
    ] = 128,
    max_epoch: Annotated[int, typer.Option(help="Trainer max epochs", min=1)] = 1,
    max_train_batches: Annotated[
        int, typer.Option(help="Limit train batches per epoch (defaults to all)", min=1)
    ] = 60_000,
    max_val_batches: Annotated[
        int,
        typer.Option(
            help="Limit validation batches during training (defaults to 128)", min=1
        ),
    ] = 128,
    max_final_val_batches: Annotated[
        int,
        typer.Option(help="Limit final validation batches (defaults to 1024)", min=1),
    ] = 1024,
    val_check_interval: Annotated[
        int,
        typer.Option(
            help="Used to compute validation schedule, validation schedule is logarithmic with this interval",
            min=1,
        ),
    ] = 50,
    val_log_interval_sampling: Annotated[
        bool,
        typer.Option(
            "--val-log-interval-sampling",
            help="Validate linearly in log space",
        ),
    ] = False,
    batch_size: Annotated[int, typer.Option(help="Batch size", min=1)] = 1,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    temperature: Annotated[float, typer.Option(help="Distillation temperature")] = 4.0,
    alpha_soft: Annotated[
        float,
        typer.Option(
            help="Weight for distillation soft targets loss, must be between in [0, 1]",
        ),
    ] = 0.8,
    alpha_sim: Annotated[
        float,
        typer.Option(
            help="Weight for distillation hidden state similarity loss",
        ),
    ] = 1.0,
    num_workers: Annotated[
        int, typer.Option(help="Number of data loading workers")
    ] = 4,
    project_name: Annotated[
        str, typer.Option(help="W&B project name")
    ] = "caduceus_distill",
    run_name_suffix: Annotated[
        str, typer.Option(help="Optional suffix to append to run name")
    ] = "",
    accumulate_grad_batches: Annotated[
        int,
        typer.Option(
            help="Number of batches to accumulate gradients over (default: 1, no accumulation)",
        ),
    ] = 1,
    no_wandb: Annotated[
        bool, typer.Option("--no-wandb", help="Disable W&B logging")
    ] = False,
    wandb_experiment_id: Annotated[
        str | None,
        typer.Option(
            "--wandb-experiment-id",
            help="W&B experiment ID to resume from, if not provided, a new run will be created",
        ),
    ] = None,
    cosine_anneal: Annotated[
        bool,
        typer.Option(
            "--cosine_anneal", help="Use cosine annealing for learning rate scheduling"
        ),
    ] = False,
    checkpoint_dirpath: Annotated[
        str | None,
        typer.Option(
            help="Path to the checkpoint directory, this can be local or ffspec compatible path"
        ),
    ] = None,
    ckpt_to_resume: Annotated[
        str | None,
        typer.Option(
            help="Checkpoint to resume from, if not provided, the latest checkpoint will be used"
        ),
    ] = None,
    teacher_model_name: Annotated[
        str,
        typer.Option(
            help="Name of the teacher model to use for distillation, defaults to Caduceus model",
        ),
    ] = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
) -> None:
    L.seed_everything(42, workers=True)

    bed_file = (
        bed_file
        or get_root_path().joinpath("data", "hg38", "human-sequences.bed").as_posix()
    )
    fasta_file = (
        fasta_file or get_root_path().joinpath("data", "hg38", "hg38.ml.fa").as_posix()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if ckpt_to_resume is not None:
        with fsspec.open(ckpt_to_resume) as fd:
            ckpt_state = torch.load(fd, map_location="cpu")
        resume_global_step = ckpt_state["global_step"]
        logger.info(
            f"Resuming from checkpoint: {ckpt_to_resume}, {resume_global_step=}"
        )
    else:
        resume_global_step = 0

    # NOTE: skip due to https://github.com/Open-Athena/caduceus-distill/issues/38
    train_dataset = DistillationDataset(
        bed_file=bed_file,
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        split="train",
        skip_batches={8590},
        device="cuda" if torch.cuda.is_available() else "cpu",
        seq_length=seq_length,
        start_idx=resume_global_step * accumulate_grad_batches,
    )
    val_dataset = DistillationDataset(
        bed_file=bed_file,
        fasta_file=fasta_file,
        tokenizer=tokenizer,
        split="valid",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seq_length=seq_length,
    )
    # TODO: is this always safe to call? What if `max_val_batches` is large?
    val_dataset.warmup(max_val_batches)

    # Create data loaders
    train_loader, val_loader, test_loader = [
        DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
        )
        for ds in [train_dataset, val_dataset, val_dataset]
    ]

    # Initialize model
    model = StudentCaduceus(
        teacher_model_name=teacher_model_name,
        n_layer=n_layer,
        d_model=d_model,
        preload_teacher=True,
        lr=lr,
        temperature=temperature,
        alpha_soft=alpha_soft,
        alpha_sim=alpha_sim,
        cosine_anneal=cosine_anneal,
    )

    # Setup logger
    wandb_logger: WandbLogger | None = None

    datetime_str = datetime.now(UTC).strftime("%Y%m%d_%H%M")
    run_name_parts = [datetime_str]
    if run_name_suffix:
        run_name_parts.append(run_name_suffix)
    full_run_name = "_".join(run_name_parts)
    logger.info(f"Run name: {full_run_name}")

    if not no_wandb:
        wandb_logger = WandbLogger(
            project=project_name,
            name=full_run_name,
            id=wandb_experiment_id,
            resume="must" if wandb_experiment_id else None,
        )

    # TODO: consider what should happen if we resume from a checkpoint? ATM it will create a new checkpoint directory
    if checkpoint_dirpath is not None:
        checkpoint_dirpath = f"{checkpoint_dirpath}/{full_run_name}/"
    else:
        checkpoint_dirpath = f"checkpoints/{full_run_name}/"
        Path(checkpoint_dirpath).mkdir(parents=True, exist_ok=True)

    logger.info(f"Using checkpoint directory: {checkpoint_dirpath}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        filename="student-caduceus__epoch={epoch:02d}__val_loss_total={val/loss/total:.3f}__step={step}",
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
    if val_log_interval_sampling:
        max_global_step = (
            int(np.ceil(max_train_batches / accumulate_grad_batches)) * max_epoch
        )
        validation_interval = max_global_step // val_check_interval
        val_schedule = set(
            np.logspace(
                validation_interval,
                np.log10(max_global_step - 1),
                num=max_global_step // val_check_interval,
                dtype=int,
            )
        )
    else:
        max_global_step = (
            int(np.ceil(max_train_batches / accumulate_grad_batches)) * max_epoch
        )
        val_schedule = set(
            range(val_check_interval, max_global_step, val_check_interval)
        )

    logger.info(f"Validation schedule: {sorted(val_schedule)}")
    validation_scheduler = ValidationScheduler(val_schedule)
    gradient_norm_logger = GradientNormLogger(val_schedule)

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=max_epoch,
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
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=1,
        limit_val_batches=max_val_batches,
        limit_test_batches=max_final_val_batches,
        accumulate_grad_batches=accumulate_grad_batches,
        # TODO: bring back default 2?
        num_sanity_val_steps=0,
    )
    if wandb_logger is not None:
        wandb_logger.log_hyperparams(
            {
                "teacher_model_name": teacher_model_name,
                "n_layer": n_layer,
                "d_model": d_model,
                "bed_file": bed_file,
                "fasta_file": fasta_file,
                "seq_length": seq_length,
                "batch_size": batch_size,
                "accumulate_grad_batches": accumulate_grad_batches,
                "learning_rate": lr,
                "temperature": temperature,
                "alpha_soft": alpha_soft,
                "alpha_sim": alpha_sim,
                "max_epochs": max_epoch,
                "max_train_batches": max_train_batches,
                "max_val_batches": max_val_batches,
                "max_final_val_batches": max_final_val_batches,
                "num_workers": num_workers,
                "val_check_interval": val_check_interval,
                "val_log_interval_sampling": val_log_interval_sampling,
                "precision": precision,
                "cosine_anneal": cosine_anneal,
            }
        )
        wandb_logger.watch(model=model, log="all", log_freq=val_check_interval)

    # Train model
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_to_resume)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    setup_basic_logging()
    # Set start method to 'spawn' to prevent fork-safety issues with
    # multi-threaded libraries (e.g. zarr) in worker processes.
    multiprocessing.set_start_method("spawn")
    typer.run(main)
