import argparse
from typing import Any, Literal

import lightning as L
import torch
import torch.nn.functional as F
import xarray as xr
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForMaskedLM


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    temperature: float = 4.0,
    alpha: float = 0.8,
) -> torch.Tensor:
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

    # Soft loss (distillation)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # NOTE: `batchmean` is required per pytorch docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
    soft_loss = F.kl_div(
        student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True
    )

    # NOTE: re T^2 scaling, from `Distilling the Knowledge in a Neural Network`:
    # > Since the magnitudes of the gradients produced by the soft targets scale as 1/T 2 it is important
    # > to multiply them by T^2 when using both hard and soft targets. This ensures that the relative
    # > contributions of the hard and soft targets remain roughly unchanged if the temperature used for
    # > distillation is changed while experimenting with meta-parameters.
    if alpha != 1.0:
        soft_loss *= temperature**2

    # Hard loss (cross-entropy)
    B, T, V = student_logits.shape
    student_logits_flat = student_logits.view(B * T, V)
    input_ids_flat = input_ids.view(B * T)
    hard_loss = F.cross_entropy(student_logits_flat, input_ids_flat)

    return alpha * soft_loss + (1 - alpha) * hard_loss


EXAMPLE_T = tuple[torch.Tensor, torch.Tensor]


class DistillationDataset(Dataset[EXAMPLE_T]):
    def __init__(
        self,
        zarr_path: str,
        split: Literal["train", "valid"] = "train",
        train_ratio: float = 0.9,
    ) -> None:
        self.zarr_path = zarr_path
        self.split = split
        self.train_ratio = train_ratio

        self.ds: xr.Dataset | None = None

        with xr.open_zarr(self.zarr_path, chunks=None) as temp_ds:
            total_samples = len(temp_ds.sample)
            train_size = int(total_samples * self.train_ratio)

            if self.split == "train":
                self.sample_slice = slice(0, train_size)
                self._len = train_size
            elif self.split == "valid":
                self.sample_slice = slice(train_size, None)
                self._len = total_samples - train_size
            else:
                raise ValueError("split must be 'train' or 'valid'")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> EXAMPLE_T:
        if self.ds is None:
            full_ds = xr.open_zarr(self.zarr_path, chunks=None)
            self.ds = full_ds.isel(sample=self.sample_slice)

        sample = self.ds.isel(sample=idx)
        input_ids = torch.tensor(sample.input_ids.values, dtype=torch.long)
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
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            input_ids,
            temperature=self.temperature,
            alpha=self.alpha,
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        input_ids, teacher_logits = batch

        outputs = self.student(input_ids)
        student_logits = outputs.logits

        # Calculate combined loss
        loss = distillation_loss(
            student_logits,
            teacher_logits,
            input_ids,
            temperature=self.temperature,
            alpha=self.alpha,
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main() -> None:
    L.seed_everything(42, workers=True)

    parser = argparse.ArgumentParser(
        description="Distill Caduceus model using soft labels"
    )
    parser.add_argument(
        "zarr_path", type=str, help="Path to Zarr file with soft labels"
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
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")

    args = parser.parse_args()

    # Initialize datasets
    train_dataset = DistillationDataset(
        args.zarr_path, split="train", train_ratio=args.train_ratio
    )
    val_dataset = DistillationDataset(
        args.zarr_path, split="valid", train_ratio=args.train_ratio
    )

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
        logger = WandbLogger(project=args.project_name, name=args.run_name)

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="student-caduceus-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
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
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)
    # TODO: is there a prettier way to tell the trainer to ignore `limit_val_batches`?
    trainer.limit_val_batches = 1.0
    trainer.validate(model, val_loader)


if __name__ == "__main__":
    main()
