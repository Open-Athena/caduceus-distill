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
    temperature: float = 4.0,
    alpha: float = 0.8,
) -> torch.Tensor:
    """
    Calculate combined distillation loss (KL divergence with temperature scaling + cross-entropy with hard targets).

    Args:
        student_logits: Raw logits from student model
        teacher_logits: Raw logits from teacher model
        input_ids: Ground truth token ids (hard targets)
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss (1-alpha for hard loss)

    Returns:
        Weighted sum of distillation and hard target loss
    """
    # Soft loss (distillation)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    # NOTE: re T^2 scaling, from `Distilling the Knowledge in a Neural Network`:
    # > Since the magnitudes of the gradients p roduced by the soft targets scale as 1/T 2 it is important
    # > to multiply them by T2 when using both hard and soft targets.  This ensures that the relative contributions
    # > of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed
    # > while experimenting with meta-parameters.
    # NOTE: `batchmean` is required per pytorch docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
        temperature**2
    )

    # Hard loss (cross-entropy)
    B, T, V = student_logits.shape
    student_logits_flat = student_logits.view(B * T, V)
    input_ids_flat = input_ids.view(B * T)
    hard_loss = F.cross_entropy(student_logits_flat, input_ids_flat)

    return alpha * soft_loss + (1 - alpha) * hard_loss


EXAMPLE_T = tuple[torch.Tensor, torch.Tensor]


class DistillationDataset(Dataset[EXAMPLE_T]):
    ds: xr.Dataset

    def __init__(
        self,
        zarr_path: str,
        split: Literal["train", "valid"] = "train",
        train_ratio: float = 0.9,
    ) -> None:
        self.ds = xr.open_zarr(zarr_path, chunks=None)

        # Split data into train/valid
        total_samples = len(self.ds.sample)
        train_size = int(total_samples * train_ratio)

        if split == "train":
            self.ds = self.ds.isel(sample=slice(0, train_size))
        elif split == "valid":
            self.ds = self.ds.isel(sample=slice(train_size, None))
        else:
            raise ValueError("split must be 'train' or 'valid'")

    def __len__(self) -> int:
        return len(self.ds.sample)

    def __getitem__(self, idx: int) -> EXAMPLE_T:
        sample = self.ds.isel(sample=idx)
        input_ids = torch.tensor(sample.input_ids.values, dtype=torch.long)
        teacher_logits = torch.tensor(sample.logits.values, dtype=torch.float32)
        return input_ids, teacher_logits


class StudentCaduceus(L.LightningModule):
    student: AutoModelForMaskedLM
    temperature: float
    lr: float

    def __init__(
        self,
        teacher_model_name: str = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
        lr: float = 1e-3,
        temperature: float = 4.0,
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

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self.student(input_ids)

    def training_step(self, batch: EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        input_ids, teacher_logits = batch

        outputs = self.student(input_ids)
        student_logits = outputs.logits

        # Calculate combined loss
        loss = distillation_loss(
            student_logits, teacher_logits, input_ids, self.temperature
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: EXAMPLE_T, batch_idx: int) -> torch.Tensor:
        input_ids, teacher_logits = batch

        outputs = self.student(input_ids)
        student_logits = outputs.logits

        # Calculate combined loss
        loss = distillation_loss(
            student_logits, teacher_logits, input_ids, self.temperature
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill Caduceus model using soft labels"
    )
    parser.add_argument(
        "zarr_path", type=str, help="Path to Zarr file with soft labels"
    )
    parser.add_argument("--max_epoch", type=int, default=1, help="Trainer max epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--temperature", type=float, default=4.0, help="Distillation temperature"
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
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Initialize model
    model = StudentCaduceus(lr=args.lr, temperature=args.temperature)

    # Setup logger
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

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epoch,
        # NOTE: bf16 is not supported on T4
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices="auto",
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
