import argparse
import os
import time
from typing import Any, Literal

import torch
import xarray as xr
from pyfaidx import Fasta
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizer

DATASET_ITEM_T = tuple[torch.Tensor, str, int, int]


class FastaDataset(Dataset[DATASET_ITEM_T]):
    fasta: Fasta
    chunk_size: int
    tokenizer: PreTrainedTokenizer
    chunks: list[tuple[str, int, int]]

    def __init__(
        self,
        fasta_file: str | os.PathLike[str],
        chunk_size: int = 131072,
        tokenizer: PreTrainedTokenizer | None = None,
    ) -> None:
        self.fasta = Fasta(str(fasta_file))
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer

        self.chunks = []
        for chr_name in self.fasta.keys():
            chr_len = len(self.fasta[chr_name])
            for start in range(0, chr_len, chunk_size):
                end = min(start + chunk_size, chr_len)
                if end - start >= chunk_size // 2:
                    self.chunks.append((chr_name, start, end))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> DATASET_ITEM_T:
        chr_name, start, end = self.chunks[idx]
        seq = str(self.fasta[chr_name][start:end])

        if len(seq) < self.chunk_size:
            seq = seq + "N" * (self.chunk_size - len(seq))
        else:
            seq = seq[: self.chunk_size]

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided")

        tokens = self.tokenizer(seq, return_tensors="pt", add_special_tokens=False)
        assert tokens.input_ids.shape == (1, self.chunk_size)
        return tokens.input_ids.squeeze(0), chr_name, start, end


def generate_soft_labels(
    fasta_file: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    chunk_size: int = 131072,
    batch_size: int = 1,
    device: Literal["cuda", "cpu"] = "cuda",
) -> None:
    model_name: str = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(
        device
    )

    dataset: FastaDataset = FastaDataset(
        fasta_file, chunk_size=chunk_size, tokenizer=tokenizer
    )

    # Allow the CPU to prepare batches in the background while the GPU is processing.
    num_workers: int = min(os.cpu_count() or 1, 8)
    dataloader: DataLoader[DATASET_ITEM_T] = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
    )

    # Initialize NVML for GPU monitoring conditionally
    nvml_available: bool = False
    handle: Any = None
    pynvml: Any = None

    if device == "cuda":
        try:
            import pynvml  # type: ignore[no-redef]

            pynvml.nvmlInit()

            # Assert that there is exactly one GPU
            device_count: int = pynvml.nvmlDeviceGetCount()
            assert device_count == 1, (
                f"Expected 1 GPU, but found {device_count}. "
                "This script is designed for single-GPU execution only."
            )

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            nvml_available = True
        except (ImportError, Exception):
            print(
                "Warning: pynvml library not found or NVIDIA driver/NVML issue. "
                "GPU metrics will not be reported."
            )

    # Initialize tracking variables
    last_print_time: float = time.time()
    total_tokens: int = 0
    tokens_since_last_print: int = 0
    total_batches: int = len(dataloader)
    start_time: float = time.time()
    processed_batches: int = 0
    was_interrupted: bool = False

    # Initialize variables for aggregating GPU stats
    sum_gpu_core_util: int = 0
    sum_gpu_mem_io_util: int = 0
    gpu_samples_count: int = 0

    try:
        for batch_idx, (input_ids, chr_names, starts, ends) in enumerate(dataloader):
            processed_batches = batch_idx + 1
            input_ids = input_ids.to(device)
            batch_tokens: int = input_ids.numel()
            tokens_since_last_print += batch_tokens
            total_tokens += batch_tokens

            # Use autocast for FP16 mixed-precision inference to boost performance on T4 GPUs.
            with torch.inference_mode(), autocast(enabled=(device == "cuda")):
                outputs = model(input_ids)
                logits = outputs.logits
                # For numerical stability, cast logits to float32 before softmax.
                probabilities = torch.softmax(logits.float(), dim=-1)

            # Sample GPU metrics after the main workload of the batch
            if nvml_available and pynvml is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    sum_gpu_core_util += util.gpu
                    sum_gpu_mem_io_util += util.memory
                    gpu_samples_count += 1
                except Exception:
                    pass

            # Calculate global indices for samples in the current batch.
            num_samples_in_batch: int = input_ids.shape[0]
            start_sample_idx: int = batch_idx * batch_size
            batch_indices: list[int] = list(
                range(start_sample_idx, start_sample_idx + num_samples_in_batch)
            )

            ds: xr.Dataset = xr.Dataset(
                {
                    "input_ids": (["sample", "sequence"], input_ids.cpu().numpy()),
                    "logits": (["sample", "sequence", "vocab"], logits.cpu().numpy()),
                    "probabilities": (
                        ["sample", "sequence", "vocab"],
                        probabilities.cpu().numpy(),
                    ),
                },
                coords={
                    "sample": batch_indices,
                    "sequence": range(input_ids.shape[1]),
                    "vocab": range(logits.shape[-1]),
                    "chr_name": (["sample"], list(chr_names)),
                    "start": (["sample"], [int(s) for s in starts]),
                    "end": (["sample"], [int(e) for e in ends]),
                },
            )

            zarr_kwargs: dict[str, Any] = {"zarr_format": 2, "consolidated": True}
            if os.path.exists(output_path):
                zarr_kwargs["append_dim"] = "sample"

            ds.to_zarr(output_path, **zarr_kwargs)

            # Log progress and tokens/second every 5 seconds
            current_time: float = time.time()
            if current_time - last_print_time >= 5:
                elapsed_since_last: float = current_time - last_print_time
                tokens_per_second: float = tokens_since_last_print / elapsed_since_last
                progress_percent: float = (batch_idx + 1) / total_batches * 100

                # Get GPU metrics if available
                gpu_metrics_str: str = ""
                if nvml_available and pynvml is not None:
                    try:
                        avg_gpu_core: float = (
                            sum_gpu_core_util / gpu_samples_count
                            if gpu_samples_count > 0
                            else 0
                        )
                        avg_gpu_mem_io: float = (
                            sum_gpu_mem_io_util / gpu_samples_count
                            if gpu_samples_count > 0
                            else 0
                        )

                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        mem_used_mib: int = mem_info.used // (1024**2)
                        mem_total_mib: int = mem_info.total // (1024**2)

                        gpu_metrics_str = (
                            f" | Avg GPU Core: {avg_gpu_core:.1f}%, Avg MemIO: {avg_gpu_mem_io:.1f}%, "
                            f"VRAM: {mem_used_mib}/{mem_total_mib} MiB"
                        )
                    except Exception:
                        gpu_metrics_str = " | GPU Metrics: N/A"

                print(
                    f"Progress: {progress_percent:.2f}% ({batch_idx + 1}/{total_batches} batches) | "
                    f"Speed: {tokens_per_second:.2f} tokens/sec{gpu_metrics_str}"
                )

                # Reset tracking for next interval
                last_print_time = current_time
                tokens_since_last_print = 0
                sum_gpu_core_util = 0
                sum_gpu_mem_io_util = 0
                gpu_samples_count = 0

    except KeyboardInterrupt:
        was_interrupted = True
        print("\nInterrupted by user!")

    except Exception as e:
        was_interrupted = True
        print(f"\nError occurred: {str(e)}")

    finally:
        # Shutdown NVML if it was initialized
        if nvml_available and pynvml is not None:
            pynvml.nvmlShutdown()

        # Calculate statistics for either completion or interruption
        total_elapsed: float = time.time() - start_time

        if processed_batches == 0:
            print("No data was processed!")
        else:
            progress_percent = processed_batches / total_batches * 100
            completion_status: str = (
                "Partial completion" if was_interrupted else "Completed"
            )

            print(
                f"\n{completion_status}: {progress_percent:.2f}% ({processed_batches}/{total_batches} batches) | "
                f"Total tokens processed: {total_tokens} | "
                f"Average speed: {total_tokens/total_elapsed:.2f} tokens/sec | "
                f"Total time: {total_elapsed:.2f}s"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate soft labels from FASTA sequences using Caduceus model"
    )
    parser.add_argument("fasta_file", type=str, help="Path to FASTA file")
    parser.add_argument("output_path", type=str, help="Output Zarr file path")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=131072,
        help="Chunk size for sequences (default: 131072)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (e.g., 'cuda' or 'cpu')",
    )

    args = parser.parse_args()
    generate_soft_labels(
        args.fasta_file, args.output_path, args.chunk_size, args.batch_size, args.device
    )
