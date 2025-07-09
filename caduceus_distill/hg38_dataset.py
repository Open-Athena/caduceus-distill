"""
This file is taken from https://github.com/kuleshov-group/caduceus/blob/0060a6d8079b6a040fc55d505e15972a327b70a6/src/dataloaders/datasets/hg38_dataset.py
Changes:
 * no RC augmentation
 * char tokenizer only
 * no mlm augmentation
 * no eos token
"""

import math
from pathlib import Path

import pandas as pd
import torch
from pyfaidx import Fasta
from transformers import PreTrainedTokenizerBase

# NOTE: this is the sequence length of the Basenji segments
MAX_ALLOWED_LENGTH = 2**17


HG38_EXAMPLE_T = tuple[torch.Tensor, str, int, int]  # (input_ids, chr_name, start, end)


class HG38Dataset(torch.utils.data.Dataset[HG38_EXAMPLE_T]):
    """Loop through bed file, retrieve (chr, start, end), query fasta file for sequence."""

    def __init__(
        self,
        *,
        split: str,
        bed_file: str | Path,
        fasta_file: str | Path,
        seq_length: int,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.max_length = seq_length
        self.tokenizer = tokenizer

        if seq_length <= MAX_ALLOWED_LENGTH:
            assert (
                MAX_ALLOWED_LENGTH % seq_length == 0
            ), "`max_length` must be a divisor of MAX_ALLOWED_LENGTH/2^17"
            self.n_tiles = MAX_ALLOWED_LENGTH // seq_length
        else:
            raise ValueError(
                f"`max_length` {seq_length} (> 2^{int(math.log(MAX_ALLOWED_LENGTH, 2))}) is too large!"
            )

        bed_path = Path(bed_file)
        assert bed_path.exists(), "Path to .bed file must exist!"
        df_raw = pd.read_csv(
            str(bed_path), sep="\t", names=["chr_name", "start", "end", "split"]
        )

        fasta_file = Path(fasta_file)
        assert (
            fasta_file.exists()
        ), f"Path to fasta file must exist! Given: {fasta_file}"

        self.fasta = Fasta(str(fasta_file))
        self.df = df_raw[df_raw["split"] == split]
        assert not self.df.empty, f"No data found for split '{split}' in {bed_file}"

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}
        for chr_name in self.fasta.keys():
            self.chr_lens[chr_name] = len(self.fasta[chr_name])

    def __len__(self) -> int:
        return len(self.df) * self.n_tiles

    def __getitem__(self, idx: int) -> HG38_EXAMPLE_T:
        """Returns a sequence of specified len"""
        # sample a random row from df
        row_idx, tile_idx = idx // self.n_tiles, idx % self.n_tiles
        row = self.df.iloc[row_idx]
        chr_name, start, end = (row.iloc[0], row.iloc[1], row.iloc[2])

        assert (
            end - start == MAX_ALLOWED_LENGTH
        ), f"Expected {MAX_ALLOWED_LENGTH} length sequence, got {end - start}"

        chromosome = self.fasta[chr_name]
        adjusted_start, adjusted_end = (
            start + tile_idx * self.max_length,
            start + (tile_idx + 1) * self.max_length,
        )
        seq = str(chromosome[adjusted_start:adjusted_end])

        assert (
            adjusted_end <= end
        ), f"Adjusted end {adjusted_end} exceeds original end {end} for chromosome {chr_name}"
        assert (
            adjusted_start >= start
        ), f"Adjusted start {adjusted_start} is less than original start {start} for chromosome {chr_name}"
        assert (
            adjusted_start >= 0
        ), f"Adjusted start {adjusted_start} is negative for chromosome {chr_name}"
        assert adjusted_end <= self.chr_lens[chr_name], (
            f"Adjusted end {adjusted_end} exceeds length of chromosome {chr_name} "
            f"({self.chr_lens[chr_name]})"
        )

        seq = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=False,
        )

        seq = seq["input_ids"]

        assert (
            len(seq) == self.max_length
        ), f"Expected {self.max_length} length sequence, got {len(seq)}. {chr_name=}, {adjusted_start=}, {adjusted_end=}, {tile_idx=}"

        # convert to tensor
        seq_tensor: torch.Tensor = torch.LongTensor(seq)
        del seq

        # replace N token with a pad token, so we can ignore it in the loss
        seq_tensor = torch.where(
            seq_tensor == self.tokenizer._vocab_str_to_int["N"],
            self.tokenizer.pad_token_id,
            seq_tensor,
        )

        return seq_tensor, chr_name, adjusted_start, adjusted_end
