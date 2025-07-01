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

MAX_ALLOWED_LENGTH = 2**20


HG38_EXAMPLE_T = tuple[torch.Tensor, str, int, int]  # (input_ids, chr_name, start, end)


class FastaInterval:
    """Retrieves sequences from a fasta file given a chromosome and start/end indices."""

    def __init__(
        self,
        *,
        fasta_file: str | Path,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), "Path to fasta file must exist!"

        self.seqs = Fasta(str(fasta_file))

        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}

        for chr_name in self.seqs.keys():
            self.chr_lens[chr_name] = len(self.seqs[chr_name])

    @staticmethod
    def _compute_interval(
        start: int, end: int, max_length: int, i_shift: int
    ) -> tuple[int, int]:
        if max_length == MAX_ALLOWED_LENGTH:
            return start, end
        if max_length < MAX_ALLOWED_LENGTH:
            assert MAX_ALLOWED_LENGTH % max_length == 0
            return start + i_shift * max_length, start + (i_shift + 1) * max_length
        else:
            raise ValueError(
                f"`max_length` {max_length} (> 2^{int(math.log(MAX_ALLOWED_LENGTH, 2))}) is too large!"
            )

    def __call__(
        self,
        chr_name: str,
        start: int,
        end: int,
        max_length: int,
        i_shift: int,
    ) -> tuple[str, int, int]:
        """
        max_length passed from dataset, not from init
        """
        chromosome = self.seqs[chr_name]
        chromosome_length = self.chr_lens[chr_name]

        start, end = self._compute_interval(start, end, max_length, i_shift)

        if end > chromosome_length:
            # Shift interval down
            start = start - (end - chromosome_length)
            end = chromosome_length
            assert start == chromosome_length - max_length

        if start < 0:
            # Shift interval up
            end = end - start
            start = 0
            assert end == max_length

        if end > chromosome_length:
            # This may occur if start + MAX_ALLOWED_LENGTH extends beyond the end of the chromosome
            start = chromosome_length - max_length
            end = chromosome_length

        seq = str(chromosome[start:end])

        # NOTE: here the start and end may have been adjusted!
        return seq, start, end


class HG38Dataset(torch.utils.data.Dataset[HG38_EXAMPLE_T]):
    """Loop through bed file, retrieve (chr, start, end), query fasta file for sequence."""

    def __init__(
        self,
        *,
        split: str,
        bed_file: str | Path,
        fasta_file: str | Path,
        max_length: int,
        tokenizer: PreTrainedTokenizerBase,
        pad_max_length: int | None,
    ):
        self.max_length = max_length
        self.pad_max_length = (
            pad_max_length if pad_max_length is not None else max_length
        )
        self.tokenizer = tokenizer

        if max_length <= MAX_ALLOWED_LENGTH:
            assert (
                MAX_ALLOWED_LENGTH % max_length == 0
            ), "`max_length` must be a power of 2!"
            self.shifts = MAX_ALLOWED_LENGTH // max_length
        else:
            raise ValueError(
                f"`max_length` {max_length} (> 2^{int(math.log(MAX_ALLOWED_LENGTH, 2))}) is too large!"
            )

        bed_path = Path(bed_file)
        assert bed_path.exists(), "Path to .bed file must exist!"

        # read bed file
        df_raw = pd.read_csv(
            str(bed_path), sep="\t", names=["chr_name", "start", "end", "split"]
        )
        # select only split df
        self.df = df_raw[df_raw["split"] == split]
        # Update end points so that sequences are all length == MAX_ALLOWED_LENGTH
        self.df.loc[:, "end"] = self.df["start"] + MAX_ALLOWED_LENGTH

        self.fasta = FastaInterval(
            fasta_file=fasta_file,
        )

    def __len__(self) -> int:
        return len(self.df) * self.shifts

    def __getitem__(self, idx: int) -> HG38_EXAMPLE_T:
        """Returns a sequence of specified len"""
        # sample a random row from df
        row_idx, shift_idx = idx // self.shifts, idx % self.shifts
        row = self.df.iloc[row_idx]
        chr_name, start, end = (row.iloc[0], row.iloc[1], row.iloc[2])

        seq, adjusted_start, adjusted_end = self.fasta(
            chr_name,
            start,
            end,
            max_length=self.max_length,
            i_shift=shift_idx,
        )
        if end - start != MAX_ALLOWED_LENGTH:
            print(row, "\nLength: ", end - start)

        seq = self.tokenizer(
            seq,
            padding="max_length",
            max_length=self.pad_max_length,
            truncation=True,
            add_special_tokens=False,
        )

        seq = seq["input_ids"]  # get input_ids

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
