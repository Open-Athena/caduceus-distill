import os
import argparse
import torch
import xarray as xr
from pyfaidx import Fasta
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm


class FastaDataset(Dataset):
    def __init__(self, fasta_file, chunk_size=131072, tokenizer=None):
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
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chr_name, start, end = self.chunks[idx]
        seq = str(self.fasta[chr_name][start:end])
        
        if len(seq) < self.chunk_size:
            seq = seq + 'N' * (self.chunk_size - len(seq))
        else:
            seq = seq[:self.chunk_size]
            
        tokens = self.tokenizer(seq, return_tensors="pt", add_special_tokens=False)
        assert tokens.input_ids.shape == (1, self.chunk_size)
        return tokens.input_ids.squeeze(0), chr_name, start, end


def generate_soft_labels(fasta_file, output_path, chunk_size=131072, batch_size=1, device="cuda"):
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    dataset = FastaDataset(fasta_file, chunk_size=chunk_size, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    for batch_idx, (input_ids, chr_names, starts, ends) in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_ids = input_ids.to(device)
        
        with torch.inference_mode():
            outputs = model(input_ids)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        batch_indices = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        
        ds = xr.Dataset({
            'input_ids': (['sample', 'sequence'], input_ids.cpu().numpy()),
            'logits': (['sample', 'sequence', 'vocab'], logits.cpu().numpy()),
            'probabilities': (['sample', 'sequence', 'vocab'], probabilities.cpu().numpy()),
        }, coords={
            'sample': batch_indices,
            'sequence': range(input_ids.shape[1]),
            'vocab': range(logits.shape[-1]),
            'chr_name': (['sample'], list(chr_names)),
            'start': (['sample'], [int(s) for s in starts]),
            'end': (['sample'], [int(e) for e in ends]),
        })
        
        ds.to_zarr(
            output_path,
            zarr_format=2,
            **(dict(append_dim="sample") if os.path.exists(output_path) else {}),
            consolidated=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate soft labels from FASTA sequences using Caduceus model")
    parser.add_argument("fasta_file", help="Path to FASTA file")
    parser.add_argument("output_path", help="Output Zarr file path")
    parser.add_argument("--chunk-size", type=int, default=131072, help="Chunk size for sequences (default: 131072)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--device", default="cuda", help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    generate_soft_labels(args.fasta_file, args.output_path, args.chunk_size, args.batch_size, args.device) 