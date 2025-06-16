## Dependencies

Due to `mamba_ssm` setup.py shenanigans, uv init is a little more complicated:

```sh
uv sync
uv sync --extra mamba
```

## Inference

### Fetch data

```sh
./bin/fetch_data
```

### Execution

```sh
uv run caduceus_inf.py data/hg38/hg38.ml.fa <OUTPUT_DIR>
```

You can adjust the batch and chunk size via CLI arguments.
