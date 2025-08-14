<p align="center">
Caduceus Distill
</p>

---

This repo is an experiment in distllation of the [Caduceus](https://github.com/kuleshov-group/caduceus) DNA model.

## Dependencies

Due to `mamba_ssm` setup.py shenanigans, uv init is a little more complicated:

```sh
uv sync
uv sync --extra mamba
```

> [!WARNING]
> You can't install mamba (`mamba-ssm`) package on a macbook.

## Results

TODO

## Run experiment

First download the data:

```sh
./bin/fetch_data
```

Example of a distillation experiment:

```sh
uv run distill \
  --batch-size=1 \
  --accumulate-grad-batches=2 \
  --temperature=1 \
  --lr=0.001 \
  --max-train-batches=32768 \
  --max-val-batches=32 \
  --max-final-val-batches=1024 \
  --val-check-interval=128 \
  --no-wandb
```

> [!WARNING]
> The command above doesn't publish to W&B (last flag). For a real experiment, you most definitely want to publish to W&B, see `--wandb-project-name` flag.

To see available options:

```sh
uv run distill --help
```
