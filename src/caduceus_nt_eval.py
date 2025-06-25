import logging
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
import xarray as xr
from datasets import Dataset, load_dataset
from omegaconf import DictConfig
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Running with config: {cfg}")

    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    features_path = output_path / "features.nc"
    features: xr.Dataset

    if not features_path.exists():
        logger.info(f"Creating features at {features_path}")
        features = create_features(cfg)
        features.to_netcdf(features_path)
    logger.info(f"Loading features from {features_path}")
    features = xr.open_dataset(features_path)
    logger.info(f"Loaded features:\n{str(features)}")

    summarize_labels(features)

    results = run_modeling(features)
    summarize_performance(results)

    # Save results
    results_path = output_path / "model_results.csv"
    results.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")


def summarize_labels(features: xr.Dataset) -> pd.DataFrame:
    with pd.option_context("display.max_rows", None):
        labels: pd.DataFrame = features[["task_name", "split", "label"]].to_dataframe()
        logger.info(
            "Label values:\n{df}".format(
                df=(labels.groupby(["task_name", "split"])["label"].unique())
            )
        )
        logger.info(
            "Label types:\n{df}".format(
                df=(
                    labels.assign(label_type=lambda df: df["label"].map(type))
                    .groupby(["task_name", "split"])["label_type"]
                    .value_counts()
                )
            )
        )
        logger.info(
            "Label frequencies:\n{df}".format(
                df=(
                    labels.groupby(["task_name", "split"])["label"].pipe(
                        lambda x: pd.concat(
                            [
                                x.value_counts(normalize=False),
                                x.value_counts(normalize=True),
                            ],
                            axis=1,
                        )
                    )
                )
            )
        )
    assert not labels["label"].isna().any(), "Found null labels"
    return labels


def summarize_performance(results: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", None):
        logger.info(f"Model Performance Metrics:\n{results}")


def load_nt_dataset(task_name: str) -> Dataset:
    # See:
    # - https://huggingface.co/spaces/InstaDeepAI/nucleotide_transformer_benchmark
    # - https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised
    # - https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks
    # - https://github.com/m42-health/gfm-random-eval/blob/575ceab00f841f2cd7f6e23810508829835871ea/nt_benchmark/ft_datasets.py#L50-L52
    # Note: both the m42 and Caduceus papers use the original dataset (nucleotide_transformer_downstream_tasks)
    # rather than the revised one (nucleotide_transformer_downstream_tasks_revised).
    dataset: Any = load_dataset(
        "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised",
        name=task_name,
        trust_remote_code=True,
    )
    return dataset


def load_caduceus(
    num_labels: int, random: bool, disable_fused_add_norm: bool = True
) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    # See https://github.com/m42-health/gfm-random-eval/blob/575ceab00f841f2cd7f6e23810508829835871ea/nt_benchmark/models.py#L55
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_config.num_labels = num_labels
    # This is a modification from the paper to see what happens if this option is NOT overridden;
    # critically, it changes the graph and does not use all pre-trained weights.  Here is the warning
    # you get when this is enabled (there is no warning when it is disabled):
    # > Some weights of the model checkpoint at kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 were not used when initializing CaduceusForMaskedLM: ['caduceus.backbone.layers.0.norm.weight', 'caduceus.backbone.layers.1.norm.weight', 'caduceus.backbone.layers.10.norm.weight', 'caduceus.backbone.layers.11.norm.weight', 'caduceus.backbone.layers.12.norm.weight', 'caduceus.backbone.layers.13.norm.weight', 'caduceus.backbone.layers.14.norm.weight', 'caduceus.backbone.layers.15.norm.weight', 'caduceus.backbone.layers.2.norm.weight', 'caduceus.backbone.layers.3.norm.weight', 'caduceus.backbone.layers.4.norm.weight', 'caduceus.backbone.layers.5.norm.weight', 'caduceus.backbone.layers.6.norm.weight', 'caduceus.backbone.layers.7.norm.weight', 'caduceus.backbone.layers.8.norm.weight', 'caduceus.backbone.layers.9.norm.weight', 'caduceus.backbone.norm_f.weight']
    # > - This IS expected if you are initializing CaduceusForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    # > - This IS NOT expected if you are initializing CaduceusForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    # > Some weights of CaduceusForMaskedLM were not initialized from the model checkpoint at kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16 and are newly initialized: ['caduceus.backbone.layers.0.norm.submodule.weight', 'caduceus.backbone.layers.1.norm.submodule.weight', 'caduceus.backbone.layers.10.norm.submodule.weight', 'caduceus.backbone.layers.11.norm.submodule.weight', 'caduceus.backbone.layers.12.norm.submodule.weight', 'caduceus.backbone.layers.13.norm.submodule.weight', 'caduceus.backbone.layers.14.norm.submodule.weight', 'caduceus.backbone.layers.15.norm.submodule.weight', 'caduceus.backbone.layers.2.norm.submodule.weight', 'caduceus.backbone.layers.3.norm.submodule.weight', 'caduceus.backbone.layers.4.norm.submodule.weight', 'caduceus.backbone.layers.5.norm.submodule.weight', 'caduceus.backbone.layers.6.norm.submodule.weight', 'caduceus.backbone.layers.7.norm.submodule.weight', 'caduceus.backbone.layers.8.norm.submodule.weight', 'caduceus.backbone.layers.9.norm.submodule.weight', 'caduceus.backbone.norm_f.submodule.weight']
    # > You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    if disable_fused_add_norm:
        model_config.fused_add_norm = False
    if random:
        model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
    else:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )

    num_gpus = torch.cuda.device_count()
    # Sequential warm-up to prevent a race condition in the Triton kernel autotuner.
    # This is necessary when using nn.DataParallel with models that use Triton JIT,
    # which is triggered here by `disable_fused_add_norm=True`.
    if num_gpus > 1 and disable_fused_add_norm:
        logger.info("Pre-warming Triton cache on all available GPUs sequentially...")
        warmup_seq_len = 1024  # A representative length to trigger compilation.
        for i in range(num_gpus):
            gpu_device = f"cuda:{i}"
            logger.info(f"  Warming up on {gpu_device}...")
            try:
                model.to(gpu_device)
                dummy_input = torch.randint(
                    0,
                    tokenizer.vocab_size,
                    (1, warmup_seq_len),
                    dtype=torch.long,
                    device=gpu_device,
                )
                with torch.no_grad():
                    _ = model(dummy_input)
                torch.cuda.synchronize(gpu_device)
                logger.info(f"  Warm-up on {gpu_device} complete.")
            except Exception as e:
                logger.warning(
                    f"  An error occurred during warm-up on {gpu_device}: {e}. "
                    "Proceeding, but errors may occur."
                )
        logger.info("All GPUs warmed up.")
        # Move model back to CPU before DataParallel wrapper.
        model.to("cpu")

    if num_gpus > 1:
        model = nn.DataParallel(model)

    model = model.to("cuda")
    return model, tokenizer


TASK_GROUPS = {
    # Tasks from https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised
    "enhancers_only": ["enhancers"],
    "representative_binary_tasks": [
        "enhancers",
        "promoter_all",
        "splice_sites_acceptors",
        "H3K4me3",
    ],
    "all_binary_tasks": [
        "promoter_all",
        "promoter_tata",
        "promoter_no_tata",
        "enhancers",
        "splice_sites_acceptors",
        "splice_sites_donors",
        "H2AFZ",
        "H3K27ac",
        "H3K27me3",
        "H3K36me3",
        "H3K4me1",
        "H3K4me2",
        "H3K4me3",
        "H3K9ac",
        "H3K9me3",
        "H4K20me1",
    ],
}


def create_features(cfg: DictConfig) -> xr.Dataset:
    datasets = []
    task_names = TASK_GROUPS[cfg.task_group]
    if cfg.get("task_limit") and cfg.task_limit > 0:
        logger.info(f"Limiting to {cfg.task_limit} tasks")
        task_names = task_names[: cfg.task_limit]

    for task_name in tqdm.tqdm(task_names, desc="Loading task datasets"):
        logger.info(f"Loading task {task_name}")
        try:
            dataset = load_nt_dataset(task_name)
            num_labels = len(set(dataset["train"]["label"]))
            model, tokenizer = load_caduceus(
                num_labels=num_labels,
                random=cfg.random,
                disable_fused_add_norm=cfg.disable_fused_add_norm,
            )
            for split in ["train", "test"]:
                ds = create_modeling_dataset(cfg, dataset[split], model, tokenizer)
                ds = ds.assign(
                    split=(("samples", [split] * ds.sizes["samples"])),
                    task_name=(("samples", [task_name] * ds.sizes["samples"])),
                )
                datasets.append(ds)
        except Exception as e:
            logger.error(f"Failed to create features for task {task_name}: {e}")
    features = xr.concat(datasets, dim="samples")
    features = features.assign_attrs({"num_labels": num_labels})
    return features


def create_modeling_dataset(
    cfg: DictConfig,
    ds: Dataset,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
) -> xr.Dataset:
    chunks = torch.split(torch.arange(len(ds)), cfg.chunk_size)
    if cfg.limit:
        logger.info(f"Limiting to {cfg.limit} samples")
        n_chunks = (cfg.limit + cfg.chunk_size - 1) // cfg.chunk_size
        chunks = chunks[:n_chunks]

    states = []
    for _, chunk in tqdm.tqdm(
        enumerate(chunks),
        total=len(chunks),
        desc=f"Creating features [dataset={ds.info.config_name}]",
    ):
        input_ids = tokenizer(
            ds[chunk.tolist()]["sequence"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to("cuda")
        with torch.no_grad():
            output = model(input_ids, output_hidden_states=True)
            chunk_features = output.hidden_states[-1].max(dim=1)[0].cpu()
        states.append(chunk_features)

    features = torch.cat(states, dim=0).numpy()
    labels = ds["label"]
    if not isinstance(labels, list):
        raise ValueError(
            f"Expected labels to be a list; got {type(labels)=}, {labels=}"
        )
    labels = labels[: len(features)] if cfg.limit else labels
    return xr.Dataset(
        {"feature": (["samples", "features"], features), "label": ("samples", labels)}
    )


def run_modeling(features: xr.Dataset, seed: int = 0) -> pd.DataFrame:
    task_names = features.task_name.to_series().drop_duplicates().to_list()
    results = []
    for task_name in tqdm.tqdm(task_names, desc="Running models"):
        logger.info(f"Running models for task {task_name!r}")

        ds = features.sel(samples=(features.task_name.values == task_name))
        assert ds.sizes["samples"] > 0, f"No samples found for task {task_name!r}"
        if len(task_names) > 1:
            assert ds.sizes["samples"] < features.sizes["samples"]
        train_mask = ds.split == "train"
        X_train = ds.feature.values[train_mask]
        y_train = ds.label.values[train_mask]
        X_test = ds.feature.values[~train_mask]
        y_test = ds.label.values[~train_mask]

        if (n_labels := len(set(y_train))) > 2:
            raise ValueError(
                "Only binary classification is supported; "
                f"found {n_labels} unique labels: {np.unique(y_train)}"
            )
        if (n_labels := len(set(y_train))) < 2:
            logger.warning(
                f"Found {n_labels} unique labels for {task_name=} but at least 2 are required; skipping"
            )
            continue

        if find_spec("lightgbm") is not None:
            from lightgbm import LGBMClassifier

            gbrt = LGBMClassifier(random_state=seed, verbose=-1)
            logger.info("Using LGBMClassifier for gradient boosting")
        else:
            logger.info("Using HistGradientBoostingClassifier for gradient boosting")
            gbrt = HistGradientBoostingClassifier(random_state=seed)
        models = {
            "lreg": make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(cv=5, max_iter=1000, random_state=seed),
            ),
            "gbrt": gbrt,
        }

        for model_name, model in models.items():
            logger.info(f"Training {task_name=}, {model_name=}...")
            model.fit(X_train, y_train)
            for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)[:, 1]
                metrics = {}
                for metric_name, (func, pred) in {
                    "roc_auc": (roc_auc_score, y_pred_proba),
                    "mcc": (matthews_corrcoef, y_pred),
                    "accuracy": (accuracy_score, y_pred),
                    "f1": (f1_score, y_pred),
                }.items():
                    try:
                        metrics[metric_name] = func(y, pred)
                    except Exception as e:
                        logger.warning(
                            f"Failed to compute {metric_name} for {task_name=}, {split=}: {e}"
                        )
                        metrics[metric_name] = float("nan")
                metrics["n_samples"] = len(y)
                metrics["n_positive"] = (y > 0).sum()
                results.extend(
                    [
                        {
                            "task_name": task_name,
                            "model": model_name,
                            "split": split,
                            "metric": k,
                            "value": v,
                        }
                        for k, v in metrics.items()
                    ]
                )

    return pd.DataFrame(results)


if __name__ == "__main__":
    main()
