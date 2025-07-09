import logging
from importlib.util import find_spec
from typing import Annotated, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
import typer
import xarray as xr
from datasets import Dataset, load_dataset
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
from upath import UPath
from upath.implementations.local import PosixUPath

from caduceus_distill.distill import StudentCaduceus
from caduceus_distill.utils.utils import sanitize_name, setup_basic_logging

logger = logging.getLogger(__name__)


def main(
    model_to_load: Annotated[
        str,
        typer.Option(
            help="Model to load; 'random' for random initialization, 'teacher' for pre-trained model, or path to a checkpoint"
        ),
    ] = "random",
    task_group: Annotated[
        str,
        typer.Option(
            help="Task group to use; e.g. `eric_relevant`",
        ),
    ] = "eric_relevant",
    task_limit: Annotated[
        int | None,
        typer.Option(
            help="Limit number of tasks to process; if None, all tasks are processed"
        ),
    ] = None,
    sample_limit: Annotated[
        int, typer.Option(help="Limit number of samples to process", min=1)
    ] = 10_000,
    chunk_size: Annotated[
        int, typer.Option(help="Chunk size for processing", min=1)
    ] = 100,
    # TODO: why do we need this option?
    disable_fused_add_norm: Annotated[
        bool, typer.Option(help="Disable fused add norm")
    ] = True,
    output_dir: Annotated[
        str,
        typer.Option(help="Output directory for results. Can be GCS or local path."),
    ] = "gs://cadu-distill/nt_eval",
) -> None:
    output_path = UPath(output_dir)
    if isinstance(output_path, PosixUPath):
        output_path.mkdir(parents=True, exist_ok=True)

    clean_model_name = sanitize_name(model_to_load)

    features_path = output_path.joinpath("features", clean_model_name)
    features: xr.Dataset

    if not features_path.exists():
        logger.info(f"Creating features at {features_path}")
        features = create_features(
            model_to_load=model_to_load,
            task_group=task_group,
            chunk_size=chunk_size,
            sample_limit=sample_limit,
            disable_fused_add_norm=disable_fused_add_norm,
            task_limit=task_limit,
        )
        features.to_zarr(features_path.as_posix())
    logger.info(f"Loading features from {features_path}")
    features = xr.open_zarr(features_path.as_posix())
    logger.info(f"Loaded features:\n{str(features)}")

    summarize_labels(features)

    results = run_modeling(features)
    summarize_performance(results)

    # Save results
    results_path = output_path.joinpath(f"{clean_model_name}.csv")
    results.to_csv(results_path.as_posix(), index=False)
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
        # NOTE: use the original dataset rather than the revised one, per Eric comment in https://gist.github.com/eric-czech/7a368d2b2503b726a91510787f4fc373#file-caduceus_nt_eval_example-py
        # > The Caduceus paper used the original benchmark at InstaDeepAI/nucleotide_transformer_downstream_tasks, so I would suggest start with that instead
        "InstaDeepAI/nucleotide_transformer_downstream_tasks",
        name=task_name,
        trust_remote_code=True,
        # NOTE: commit 3 days ago https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks/commit/e8674d5c5940d2247cc16c5c589b2aec78c0cf94
        # broken reading specific task datasets, so we point at the previous commit from Sep 16, 2024
        revision="bba8d846099e57fa5ef0556c27055491550e8aeb",
    )
    return dataset


def load_caduceus(
    *, num_labels: int, model_to_load: str, disable_fused_add_norm: bool = True
) -> tuple[nn.Module, PreTrainedTokenizer]:  # noqa
    # See https://github.com/m42-health/gfm-random-eval/blob/575ceab00f841f2cd7f6e23810508829835871ea/nt_benchmark/models.py#L55
    model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # TODO (rav): what is the purpose of this attribute, I don't think this has any impact
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
    if model_to_load == "random":
        model = AutoModelForMaskedLM.from_config(model_config, trust_remote_code=True)
    elif model_to_load == "teacher":
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=model_config, trust_remote_code=True
        )
    else:
        model = StudentCaduceus.load_from_checkpoint(model_to_load)

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
    "eric_relevant": [
        "splice_sites_acceptors",
        "splice_sites_donors",
    ],
    "representative_binary_tasks": [
        "splice_sites_acceptors",
        "splice_sites_donors",
        "enhancers",
        "promoter_all",
        "H3K4me3",
    ],
    "all_binary_tasks": [
        "splice_sites_acceptors",
        "splice_sites_donors",
        "promoter_all",
        "promoter_tata",
        "promoter_no_tata",
        "enhancers",
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


def create_features(
    *,
    model_to_load: str,
    task_group: str,
    chunk_size: int,
    sample_limit: int,
    disable_fused_add_norm: bool,
    task_limit: int | None = None,
) -> xr.Dataset:
    datasets = []
    task_names = TASK_GROUPS[task_group]
    if task_limit is not None and task_limit > 0:
        logger.info(f"Limiting to {task_limit} tasks")
        task_names = task_names[:task_limit]

    for task_name in tqdm.tqdm(task_names, desc="Loading task datasets"):
        logger.info(f"Loading task {task_name}")
        try:
            dataset = load_nt_dataset(task_name)
            num_labels = len(set(dataset["train"]["label"]))
            model, tokenizer = load_caduceus(
                num_labels=num_labels,
                model_to_load=model_to_load,
                disable_fused_add_norm=disable_fused_add_norm,
            )
            for split in ["train", "test"]:
                ds = create_modeling_dataset(
                    chunk_size=chunk_size,
                    sample_limit=sample_limit,
                    ds=dataset[split],
                    model=model,
                    tokenizer=tokenizer,
                )
                ds = ds.assign(
                    split=(("samples", [split] * ds.sizes["samples"])),
                    task_name=(("samples", [task_name] * ds.sizes["samples"])),
                )
                datasets.append(ds)
        except Exception:
            logger.exception(f"Failed to create features for task {task_name}")
    features = xr.concat(datasets, dim="samples")
    features = features.assign_attrs({"num_labels": num_labels})
    return features


def create_modeling_dataset(
    *,
    chunk_size: int,
    sample_limit: int,
    ds: Dataset,
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
) -> xr.Dataset:
    chunks = torch.split(torch.arange(len(ds)), chunk_size)
    if sample_limit:
        logger.info(f"Limiting to {sample_limit} samples")
        n_chunks = (sample_limit + chunk_size - 1) // chunk_size
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
    labels = labels[: len(features)] if sample_limit else labels
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
    setup_basic_logging()
    typer.run(main)
