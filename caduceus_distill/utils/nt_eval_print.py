from typing import Annotated

import fsspec
import pandas as pd
import typer


def set_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(["task_name", "model", "split", "metric"])


def main(
    experiment_path: Annotated[
        str | None, typer.Argument(help="Path to the experiment result CSV file")
    ] = None,
    output_dir: Annotated[
        str, typer.Option(help="Result directory")
    ] = "gs://cadu-distill/nt_eval",
) -> None:
    df = set_index(pd.read_csv(f"{output_dir}/random.csv")).rename(
        columns={"value": "random"}
    )
    teacher_results = set_index(pd.read_csv(f"{output_dir}/teacher.csv")).rename(
        columns={"value": "teacher"}
    )

    if experiment_path is None:
        fs = fsspec.filesystem("gs")
        for e in fs.ls(output_dir):
            if not e.endswith(".csv"):
                continue
            if e.endswith("random.csv") or e.endswith("teacher.csv"):
                continue

            df = df.join(
                set_index(pd.read_csv(f"gs://{e}")).rename(
                    columns={"value": f"val_{e.split('.csv')[0].split('_ckpt')[0].split('_val_loss_total_')[1]}"}
                )
            )
    else:

        df = df.join(
            set_index(pd.read_csv(experiment_path)).rename(
                columns={"value": "experiment"}
            )
        )
    df = df.join(teacher_results)
    df.reset_index().to_markdown("/tmp/p.md")
    print(df)


if __name__ == "__main__":
    typer.run(main)
