from typing import Annotated

import pandas as pd
import typer


def set_index(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index(["task_name", "model", "split", "metric"])


def main(
    experiment_path: Annotated[
        str, typer.Argument(help="Path to the experiment result CSV file")
    ],
) -> None:
    random_results = set_index(
        pd.read_csv("gs://cadu-distill/nt_eval/random.csv")
    ).rename(columns={"value": "random"})
    teacher_results = set_index(
        pd.read_csv("gs://cadu-distill/nt_eval/teacher.csv")
    ).rename(columns={"value": "teacher"})
    experiment_results = set_index(pd.read_csv(experiment_path)).rename(
        columns={"value": "experiment"}
    )

    print(random_results.join(experiment_results).join(teacher_results))


if __name__ == "__main__":
    typer.run(main)
