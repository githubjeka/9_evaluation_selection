from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold, cross_validate

from .data import get_dataset
from .pipeline import create_pipeline


def validate_logreg_c(ctx: Any, param: Any, value: Any) -> Any:
    if value > 0 and value == 1.0:
        return value
    else:
        raise click.BadParameter("logreg_c format must (0,1]'")


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="models/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
    callback=validate_logreg_c,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        use_scaler: bool,
        max_iter: int,
        logreg_c: float,
) -> None:
    X, y = get_dataset(dataset_path)

    kfold = KFold(n_splits=10, random_state=random_state, shuffle=True)

    with mlflow.start_run():
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_param("random_state", random_state)

        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        mlflow.sklearn.log_model(pipeline, artifact_path="sklearn-model")

        scoring = ["accuracy", "precision_macro", "recall_macro"]
        score = cross_validate(pipeline, X, y, scoring=scoring, cv=kfold)

        mlflow.log_metric("accuracy", np.array(score["test_accuracy"]).mean())
        mlflow.log_metric("precision", np.array(score["test_precision_macro"]).mean())
        mlflow.log_metric("recall", np.array(score["test_recall_macro"]).mean())

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
