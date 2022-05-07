from pathlib import Path

import numpy as np
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold, cross_validate

from .data import get_dataset
from .pipeline import create_tree


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
    "--max_depth",
    default=0,
    type=int,
    show_default=True,
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        max_depth: int,
) -> None:
    X, y = get_dataset(dataset_path)

    kfold = KFold(n_splits=10, random_state=random_state, shuffle=True)

    with mlflow.start_run():
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        pipeline = create_tree(max_depth, random_state)
        mlflow.sklearn.log_model(pipeline, artifact_path="sklearn-model")

        scoring = ['accuracy', 'precision_macro', 'recall_macro']
        score = cross_validate(pipeline, X, y, scoring=scoring, cv=kfold)

        mlflow.log_metric("accuracy", np.array(score['test_accuracy']).mean())
        mlflow.log_metric("precision", np.array(score['test_precision_macro']).mean())
        mlflow.log_metric("recall", np.array(score['test_recall_macro']).mean())

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
