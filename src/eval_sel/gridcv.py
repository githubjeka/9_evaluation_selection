from pathlib import Path
from typing import Any, Dict

import click
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier

from .data import get_dataset


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
def train(
    dataset_path: Path,
    random_state: int,
) -> None:
    X, y = get_dataset(dataset_path)

    with mlflow.start_run():
        model = DecisionTreeClassifier(random_state=random_state)

        params = dict()  # type: Dict[str, Any]
        params["max_features"] = ["auto", "sqrt", "log2"]
        params["criterion"] = ["gini", "entropy"]
        params["max_depth"] = [None, 5, 3]
        params["splitter"] = ["best", "random"]

        cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
        search = GridSearchCV(
            model, params, scoring="accuracy", cv=cv_inner, refit=True
        )

        cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
        score = cross_validate(search, X, y, scoring="accuracy", cv=cv_outer)

        mlflow.log_metric("accuracy", np.array(score["test_score"]).mean())
        mlflow.sklearn.log_model(search, artifact_path="sklearn-model")
