from pathlib import Path

import click
import numpy as np
from sklearn.model_selection import GridSearchCV

from .data import get_test_dataset, get_dataset, write_csv
from .pipeline import create_kaggle_model


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-d",
    "--dataset_test_path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-result-path",
    default="models/model-submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def results(
    dataset_path: Path,
    dataset_test_path: Path,
    save_result_path: Path,
) -> None:
    X, y = get_dataset(dataset_path, True)

    params = {
        "classifier__min_samples_leaf": [1, 4, 7],
        "classifier__max_depth": [34, 38, 32],
    }
    np.random.seed(1)
    model = GridSearchCV(create_kaggle_model(), params, cv=5, refit="True", n_jobs=-1)
    model.fit(X, y)

    best_model = model.best_estimator_.steps[0][1]
    best_model.fit(X, y)

    X_test = get_test_dataset(dataset_test_path, True)

    results = best_model.predict(X_test)
    write_csv(
        tuple(map(int, X_test.Id.to_numpy())),
        tuple(map(int, results)),
        save_result_path,
    )
