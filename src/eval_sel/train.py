from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict

from .data import get_dataset
from .pipeline import create_pipeline


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
        predict = cross_val_predict(pipeline, X, y, cv=kfold)

        accuracy = accuracy_score(y, predict)
        precision = precision_score(y, predict, average='macro')
        recall = recall_score(y, predict, average='macro')
        mse = mean_squared_error(y, predict)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("mse", mse)

        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"precision: {precision}.")
        click.echo(f"Recall: {recall}.")
        click.echo(f"MSE: {mse}.")

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
