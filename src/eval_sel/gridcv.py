from pathlib import Path

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
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
def train(dataset_path: Path, random_state: int, ) -> None:
    X, y = get_dataset(dataset_path)
    X_std = StandardScaler().fit_transform(X)

    cv_outer = KFold(n_splits=10, shuffle=True, random_state=random_state)
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X_std[train_ix, :], X_std[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
        with mlflow.start_run():
            model = DecisionTreeClassifier(random_state=random_state)

            space = dict()
            space['max_features'] = ["auto", "sqrt", "log2"]
            space['criterion'] = ["gini", "entropy"]
            space['max_depth'] = [None, 5, 3]
            space['splitter'] = ['best', 'random']

            search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
            result = search.fit(X_train, y_train)
            best_model = result.best_estimator_
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            best_params = result.best_params_
            mlflow.log_param("max_depth", best_params['max_depth'])
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("criterion", best_params['criterion'])
            mlflow.log_param("splitter", best_params['splitter'])
            mlflow.log_param("max_features", best_params['max_features'])

            mlflow.log_metric("accuracy", accuracy)

            mlflow.sklearn.log_model(best_model, artifact_path="sklearn-model")
