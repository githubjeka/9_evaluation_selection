from pathlib import Path
from typing import Tuple

import click
import numpy as np
import math
import pandas as pd

ID_COLUMN = "Id"
RESULT_COLUMN = "Cover_Type"


def get_dataset(
    csv_path: Path, features: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    X = dataset.drop(RESULT_COLUMN, axis=1)
    y = dataset[RESULT_COLUMN]
    if features:
        return apply_feature(X), y
    return X, y


def get_test_dataset(csv_path: Path, features: bool = True) -> pd.DataFrame:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Test Dataset shape: {dataset.shape}.")
    if features:
        return apply_feature(dataset)
    return dataset


def apply_feature(
    X: pd.DataFrame,
) -> pd.DataFrame:
    X = X.drop(["Soil_Type7", "Soil_Type15"], axis=1)
    a = X["Horizontal_Distance_To_Hydrology"]
    b = X["Vertical_Distance_To_Hydrology"]
    X["distance_to_hydrology"] = np.sqrt(np.power(a, 2) + np.power(b, 2))
    X["Horizontal_distance"] = (
        X["Horizontal_Distance_To_Hydrology"]
        + X["Horizontal_Distance_To_Roadways"]
        + X["Horizontal_Distance_To_Fire_Points"]
    ) / 3
    X["average_hillshade"] = (
        X["Hillshade_3pm"] + X["Hillshade_Noon"] + X["Hillshade_9am"]
    ) / 3
    X["Aspect_hillshade"] = (X["Aspect"] * X["Hillshade_9am"]) / 255
    X["slope_hillshade"] = (X["Slope"] * X["Hillshade_Noon"]) / 255
    X["Elevation"] = [math.floor(v / 50.0) for v in X["Elevation"]]
    return X


def write_csv(ids: Tuple[int, ...], predicts: Tuple[int, ...], save_path: Path) -> None:
    data_frame = pd.DataFrame({ID_COLUMN: list(ids), RESULT_COLUMN: list(predicts)})
    data_frame.set_index([ID_COLUMN])
    data_frame.to_csv(save_path, index=False)
