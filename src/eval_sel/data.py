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
    X["DistHydro"] = (
        X["Horizontal_Distance_To_Hydrology"] ** 2
        + X["Vertical_Distance_To_Hydrology"] ** 2
    ) ** 0.5
    X["RelHydroRoad"] = abs(
        X["Horizontal_Distance_To_Hydrology"] - X["Horizontal_Distance_To_Roadways"]
    )
    X["RelFireRoad"] = abs(
        X["Horizontal_Distance_To_Fire_Points"] - X["Horizontal_Distance_To_Roadways"]
    )
    X["RelHydroFire"] = abs(
        X["Horizontal_Distance_To_Hydrology"] - X["Horizontal_Distance_To_Fire_Points"]
    )
    X["average_hillshade"] = np.mean(
        X["Hillshade_3pm"] + X["Hillshade_Noon"] + X["Hillshade_9am"]
    )
    X["Aspect_hillshade"] = (X["Aspect"] * X["Hillshade_9am"]) / 255
    X["slope_hillshade"] = (X["Slope"] * X["Hillshade_Noon"]) / 255

    X["Elevation_roadways"] = (
        X["Elevation"] - 0.02 * X["Horizontal_Distance_To_Roadways"]
    )
    X["Elevation_vd"] = X["Elevation"] - X["Vertical_Distance_To_Hydrology"]
    X["Elevation_hd"] = X["Elevation"] - 0.2 * X["Horizontal_Distance_To_Hydrology"]
    X["Elevation"] = [math.floor(v / 50.0) for v in X["Elevation"]]

    X = X.drop(
        [
            "Aspect",
            "Slope",
            "Hillshade_3pm",
            "Hillshade_9am",
            "Hillshade_Noon",
        ],
        axis=1,
    )
    return X


def write_csv(ids: Tuple[int, ...], predicts: Tuple[int, ...], save_path: Path) -> None:
    data_frame = pd.DataFrame({ID_COLUMN: list(ids), RESULT_COLUMN: list(predicts)})
    data_frame.set_index([ID_COLUMN])
    data_frame.to_csv(save_path, index=False)
