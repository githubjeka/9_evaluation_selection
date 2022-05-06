from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(use_scaler: bool, max_iter: int, log_reg_c: float, random_state: int) -> Pipeline:
    pipeline_steps = []

    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=log_reg_c
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)
