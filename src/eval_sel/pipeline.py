from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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


def create_tree(max_depth: int, criterion: str, random_state: int) -> Pipeline:
    pipeline_steps = [(
        "tree",
        DecisionTreeClassifier(random_state=random_state, max_depth=max_depth if max_depth > 0 else None,
                               criterion=criterion),
    )]

    return Pipeline(steps=pipeline_steps)
