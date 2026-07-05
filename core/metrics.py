from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ValidationMetrics:
    name: str
    split: str
    auc: float
    f1: float
    accuracy: float
    precision: float
    recall: float
    notes: str = ""


def xy(frame: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame.drop(columns=[target]).values.astype("float32"),
        frame[target].values.astype("float32"),
    )


def score(
    name: str, y_true: np.ndarray, y_prob: np.ndarray, notes: str = ""
) -> ValidationMetrics:
    y_pred = (y_prob >= 0.5).astype(int)
    return ValidationMetrics(
        name=name,
        split="validation",
        auc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        notes=notes,
    )
