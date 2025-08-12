import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error
)
from typing import Dict


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    metrics = {
        "rmse": np.sqrt(mean_squared_error(predictions, targets)),
        "mape": mean_absolute_percentage_error(predictions, targets),
        "mae": mean_absolute_error(predictions, targets)
    }
    return metrics


def print_metrics(predictions: np.ndarray, targets: np.ndarray) -> None:
    metrics = calculate_metrics(predictions, targets)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAPE: {metrics['mape']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
