import numpy as np
import logging
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error
)
from typing import Dict

logger = logging.getLogger("property-api.evaluator")


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    metrics = {
        "rmse": np.sqrt(mean_squared_error(predictions, targets)),
        "mape": mean_absolute_percentage_error(predictions, targets),
        "mae": mean_absolute_error(predictions, targets)
    }
    return metrics


def print_metrics(predictions: np.ndarray, targets: np.ndarray) -> None:
    metrics = calculate_metrics(predictions, targets)
    logger.info("RMSE: %.4f", metrics['rmse'])
    logger.info("MAPE: %.4f", metrics['mape'])
    logger.info("MAE:  %.4f", metrics['mae'])
