import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def make_predictions(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    predictions = pipeline.predict(X)
    return predictions


def predict_single(pipeline: Pipeline, features: dict) -> float:
    X = pd.DataFrame([features])
    prediction = pipeline.predict(X)[0]
    return prediction
