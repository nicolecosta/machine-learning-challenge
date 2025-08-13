import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline

logger = logging.getLogger("property-api.predictor")


def make_predictions(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    try:
        if X.empty:
            raise ValueError("Input data is empty")
        
        if not hasattr(pipeline, 'predict'):
            raise ValueError("Pipeline does not have predict method")
        
        logger.info("Making predictions for %d samples", len(X))
        predictions = pipeline.predict(X)
        
        if len(predictions) != len(X):
            raise ValueError("Prediction count mismatch with input count")
        
        logger.info("Predictions completed successfully")
        return predictions
        
    except ValueError as e:
        logger.error("Prediction validation error: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected prediction error: %s", str(e))
        raise RuntimeError(f"Prediction failed: {str(e)}")


def predict_single(pipeline: Pipeline, features: dict) -> float:
    try:
        if not features:
            raise ValueError("Features dictionary is empty")
        
        X = pd.DataFrame([features])
        predictions = make_predictions(pipeline, X)
        
        prediction = predictions[0]
        if not isinstance(prediction, (int, float)) or pd.isna(prediction):
            raise ValueError("Invalid prediction result")
        
        return float(prediction)
        
    except ValueError as e:
        logger.error("Single prediction validation error: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected single prediction error: %s", str(e))
        raise RuntimeError(f"Single prediction failed: {str(e)}")
