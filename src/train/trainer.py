from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
import pandas as pd
import logging
import os
from typing import Dict, Any

logger = logging.getLogger("property-api.trainer")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
DEFAULT_MODEL_PARAMS = {
    "learning_rate": 0.01,
    "n_estimators": 300,
    "max_depth": 5,
    "loss": "absolute_error",
    "random_state": RANDOM_STATE
}


def create_model_pipeline(preprocessor: ColumnTransformer, 
                         model_params: Dict[str, Any] = None) -> Pipeline:
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()
    
    try:
        steps = [
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(**model_params))
        ]
        
        pipeline = Pipeline(steps)
        logger.info("Model pipeline created successfully with random_state=%s", 
                   model_params.get('random_state', 'not set'))
        return pipeline
        
    except Exception as e:
        logger.error("Error creating model pipeline: %s", str(e))
        raise ValueError(f"Failed to create model pipeline: {str(e)}")


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    try:
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty")
        
        if len(X_train) != len(y_train):
            raise ValueError("Feature and target data have different lengths")
        
        if y_train.isnull().any():
            raise ValueError("Target data contains null values")
        
        logger.info("Starting model training with %d samples", len(X_train))
        pipeline.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        
        return pipeline
        
    except ValueError as e:
        logger.error("Training validation error: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected training error: %s", str(e))
        raise RuntimeError(f"Model training failed: {str(e)}")
