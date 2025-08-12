from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import Dict, Any


def create_model_pipeline(preprocessor: ColumnTransformer, 
                         model_params: Dict[str, Any] = None) -> Pipeline:
    if model_params is None:
        model_params = {
            "learning_rate": 0.01,
            "n_estimators": 300,
            "max_depth": 5,
            "loss": "absolute_error"
        }
    
    steps = [
        ('preprocessor', preprocessor),
        ('model', GradientBoostingRegressor(**model_params))
    ]
    
    pipeline = Pipeline(steps)
    return pipeline


def train_model(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipeline.fit(X_train, y_train)
    return pipeline
