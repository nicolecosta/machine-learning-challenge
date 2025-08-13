import sys
import os
import joblib
import logging
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__)))

from config import RANDOM_STATE
from process.data_sources import create_data_source
from process.preprocessor import get_feature_columns, create_preprocessor
from train.trainer import create_model_pipeline, train_model
from predict.predictor import make_predictions
from predict.evaluator import print_metrics
from config import (CATEGORICAL_COLS, TARGET_COL, DATA_SOURCE_TYPE, 
                   DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH, DATABASE_URL, 
                   TRAIN_QUERY, TEST_QUERY)

logger = logging.getLogger("property-api.training")


def set_random_seeds(seed: int = RANDOM_STATE):
    """Set random seeds for reproducibility across all libraries."""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info("Random seeds set to %d for reproducibility", seed)


def main():
    try:
        set_random_seeds()
        
        logger.info("Starting model training pipeline")
        
        logger.info("Loading data...")
        if DATA_SOURCE_TYPE.lower() == 'csv':
            data_source = create_data_source('csv', 
                                           train_path=DEFAULT_TRAIN_PATH, 
                                           test_path=DEFAULT_TEST_PATH)
        elif DATA_SOURCE_TYPE.lower() == 'sql':
            data_source = create_data_source('sql',
                                           connection_string=DATABASE_URL,
                                           train_query=TRAIN_QUERY,
                                           test_query=TEST_QUERY)
        else:
            raise ValueError(f"Unsupported data source type: {DATA_SOURCE_TYPE}")
        
        train, test = data_source.load_training_data()
        
        logger.info("Preparing features...")
        train_cols = get_feature_columns(train.columns)
        
        if not train_cols:
            raise ValueError("No feature columns found after filtering")
        
        if TARGET_COL not in train.columns or TARGET_COL not in test.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in datasets")
        
        logger.info("Creating preprocessor...")
        preprocessor = create_preprocessor(CATEGORICAL_COLS)
        
        logger.info("Creating model pipeline...")
        pipeline = create_model_pipeline(preprocessor)
        
        logger.info("Training model...")
        trained_pipeline = train_model(pipeline, train[train_cols], train[TARGET_COL])
        
        logger.info("Making predictions...")
        test_predictions = make_predictions(trained_pipeline, test[train_cols])
        test_target = test[TARGET_COL].values
        
        logger.info("Evaluation metrics:")
        print_metrics(test_predictions, test_target)
        
        logger.info("Saving model...")
        model_path = Path("models/property_model.joblib")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': trained_pipeline,
            'feature_columns': train_cols
        }
        
        joblib.dump(model_data, model_path)
        logger.info("Model saved successfully")
        
    except FileNotFoundError as e:
        logger.error("Required file not found: %s", str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error("Data validation error: %s", str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error in training pipeline: %s", str(e))
        sys.exit(1)
    logger.info("Model training completed successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
