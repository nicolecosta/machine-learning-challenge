import sys
import os
import joblib
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__)))

from process.data_loader import load_data
from process.preprocessor import get_feature_columns, create_preprocessor
from train.trainer import create_model_pipeline, train_model
from predict.predictor import make_predictions
from predict.evaluator import print_metrics
from config import CATEGORICAL_COLS, TARGET_COL, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH

logger = logging.getLogger("property-api.training")


def main():
    logger.info("Loading data...")
    train, test = load_data(DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH)
    
    logger.info("Preparing features...")
    train_cols = get_feature_columns(train.columns)
    
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
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
