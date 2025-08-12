import sys
import os
import joblib
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__)))

from process.data_loader import load_data
from process.preprocessor import get_feature_columns, create_preprocessor
from train.trainer import create_model_pipeline, train_model
from predict.predictor import make_predictions
from predict.evaluator import print_metrics
from config import CATEGORICAL_COLS, TARGET_COL, DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH


def main():
    print("Loading data...")
    train, test = load_data(DEFAULT_TRAIN_PATH, DEFAULT_TEST_PATH)
    
    print("Preparing features...")
    train_cols = get_feature_columns(train.columns)
    
    print("Creating preprocessor...")
    preprocessor = create_preprocessor(CATEGORICAL_COLS)
    
    print("Creating model pipeline...")
    pipeline = create_model_pipeline(preprocessor)
    
    print("Training model...")
    trained_pipeline = train_model(pipeline, train[train_cols], train[TARGET_COL])
    
    print("Making predictions...")
    test_predictions = make_predictions(trained_pipeline, test[train_cols])
    test_target = test[TARGET_COL].values
    
    print("Evaluation metrics:")
    print_metrics(test_predictions, test_target)
    
    # Save model for API
    print("Saving model...")
    model_path = Path("models/property_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': trained_pipeline,
        'feature_columns': train_cols
    }
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
