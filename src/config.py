from typing import List, Dict, Any

CATEGORICAL_COLS: List[str] = ["type", "sector"]
TARGET_COL: str = "price"
ID_COLS: List[str] = ["id", "target"]

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.01,
    "n_estimators": 300,
    "max_depth": 5,
    "loss": "absolute_error"
}

DEFAULT_TRAIN_PATH: str = "data/train.csv"
DEFAULT_TEST_PATH: str = "data/test.csv"
