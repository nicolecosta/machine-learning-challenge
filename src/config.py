from typing import List, Dict, Any
import os

RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))

CATEGORICAL_COLS: List[str] = ["type", "sector"]
TARGET_COL: str = "price"
ID_COLS: List[str] = ["id", "target"]

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.01,
    "n_estimators": 300,
    "max_depth": 5,
    "loss": "absolute_error",
    "random_state": RANDOM_STATE
}

DATA_SOURCE_TYPE: str = os.getenv("DATA_SOURCE_TYPE", "csv")

DEFAULT_TRAIN_PATH: str = "data/train.csv"
DEFAULT_TEST_PATH: str = "data/test.csv"

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

TRAIN_QUERY: str = os.getenv("TRAIN_QUERY", "SELECT * FROM properties WHERE dataset_type = 'train'")
TEST_QUERY: str = os.getenv("TEST_QUERY", "SELECT * FROM properties WHERE dataset_type = 'test'")
