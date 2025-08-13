import pandas as pd
import logging
from typing import Tuple
from pathlib import Path

logger = logging.getLogger("property-api.data_loader")


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if not Path(train_path).exists():
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        if not Path(test_path).exists():
            raise FileNotFoundError(f"Test data file not found: {test_path}")
        
        logger.info("Loading training data...")
        train = pd.read_csv(train_path)
        
        logger.info("Loading test data...")
        test = pd.read_csv(test_path)
        
        if train.empty:
            raise ValueError("Training dataset is empty")
        if test.empty:
            raise ValueError("Test dataset is empty")
        
        logger.info("Data loaded successfully - Train: %d rows, Test: %d rows", 
                   len(train), len(test))
        
        return train, test
        
    except pd.errors.EmptyDataError:
        logger.error("One or more CSV files are empty")
        raise ValueError("CSV files contain no data")
    except pd.errors.ParserError as e:
        logger.error("CSV parsing error: %s", str(e))
        raise ValueError(f"Invalid CSV format: {str(e)}")
    except FileNotFoundError as e:
        logger.error("File not found: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error loading data: %s", str(e))
        raise
