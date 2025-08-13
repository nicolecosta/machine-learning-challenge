import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path

logger = logging.getLogger("property-api.data_sources")


class DataSource(ABC):
    
    @abstractmethod
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class CSVDataSource(DataSource):
    
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not Path(self.train_path).exists():
            raise FileNotFoundError(f"Training data file not found: {self.train_path}")
        if not Path(self.test_path).exists():
            raise FileNotFoundError(f"Test data file not found: {self.test_path}")
        
        logger.info("Loading training data from CSV...")
        train = pd.read_csv(self.train_path)
        
        logger.info("Loading test data from CSV...")
        test = pd.read_csv(self.test_path)
        
        if train.empty or test.empty:
            raise ValueError("One or more datasets are empty")
        
        logger.info("CSV data loaded - Train: %d rows, Test: %d rows", len(train), len(test))
        return train, test


class SQLDataSource(DataSource):
    
    def __init__(self, connection_string: str, train_query: str = "SELECT * FROM train_data", 
                 test_query: str = "SELECT * FROM test_data"):
        self.connection_string = connection_string
        self.train_query = train_query
        self.test_query = test_query
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            import sqlalchemy
        except ImportError:
            raise ImportError("sqlalchemy is required for SQL data sources. Install with: pip install sqlalchemy")
        
        logger.info("Loading training data from SQL...")
        engine = sqlalchemy.create_engine(self.connection_string)
        
        train = pd.read_sql(self.train_query, engine)
        test = pd.read_sql(self.test_query, engine)
        
        if train.empty or test.empty:
            raise ValueError("One or more datasets are empty")
        
        logger.info("SQL data loaded - Train: %d rows, Test: %d rows", len(train), len(test))
        return train, test


def create_data_source(source_type: str, **kwargs) -> DataSource:
    if source_type.lower() == 'csv':
        return CSVDataSource(
            kwargs.get('train_path', 'data/train.csv'),
            kwargs.get('test_path', 'data/test.csv')
        )
    elif source_type.lower() == 'sql':
        return SQLDataSource(
            kwargs['connection_string'],
            kwargs.get('train_query', 'SELECT * FROM train_data'),
            kwargs.get('test_query', 'SELECT * FROM test_data')
        )
    else:
        raise ValueError(f"Unsupported source type: {source_type}")


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    source = CSVDataSource(train_path, test_path)
    return source.load_training_data()
