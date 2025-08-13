from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
import logging
from typing import List

logger = logging.getLogger("property-api.preprocessor")


def get_feature_columns(df_columns: List[str]) -> List[str]:
    try:
        excluded_cols = ['id', 'target']
        feature_cols = [col for col in df_columns if col not in excluded_cols]
        
        if not feature_cols:
            raise ValueError("No feature columns found after filtering")
        
        logger.info("Feature columns identified: %d columns", len(feature_cols))
        return feature_cols
        
    except Exception as e:
        logger.error("Error processing feature columns: %s", str(e))
        raise ValueError(f"Failed to process feature columns: {str(e)}")


def create_preprocessor(categorical_cols: List[str]) -> ColumnTransformer:
    try:
        if not categorical_cols:
            raise ValueError("No categorical columns provided")
        
        categorical_transformer = TargetEncoder()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, categorical_cols)
            ]
        )
        
        logger.info("Preprocessor created for %d categorical columns", len(categorical_cols))
        return preprocessor
        
    except Exception as e:
        logger.error("Error creating preprocessor: %s", str(e))
        raise ValueError(f"Failed to create preprocessor: {str(e)}")
