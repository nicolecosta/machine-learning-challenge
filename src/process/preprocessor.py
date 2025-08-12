from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from typing import List


def get_feature_columns(df_columns: List[str]) -> List[str]:
    return [col for col in df_columns if col not in ['id', 'target']]


def create_preprocessor(categorical_cols: List[str]) -> ColumnTransformer:
    categorical_transformer = TargetEncoder()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor
