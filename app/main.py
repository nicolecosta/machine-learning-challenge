import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status

from .auth import get_api_key
from .schemas import PropertyFeatures, PredictionResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("property-api")


class ModelManager:
    def __init__(self):
        self.model: Optional[object] = None
        self.feature_columns: Optional[list] = None
        self.is_loaded: bool = False
        self._model_path = Path(__file__).parent.parent / "models" / "property_model.joblib"
    
    def load_model(self) -> bool:
        try:
            if not self._model_path.exists():
                logger.error("Model file not found")
                self.is_loaded = False
                return False
                
            logger.info("Loading machine learning model...")
            data = joblib.load(self._model_path)
            
            if not isinstance(data, dict) or 'model' not in data or 'feature_columns' not in data:
                logger.error("Invalid model file format - missing required keys")
                self.is_loaded = False
                return False
            
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            
            if not hasattr(self.model, 'predict'):
                logger.error("Loaded object does not have predict method")
                self.is_loaded = False
                return False
                
            self.is_loaded = True
            logger.info("Model loaded successfully - Features: %d", len(self.feature_columns))
            return True
            
        except FileNotFoundError:
            logger.error("Model file not found")
            self.is_loaded = False
            return False
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, EOFError):
            logger.error("Model file appears to be corrupted")
            self.is_loaded = False
            return False
        except Exception as e:
            logger.error("Unexpected error loading model: %s", str(e))
            self.is_loaded = False
            return False
    
    def predict(self, features_df: pd.DataFrame) -> float:
        if not self.is_loaded or self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="Model not available"
            )
        
        try:
            if features_df.empty:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Empty input data"
                )
            
            missing_cols = set(self.feature_columns) - set(features_df.columns)
            if missing_cols:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required columns: {list(missing_cols)}"
                )
            
            features_df = features_df[self.feature_columns]
            prediction = self.model.predict(features_df)[0]
            
            if not isinstance(prediction, (int, float)) or pd.isna(prediction):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Invalid prediction result"
                )
            
            return float(prediction)
            
        except HTTPException:
            raise
        except ValueError as e:
            logger.error("Input validation error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input data format"
            )
        except Exception as e:
            logger.error("Prediction error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed"
            )

model_manager = ModelManager()

app = FastAPI(
    title="Property Friends - Price Prediction API",
    description="API for predicting property prices using machine learning",
    version="1.0.0",
    docs_url="/docs",
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Property Friends API...")
    success = model_manager.load_model()
    if success:
        logger.info("API startup completed successfully")
    else:
        logger.warning("API started but model is not available")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "unhealthy",
        model_loaded=model_manager.is_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", 
          response_model=PredictionResponse,
          summary="Predict Property Price",
          description="Predict Chilean property prices using machine learning. Requires property details like type, location, area, and coordinates.")
async def predict_property_price(
    features: PropertyFeatures,
    api_key: str = Depends(get_api_key)
):
    try:
        logger.info("Prediction request received for property type: %s, sector: %s", 
                   features.type, features.sector)
        
        # Pydantic validation handles all input validation automatically
        # No need for manual validation here
        
        df = pd.DataFrame([features.dict()])
        predicted_price = model_manager.predict(df)
        
        if predicted_price <= 0:
            logger.warning("Unusual prediction result: %f", predicted_price)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Invalid prediction result - please check input values"
            )
        
        logger.info("Prediction successful: %f", predicted_price)
        
        return PredictionResponse(
            predicted_price=predicted_price,
            status="success",
            model_version="v1.0"
        )
        
    except HTTPException as e:
        logger.warning("HTTP exception in prediction: %s", e.detail)
        raise
    except Exception as e:
        logger.error("Unexpected prediction error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal server error"
        )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Property Friends - Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }
