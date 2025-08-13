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
                logger.error("Model file not found at: %s", self._model_path)
                self.is_loaded = False
                return False
                
            logger.info("Loading existing model...")
            data = joblib.load(self._model_path)
            self.model = data['model']
            self.feature_columns = data['feature_columns']
            self.is_loaded = True
            logger.info("Model loaded successfully - Features: %d", len(self.feature_columns))
            return True
            
        except Exception as e:
            logger.error("Model loading failed: %s", e)
            self.is_loaded = False
            return False
    
    def predict(self, features_df: pd.DataFrame) -> float:
        if not self.is_loaded or self.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
                detail="Model not available"
            )
        
        prediction = self.model.predict(features_df)[0]
        return float(prediction)

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
        logger.info("Prediction request received")
        
        df = pd.DataFrame([features.dict()])
        predicted_price = model_manager.predict(df)
        
        logger.info("Prediction successful")
        
        return PredictionResponse(
            predicted_price=predicted_price,
            status="success",
            model_version="v1.0"
        )
        
    except HTTPException:
        logger.warning("Authentication failed")
        raise
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Prediction failed"
        )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Property Friends - Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }
