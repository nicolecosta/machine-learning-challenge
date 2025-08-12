from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class PropertyFeatures(BaseModel):
    type: Literal["casa", "departamento"] = Field(..., example="casa")
    sector: Literal["la reina", "las condes", "lo barnechea", "nunoa", "providencia", "vitacura"] = Field(..., example="las condes")
    net_usable_area: float = Field(..., example=140.0, gt=0.0)
    net_area: float = Field(..., example=170.0, gt=0.0)
    n_rooms: float = Field(..., example=3.0, ge=0.0)
    n_bathroom: float = Field(..., example=2.0, ge=0.0)
    latitude: float = Field(..., example=-33.40123)
    longitude: float = Field(..., example=-70.58056)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "casa",
                "sector": "las condes",
                "net_usable_area": 140.0,
                "net_area": 170.0,
                "n_rooms": 3.0,
                "n_bathroom": 2.0,
                "latitude": -33.40123,
                "longitude": -70.58056
            }
        },
    )


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., example=125000000.0, gt=0.0)
    status: str = Field(default="success", example="success")
    model_version: str = Field(default="v1.0")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_price": 125000000.0,
                "status": "success",
                "model_version": "v1.0"
            }
        }
    )


class HealthResponse(BaseModel):
    status: str = Field(...)
    model_loaded: bool = Field(...)
    timestamp: str = Field(...)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2024-01-15T10:30:00.123456"
            }
        }
    )
