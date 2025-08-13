# Property Friends - Real Estate Valuation Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

A production-ready machine learning pipeline for predicting residential property prices in Chile. This system transforms a Jupyter notebook prototype into a scalable, secure API with comprehensive logging and monitoring capabilities.

## 🎯 Challenge Deliverables

This project addresses all requirements from the Property-Friends Real Estate case:

✅ **Robust ML Pipeline**: Automated training pipeline with reproducibility and scalability  
✅ **Production API**: FastAPI-based service with property valuation predictions  
✅ **Docker Deployment**: Containerized application ready for production  
✅ **API Security**: API key authentication system  
✅ **Comprehensive Logging**: Request/response logging for model monitoring  
✅ **Database Abstraction**: Ready for future database integration  
✅ **Documentation**: Complete setup and usage instructions

## Overview

This project productionizes a Jupyter notebook into a deployable ML solution. The system trains a Gradient Boosting model to predict property valuations and exposes predictions through a REST API.

## Key Features

✅ **Ready for Production** - Dockerized, secure API with logging  
✅ **Database Ready** - Easy switch from CSV to your database  
✅ **Chilean Market Focused** - Validates coordinates and neighborhoods  
✅ **Fully Documented** - Interactive API docs at `/docs`  
✅ **Secure** - API key authentication built-in

## Quick Start

### Prerequisites
- Python 3.11+ installed
- UV package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- Docker (optional, for containerized deployment)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd machine-learning-challenge
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Prepare Data
Place your training data files in the `data/` directory:
- `data/train.csv` - Training dataset
- `data/test.csv` - Test dataset

**Note**: Data files are not included in this repository as they are proprietary to the client.

### 4. Train the Model
```bash
uv run python src/main.py
```

### 5. Start the API
```bash
export API_KEY=your-secret-key
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. Test the API
Visit `http://localhost:8000/docs` for interactive API documentation.

## Configuration Options

### Using CSV Files (Current Setup)
No additional configuration needed. The system uses:
- `data/train.csv` - Training data
- `data/test.csv` - Test data

### Switching to Database (Future)
When ready to connect to your database:

1. **Set environment variables:**
```bash
export DATA_SOURCE_TYPE=sql
export DATABASE_URL=your-database-connection-string
```

2. **Install database support:**
```bash
uv sync --extra sql 
```

3. **Run training:**
```bash
uv run python src/main.py
```

## Docker Deployment

### Build and Run
```bash
docker build -t property-api .
docker run -p 8000:8000 -e API_KEY=your-secret-key property-api
```

### With Database
```bash
docker run -p 8000:8000 \
  -e API_KEY=your-secret-key \
  -e DATA_SOURCE_TYPE=sql \
  -e DATABASE_URL=your-database-url \
  property-api
```

## API Usage

### Authentication
All API endpoints (except `/health` and `/docs`) require an API key:
```bash
export API_KEY=your-secret-key
```

### Predict Property Price
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "casa",
    "sector": "las condes",
    "net_usable_area": 140.0,
    "net_area": 170.0,
    "n_rooms": 3.0,
    "n_bathroom": 2.0,
    "latitude": -33.40123,
    "longitude": -70.58056
  }'
```

## Project Structure

```
├── app/              # FastAPI application
│   ├── main.py       # API endpoints and server
│   ├── auth.py       # API key authentication
│   └── schemas.py    # Request/response models
├── src/              # ML pipeline modules
│   ├── main.py       # Training pipeline orchestrator
│   ├── config.py     # Configuration settings
│   ├── process/      # Data processing
│   ├── train/        # Model training
│   └── predict/      # Prediction and evaluation
├── models/           # Trained model storage
├── data/             # Training data (not in repo)
├── notebooks/        # Original Jupyter notebook
└── Dockerfile        # Container configuration
```

## Model Assumptions & Design Decisions

### Data Assumptions
- **Property coordinates** are within Chilean territory (-56° to -17° latitude, -81° to -66° longitude)
- **Property types** limited to "casa" (house) and "departamento" (apartment) as per Chilean market standards
- **Valid Santiago neighborhoods**: la reina, las condes, lo barnechea, nunoa, providencia, vitacura
- **Area constraints**: Net usable area cannot exceed total net area (logical validation)
- **Feature completeness**: All required features (type, sector, areas, rooms, bathrooms, coordinates) must be provided
- **Data quality**: Training data is assumed to be clean and representative of current Chilean market conditions

### Modeling Assumptions
- **Target variable**: Property prices follow a distribution suitable for regression modeling
- **Feature relationships**: Linear and non-linear relationships between features and price exist
- **Temporal stability**: Model trained on historical data remains valid for current predictions
- **Market consistency**: Chilean real estate market patterns are consistent across the included neighborhoods

### Model Choices
- The model given in the original notebook is optimal for the client, and was implemented in a more robust way, but not changing the core implementation

### Architecture Decisions
- **Modular Design**: Separated data processing, training, and prediction for maintainability and testing
- **Data Abstraction**: Interface supports both CSV and SQL sources for future database integration
- **Error Handling**: Comprehensive validation and logging throughout pipeline for production monitoring
- **Security**: API key authentication system as required by the client
- **Database Ready**: Abstraction layer prepared for future direct database connection (client requirement)

## Areas for Improvement

### Model Enhancements
- **Enhanced Data Preprocessing**
- **Feature Engineering**
- **Advanced Models**
- **Cross-Validation**
- **Hyperparameter Tuning**

## Monitoring & Logging

The system includes comprehensive logging for production monitoring, including health checks
