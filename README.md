# Property Friends - Real Estate Valuation Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Machine learning pipeline for predicting residential property prices in Chile based on property features.

## Overview

This project productionizes a Jupyter notebook into a deployable ML solution. The system trains a Gradient Boosting model to predict property valuations and exposes predictions through a REST API.

## Installation

```bash
# Install dependencies using UV
uv sync
```

## Technical Requirements

- Docker deployment
- API logging for monitoring
- API documentation (FastAPI recommended)
- Basic security (API keys)
- Database abstraction for future integration
