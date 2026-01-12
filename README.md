# Automated Credit Risk Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MLflow](https://img.shields.io/badge/MLflow-2.6+-orange)](https://mlflow.org/)

## Overview
An end-to-end machine learning system for credit risk assessment with automated retraining, model monitoring, and production API deployment. Built following MLOps best practices.

## Architecture
![System Architecture](docs/architecture.png)

## Features
- **Automated ML Pipeline**: From raw data to deployed model
- **Experiment Tracking**: MLflow integration for reproducibility
- **Model Serving**: REST API with FastAPI
- **CI/CD Pipeline**: Automated testing and deployment
- **Drift Detection**: Monitor model performance over time
- **Interactive Dashboard**: Streamlit app for monitoring

## Tech Stack
| Category | Tools |
|----------|-------|
| **ML Framework** | Scikit-learn, XGBoost |
| **MLOps** | MLflow, DVC, Evidently |
| **API** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Containerization** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Testing** | Pytest, Pytest-cov |

## Project Structure
credit-risk-mlops/
├── .github/workflows/ # CI/CD pipelines
├── data/ # Data storage
│ ├── raw/ # Raw datasets
│ ├── processed/ # Processed data
│ └── external/ # External datasets
├── notebooks/ # Jupyter notebooks for EDA
├── config/ # Configuration files
├── src/ # Source code
│ ├── data/ # Data processing
│ ├── features/ # Feature engineering
│ ├── models/ # Model training
│ ├── visualization/ # Visualization utilities
│ └── utils/ # Helper functions
├── api/ # FastAPI application
├── tests/ # Unit and integration tests
├── monitoring/ # Monitoring dashboard
└── docs/ # Documentation

text

## Quick Start

### Prerequisites
- Python 3.9+
- Git
- Docker (optional)

### Installation
```bash
# Clone repository
git clone https://github.com/Asad-Farooq4421/credit-risk-mlops.git
cd credit-risk-mlops

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
Running the Application
bash
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# Train model
python src/models/train.py

# Start API
uvicorn api.app:app --reload --port 8000

# Start monitoring dashboard
streamlit run monitoring/dashboard.py
Model Performance
(Will be updated after model training)

License
MIT License - see LICENSE file for details.

Contributing
Contributions welcome! Please read CONTRIBUTING.md for details.

Contact
Asad Farooq
Data Science Enthusiast | Python | Pandas | SQL | Tableau | Turning Data into Actionable Insights

GitHub: @Asad-Farooq4421

LinkedIn: asad-farooq-data-scientist

Email: itsasadfarooq421@gmail.com
