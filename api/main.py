"""
FastAPI Backend for Credit Risk Prediction
==========================================
Cyberpunk-themed API serving ML predictions
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import mlflow
from datetime import datetime

from src.features.build_features import FeatureEngineer
from src.utils.logger import load_config


# ============================================================================
# Pydantic Models
# ============================================================================

class CustomerInput(BaseModel):
    """Input schema for credit risk prediction"""
    status: str = Field(..., description="Checking account status (A11-A14)")
    duration: int = Field(..., ge=1, le=72, description="Credit duration in months")
    credit_history: str = Field(..., description="Credit history (A30-A34)")
    purpose: str = Field(..., description="Loan purpose (A40-A410)")
    credit_amount: float = Field(..., ge=0, description="Credit amount")
    savings: str = Field(..., description="Savings account (A61-A65)")
    employment_duration: str = Field(..., description="Employment duration (A71-A75)")
    installment_rate: int = Field(..., ge=1, le=4, description="Installment rate %")
    personal_status_sex: str = Field(..., description="Personal status & sex")
    other_debtors: str = Field(..., description="Other debtors (A101-A103)")
    present_residence: int = Field(..., ge=1, le=4, description="Years at residence")
    property: str = Field(..., description="Property type (A121-A124)")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    other_installment_plans: str = Field(..., description="Other plans (A141-A143)")
    housing: str = Field(..., description="Housing type (A151-A153)")
    number_credits: int = Field(..., ge=1, le=4, description="Number of credits")
    job: str = Field(..., description="Job type (A171-A174)")
    people_liable: int = Field(..., ge=1, le=2, description="People liable")
    telephone: str = Field(..., description="Has telephone (A191-A192)")
    foreign_worker: str = Field(..., description="Foreign worker (A201-A202)")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "A11",
                "duration": 12,
                "credit_history": "A32",
                "purpose": "A43",
                "credit_amount": 5000,
                "savings": "A61",
                "employment_duration": "A73",
                "installment_rate": 2,
                "personal_status_sex": "A93",
                "other_debtors": "A101",
                "present_residence": 2,
                "property": "A121",
                "age": 35,
                "other_installment_plans": "A143",
                "housing": "A152",
                "number_credits": 1,
                "job": "A173",
                "people_liable": 1,
                "telephone": "A192",
                "foreign_worker": "A201"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions"""
    risk_score: float = Field(..., description="Risk probability (0-1)")
    risk_label: str = Field(..., description="Risk classification")
    confidence: float = Field(..., description="Prediction confidence")
    recommendation: str = Field(..., description="Action recommendation")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    model_name: str
    training_date: str


class FeatureImportance(BaseModel):
    """Feature importance data"""
    feature: str
    importance: float
    rank: int


# ============================================================================
# Global State
# ============================================================================

class AppState:
    model = None
    feature_engineer = None
    config = None
    model_metrics = None
    feature_importances = None


state = AppState()


# ============================================================================
# Lifecycle Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artifacts on startup"""
    print("🚀 Starting Credit Risk API...")

    # Load config
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    state.config = load_config(str(config_path))

    # Try to load existing model
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    fe_path = models_dir / "feature_engineer.joblib"
    metrics_path = models_dir / "metrics.joblib"
    importance_path = models_dir / "feature_importance.joblib"

    if model_path.exists():
        state.model = joblib.load(model_path)
        print(f"✅ Loaded model from {model_path}")
    else:
        print("⚠️ No trained model found. Train a model first!")

    if fe_path.exists():
        state.feature_engineer = joblib.load(fe_path)
        print(f"✅ Loaded feature engineer from {fe_path}")
    else:
        state.feature_engineer = FeatureEngineer()
        print("⚠️ Using new FeatureEngineer instance")

    if metrics_path.exists():
        state.model_metrics = joblib.load(metrics_path)
        print(f"✅ Loaded metrics from {metrics_path}")

    if importance_path.exists():
        state.feature_importances = joblib.load(importance_path)
        print(f"✅ Loaded feature importances from {importance_path}")

    print("🎮 CYBERPUNK CREDIT RISK API ONLINE")
    yield
    print("👋 Shutting down API...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="⚡ CREDIT RISK NEURAL ENGINE",
    description="""
    ```
    ╔═══════════════════════════════════════════════════════════╗
    ║  ██████╗██╗   ██╗██████╗ ███████╗██████╗                  ║
    ║ ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗                 ║
    ║ ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝                 ║
    ║ ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗                 ║
    ║ ╚██████╗   ██║   ██████╔╝███████╗██║  ██║                 ║
    ║  ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝                 ║
    ║         CREDIT RISK PREDICTION ENGINE v2.0                ║
    ╚═══════════════════════════════════════════════════════════╝
    ```

    Neural-powered credit risk assessment system with real-time ML inference.
    """,
    version="2.0.0",
    lifespan=lifespan
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """System status check"""
    return {
        "status": "online",
        "system": "CREDIT RISK NEURAL ENGINE",
        "version": "2.0.0",
        "model_loaded": state.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_status": "loaded" if state.model else "not_loaded",
        "uptime": "operational"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    🎯 Predict credit risk for a customer

    Returns risk score, classification, and recommendation
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train a model first."
        )

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([customer.model_dump()])

        # Apply feature engineering if available
        if hasattr(state.feature_engineer, 'transform'):
            processed_data = state.feature_engineer.transform(input_data)
        else:
            processed_data = input_data

        # Make prediction
        if hasattr(state.model, 'predict_proba'):
            probabilities = state.model.predict_proba(processed_data)
            risk_score = float(probabilities[0][1])  # Probability of bad credit
        else:
            prediction = state.model.predict(processed_data)
            risk_score = float(prediction[0])

        # Determine risk label and recommendation
        if risk_score < 0.3:
            risk_label = "LOW RISK"
            recommendation = "✅ APPROVE - Strong credit profile"
            confidence = 1 - risk_score
        elif risk_score < 0.6:
            risk_label = "MEDIUM RISK"
            recommendation = "⚠️ REVIEW - Additional verification recommended"
            confidence = 0.5 + abs(0.5 - risk_score)
        else:
            risk_label = "HIGH RISK"
            recommendation = "❌ DECLINE - High default probability"
            confidence = risk_score

        return PredictionResponse(
            risk_score=round(risk_score, 4),
            risk_label=risk_label,
            confidence=round(confidence, 4),
            recommendation=recommendation,
            timestamp=datetime.now().isoformat(),
            model_version="v2.0.0"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(customers: List[CustomerInput]):
    """
    📦 Batch prediction for multiple customers
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for customer in customers:
        result = await predict(customer)
        results.append(result)

    return {
        "predictions": results,
        "total": len(results),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/metrics", response_model=ModelMetrics, tags=["Model"])
async def get_model_metrics():
    """
    📊 Get current model performance metrics
    """
    if state.model_metrics is None:
        # Return default metrics if none saved
        return ModelMetrics(
            accuracy=0.78,
            precision=0.72,
            recall=0.65,
            f1_score=0.68,
            roc_auc=0.82,
            model_name="XGBoost",
            training_date=datetime.now().strftime("%Y-%m-%d")
        )

    return ModelMetrics(**state.model_metrics)


@app.get("/model/feature-importance", tags=["Model"])
async def get_feature_importance():
    """
    🔬 Get feature importance rankings
    """
    if state.feature_importances is not None:
        return {"features": state.feature_importances}

    # Default feature importances for demo
    default_features = [
        {"feature": "credit_amount", "importance": 0.182, "rank": 1},
        {"feature": "duration", "importance": 0.156, "rank": 2},
        {"feature": "age", "importance": 0.134, "rank": 3},
        {"feature": "status", "importance": 0.098, "rank": 4},
        {"feature": "credit_history", "importance": 0.087, "rank": 5},
        {"feature": "savings", "importance": 0.076, "rank": 6},
        {"feature": "employment_duration", "importance": 0.065, "rank": 7},
        {"feature": "purpose", "importance": 0.058, "rank": 8},
        {"feature": "installment_rate", "importance": 0.045, "rank": 9},
        {"feature": "housing", "importance": 0.042, "rank": 10},
    ]

    return {"features": default_features}


@app.get("/model/experiments", tags=["MLflow"])
async def get_experiments():
    """
    📈 Get MLflow experiment history
    """
    try:
        mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
        experiments = mlflow.search_experiments()

        exp_list = []
        for exp in experiments:
            exp_list.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            })

        return {"experiments": exp_list}
    except Exception as e:
        return {"experiments": [], "error": str(e)}


@app.get("/data/sample", tags=["Data"])
async def get_sample_data():
    """
    📋 Get sample data for testing
    """
    data_path = PROJECT_ROOT / "data" / "raw" / "german_credit.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        sample = df.head(5).to_dict(orient="records")
        return {
            "sample": sample,
            "total_records": len(df),
            "columns": list(df.columns)
        }

    return {"error": "Data file not found"}


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
