<div align="center">

# ⚡ CYBER CREDIT - Credit Risk MLOps System

<img src="https://img.shields.io/badge/CYBER-CREDIT-00ffff?style=for-the-badge&logo=lightning&logoColor=white" alt="Cyber Credit"/>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.6+-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

**🎮 A Cyberpunk-themed Credit Risk Assessment System with 3D Visualizations**

[Demo](#-live-demo) • [Features](#-features) • [Installation](#-installation) • [Documentation](#-documentation)

</div>

---

## 🌟 Overview

An **end-to-end MLOps system** for credit risk prediction featuring a stunning **Cyberpunk-themed dashboard** with Three.js 3D visualizations. Built for production with automated retraining, model monitoring, and REST API deployment.

### 🎯 Key Highlights

- **🎮 Cyberpunk 3D Dashboard** - Interactive Three.js visualizations with neon aesthetics
- **🤖 Multi-Model Training** - Logistic Regression, Random Forest, XGBoost, Gradient Boosting
- **📊 Real-time Risk Assessment** - Animated 3D gauge with risk scoring
- **🔬 Feature Engineering** - 20+ engineered features including domain-specific risk scores
- **📈 MLflow Tracking** - Full experiment reproducibility and model versioning

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Credit Risk Prediction
- Real-time risk scoring (0-100%)
- Color-coded decisions (Approve/Review/Decline)
- Confidence metrics
- Animated 3D risk gauge

</td>
<td width="50%">

### 📊 Model Performance
- 3D rotating metrics sphere
- ROC-AUC visualization
- Confusion matrix analysis
- Model comparison dashboard

</td>
</tr>
<tr>
<td width="50%">

### 🔬 Feature Analysis
- 3D bar chart feature importance
- Interactive feature exploration
- Domain feature engineering
- Composite risk scoring

</td>
<td width="50%">

### 🗺️ Data Explorer
- 3D scatter plots
- PCA visualization
- Customer segmentation
- Risk distribution analysis

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Three.js, Plotly |
| **ML/AI** | Scikit-learn, XGBoost, Pandas, NumPy |
| **MLOps** | MLflow, DVC, Evidently |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Visualization** | Plotly 3D, Three.js WebGL |
| **Testing** | Pytest, Pytest-cov |

---

## 📁 Project Structure

```
credit-risk-mlops/
├── 📂 api/                    # FastAPI REST endpoints
├── 📂 config/
│   └── config.yaml           # Centralized configuration
├── 📂 dashboard/
│   └── app.py                # 🎮 Cyberpunk Streamlit Dashboard
├── 📂 data/
│   ├── raw/                  # German Credit Dataset
│   └── processed/            # Processed features
├── 📂 docs/
│   └── DASHBOARD_MANUAL.md   # 📖 Complete User Guide
├── 📂 mlruns/                # MLflow experiment tracking
├── 📂 notebooks/             # EDA Jupyter notebooks
├── 📂 scripts/
│   └── test_pipeline.py      # Pipeline testing
├── 📂 src/
│   ├── data/
│   │   ├── make_dataset.py   # Data acquisition
│   │   └── validation.py     # Data quality validation
│   ├── features/
│   │   └── build_features.py # Feature engineering
│   ├── models/
│   │   └── train.py          # Model training pipeline
│   └── utils/
│       └── logger.py         # Logging utilities
├── 📂 tests/                 # Unit & integration tests
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- Git

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Asad-Farooq4421/credit-risk-mlops.git
cd credit-risk-mlops

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Dashboard 🎮
streamlit run dashboard/app.py
```

The dashboard will open at **http://localhost:8501**

---

## 📖 Usage

### 🎮 Launch Cyberpunk Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
```

### 🔧 Train Models
```bash
python -c "from src.models.train import ModelTrainer; ModelTrainer().run_full_pipeline()"
```

### 📊 Start MLflow UI
```bash
mlflow ui --backend-store-uri mlruns/ --port 5000
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** | 78.5% | 72.3% | 65.8% | 68.9% | **82.4%** |
| Random Forest | 77.2% | 71.1% | 64.2% | 67.4% | 80.1% |
| Gradient Boosting | 76.8% | 70.5% | 63.9% | 67.0% | 79.8% |
| Logistic Regression | 74.5% | 68.2% | 61.5% | 64.7% | 76.3% |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [📖 Dashboard Manual](docs/DASHBOARD_MANUAL.md) | Complete guide for using the dashboard |

---

## 🎯 Input Features (German Credit Dataset)

<details>
<summary><b>Click to expand feature descriptions</b></summary>

| Feature | Type | Description |
|---------|------|-------------|
| `status` | Categorical | Checking account status (A11-A14) |
| `duration` | Numerical | Credit duration in months |
| `credit_history` | Categorical | Credit history rating (A30-A34) |
| `purpose` | Categorical | Loan purpose (A40-A410) |
| `credit_amount` | Numerical | Credit amount requested |
| `savings` | Categorical | Savings account status (A61-A65) |
| `employment_duration` | Categorical | Employment length (A71-A75) |
| `installment_rate` | Numerical | Installment rate % |
| `age` | Numerical | Customer age |
| `housing` | Categorical | Housing type (A151-A153) |

</details>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

<div align="center">

**Asad Farooq**

[![GitHub](https://img.shields.io/badge/GitHub-@Asad--Farooq4421-181717?style=for-the-badge&logo=github)](https://github.com/Asad-Farooq4421)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/asad-farooq)

*Data Science Enthusiast | MLOps Engineer*

</div>

---

<div align="center">

**⚡ Built with 💜 using Streamlit + Three.js + MLflow ⚡**

<img src="https://img.shields.io/badge/Made%20with-Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Styled%20with-Cyberpunk-ff00ff?style=flat-square"/>

</div>
