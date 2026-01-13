# 🎮 CYBERPUNK CREDIT RISK DASHBOARD - USER MANUAL

<div align="center">

![Dashboard](https://img.shields.io/badge/CYBER-CREDIT-00ffff?style=for-the-badge&logo=streamlit)
![Version](https://img.shields.io/badge/VERSION-2.0.0-ff00ff?style=for-the-badge)
![Status](https://img.shields.io/badge/STATUS-ONLINE-00ff00?style=for-the-badge)

**A Modern 3D Credit Risk Assessment Interface**

</div>

---

## 📋 TABLE OF CONTENTS

1. [Getting Started](#-getting-started)
2. [Navigation](#-navigation)
3. [Prediction Module](#-prediction-module---input-guide)
4. [Metrics Module](#-metrics-module)
5. [Features Module](#-features-module)
6. [Data Explorer](#-data-explorer)
7. [Input Field Reference](#-input-field-reference)
8. [Understanding Results](#-understanding-results)
9. [Troubleshooting](#-troubleshooting)

---

## 🚀 GETTING STARTED

### Prerequisites
```bash
pip install streamlit plotly streamlit-lottie pandas numpy
```

### Launching the Dashboard
```bash
cd credit-risk-mlops
streamlit run dashboard/app.py --server.port 8501
```

The dashboard will open at: **http://localhost:8501**

---

## 🧭 NAVIGATION

The sidebar contains 5 main modules:

| Icon | Module | Description |
|------|--------|-------------|
| 🎯 | **PREDICT** | Credit risk prediction with 3D gauge visualization |
| 📊 | **METRICS** | Model performance metrics with 3D rotating sphere |
| 🔬 | **FEATURES** | Feature importance analysis with 3D bar charts |
| 🗺️ | **EXPLORE** | 3D data exploration and scatter plots |
| ⚙️ | **SETTINGS** | Dashboard configuration options |

---

## 🎯 PREDICTION MODULE - INPUT GUIDE

This is the main module where you input customer data to predict credit risk.

### Step-by-Step Instructions

#### 1️⃣ Navigate to Prediction Page
- Click **"🎯 PREDICT"** in the sidebar

#### 2️⃣ Fill Customer Data Form

The form has two columns of inputs:

##### **LEFT COLUMN:**

| Field | How to Fill | Example |
|-------|-------------|---------|
| **Account Status** | Select from dropdown | A12 |
| **Duration** | Enter loan duration in months (1-72) | 24 |
| **Credit Amount** | Enter requested credit in currency (0-50000) | 5000 |
| **Age** | Enter customer age (18-100) | 35 |
| **Savings** | Select savings account status | A62 |

##### **RIGHT COLUMN:**

| Field | How to Fill | Example |
|-------|-------------|---------|
| **Credit History** | Select credit history rating | A32 |
| **Purpose** | Select loan purpose | A43 |
| **Employment Duration** | Select employment length | A73 |
| **Housing** | Select housing situation | A152 |
| **Job Type** | Select job category | A173 |

#### 3️⃣ Submit for Analysis
- Click the **"⚡ ANALYZE RISK"** button
- Wait for the neural network to process (~1-2 seconds)

#### 4️⃣ View Results
- 3D animated gauge shows risk percentage
- Color-coded decision: GREEN (approve) / YELLOW (review) / RED (decline)
- Confidence score displayed below

---

## 📊 INPUT FIELD REFERENCE

### Account Status (status)
| Code | Meaning | Risk Impact |
|------|---------|-------------|
| **A11** | Balance < 0 DM (overdrawn) | 🔴 Higher Risk |
| **A12** | 0 ≤ Balance < 200 DM | 🟡 Medium Risk |
| **A13** | Balance ≥ 200 DM | 🟢 Lower Risk |
| **A14** | No checking account | 🟡 Medium Risk |

### Credit History (credit_history)
| Code | Meaning | Risk Impact |
|------|---------|-------------|
| **A30** | No credits / all paid back | 🟢 Lower Risk |
| **A31** | All credits at this bank paid | 🟢 Lower Risk |
| **A32** | Existing credits paid on time | 🟢 Lower Risk |
| **A33** | Delay in past payments | 🟡 Medium Risk |
| **A34** | Critical account / other credits | 🔴 Higher Risk |

### Purpose (purpose)
| Code | Meaning |
|------|---------|
| **A40** | Car (new) |
| **A41** | Car (used) |
| **A42** | Furniture/equipment |
| **A43** | Radio/television |
| **A44** | Domestic appliances |
| **A45** | Repairs |
| **A46** | Education |

### Savings Account (savings)
| Code | Meaning | Risk Impact |
|------|---------|-------------|
| **A61** | < 100 DM | 🔴 Higher Risk |
| **A62** | 100 ≤ ... < 500 DM | 🟡 Medium Risk |
| **A63** | 500 ≤ ... < 1000 DM | 🟢 Lower Risk |
| **A64** | ≥ 1000 DM | 🟢 Lower Risk |
| **A65** | Unknown / no savings | 🟡 Medium Risk |

### Employment Duration (employment)
| Code | Meaning | Risk Impact |
|------|---------|-------------|
| **A71** | Unemployed | 🔴 Higher Risk |
| **A72** | < 1 year employed | 🟡 Medium Risk |
| **A73** | 1 ≤ ... < 4 years | 🟢 Lower Risk |
| **A74** | 4 ≤ ... < 7 years | 🟢 Lower Risk |
| **A75** | ≥ 7 years employed | 🟢 Lower Risk |

### Housing (housing)
| Code | Meaning | Risk Impact |
|------|---------|-------------|
| **A151** | Rent | 🟡 Medium Risk |
| **A152** | Own property | 🟢 Lower Risk |
| **A153** | For free (with family) | 🟡 Medium Risk |

### Job Type (job)
| Code | Meaning | Risk Impact |
|------|---------|-------------|
| **A171** | Unemployed/unskilled (non-resident) | 🔴 Higher Risk |
| **A172** | Unskilled (resident) | 🟡 Medium Risk |
| **A173** | Skilled employee | 🟢 Lower Risk |
| **A174** | Management/self-employed/highly qualified | 🟢 Lower Risk |

---

## 📊 METRICS MODULE

Navigate to **"📊 METRICS"** to view model performance.

### Features:
- **3D Rotating Sphere** with orbiting metric indicators
- **Performance Cards** showing:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

### Interpreting Metrics:

| Metric | Good Value | Excellent Value |
|--------|-----------|-----------------|
| Accuracy | > 70% | > 85% |
| Precision | > 65% | > 80% |
| Recall | > 60% | > 75% |
| F1-Score | > 65% | > 80% |
| ROC-AUC | > 75% | > 90% |

---

## 🔬 FEATURES MODULE

Navigate to **"🔬 FEATURES"** to analyze feature importance.

### Features:
- **3D Bar Chart** showing feature importance rankings
- **Interactive hover** for detailed importance values
- **Rotate view** by clicking and dragging

### Top Important Features (typically):
1. **credit_amount** - Loan amount requested
2. **duration** - Loan duration in months
3. **age** - Customer age
4. **status** - Checking account status
5. **credit_history** - Past credit behavior

---

## 🗺️ DATA EXPLORER

Navigate to **"🗺️ EXPLORE"** for 3D data visualization.

### Features:
- **3D Scatter Plot** with:
  - X-axis: Credit Amount
  - Y-axis: Duration
  - Z-axis: Age
  - Color: Risk (Cyan = Good, Pink = Bad)

### Interacting with the Plot:
- **Rotate**: Click and drag
- **Zoom**: Scroll wheel
- **Pan**: Right-click and drag
- **Hover**: See detailed data point info

---

## 🎨 UNDERSTANDING RESULTS

### Risk Score Interpretation

| Risk Range | Color | Decision | Action |
|------------|-------|----------|--------|
| **0% - 30%** | 🟢 GREEN | APPROVED | Proceed with loan |
| **30% - 60%** | 🟡 YELLOW | REVIEW REQUIRED | Manual assessment needed |
| **60% - 100%** | 🔴 RED | DECLINED | High default probability |

### Confidence Score
- **> 80%**: High confidence in prediction
- **50-80%**: Moderate confidence
- **< 50%**: Low confidence, consider additional factors

---

## 📝 EXAMPLE SCENARIOS

### Scenario 1: Low Risk Customer
```
Account Status: A13 (≥200 DM balance)
Duration: 12 months
Credit Amount: 3000
Age: 45
Credit History: A32 (paid on time)
Savings: A63 (500-1000 DM)
Employment: A74 (4-7 years)
Housing: A152 (own property)
Job: A173 (skilled employee)

Expected Result: ✅ LOW RISK (~15-25%)
```

### Scenario 2: High Risk Customer
```
Account Status: A11 (overdrawn)
Duration: 48 months
Credit Amount: 25000
Age: 22
Credit History: A34 (critical)
Savings: A61 (<100 DM)
Employment: A71 (unemployed)
Housing: A151 (rent)
Job: A171 (unskilled non-resident)

Expected Result: ❌ HIGH RISK (~70-85%)
```

### Scenario 3: Medium Risk Customer
```
Account Status: A12 (0-200 DM)
Duration: 24 months
Credit Amount: 8000
Age: 30
Credit History: A33 (past delays)
Savings: A62 (100-500 DM)
Employment: A72 (<1 year)
Housing: A153 (with family)
Job: A172 (unskilled resident)

Expected Result: ⚠️ MEDIUM RISK (~40-55%)
```

---

## 🔧 TROUBLESHOOTING

### Common Issues

#### Dashboard Won't Start
```bash
# Check if port is in use
netstat -ano | findstr :8501

# Try different port
streamlit run dashboard/app.py --server.port 8502
```

#### 3D Visualizations Not Loading
- Ensure JavaScript is enabled in browser
- Try refreshing the page (F5)
- Clear browser cache

#### Slow Performance
- Close other browser tabs
- Reduce data points in explorer (automatic sampling enabled)

#### Form Not Submitting
- Ensure all required fields are filled
- Check for valid numeric ranges (duration: 1-72, age: 18-100, amount: 0-50000)

---

## ⌨️ KEYBOARD SHORTCUTS

| Key | Action |
|-----|--------|
| `R` | Refresh page |
| `S` | Focus on sidebar |
| `Esc` | Close dropdowns |

---

## 🌐 DEPLOYMENT (Streamlit Cloud)

### Steps to Deploy:

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repository: `credit-risk-mlops`
5. Set main file: `dashboard/app.py`
6. Click **Deploy**

### Required `requirements.txt`:
```
streamlit>=1.28.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
streamlit-lottie>=0.0.5
```

---

## 📞 SUPPORT

For issues or questions:
- GitHub: [@Asad-Farooq4421](https://github.com/Asad-Farooq4421)
- Project: [credit-risk-mlops](https://github.com/Asad-Farooq4421/credit-risk-mlops)

---

<div align="center">

**⚡ CYBER CREDIT v2.0.0 ⚡**

*Built with Streamlit + Three.js + Plotly*

</div>
