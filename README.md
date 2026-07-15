# Credit Card Fraud Detection & Transaction Risk Scoring

> **$38 billion.** That's the estimated annual cost of payment card fraud globally — and it's growing.  
> This project builds an end-to-end ML system that detects fraudulent transactions in real time and forecasts fraud volume spikes before they happen.

---

## The Problem

Traditional rule-based fraud systems generate excessive false positives and miss novel attack patterns. Financial institutions need models that:

- Score transaction risk continuously (not just flag/no-flag)
- Explain *why* a transaction is suspicious (regulatory requirement)
- Anticipate fraud velocity surges — not just react to them

---

## Solution Overview

This system combines two complementary models:

| Component | Approach | Output |
|---|---|---|
| **Fraud Detection** | XGBoost + TensorFlow (Keras) Neural Net | Risk score 0–1 per transaction |
| **Fraud Forecasting** | ARIMA + Prophet | Predicted fraud volume over time |

A live **Streamlit dashboard** accepts transaction inputs and returns a real-time risk score with explainability output.

---

## Results

| Model | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| XGBoost | 0.92 | 0.82 | 0.87 | 0.77 |
| Neural Net (TensorFlow) | 0.95 | 0.82 | 0.88 | 0.85 |


**Business translation:** At the optimal classification threshold, the model catches an estimated **$49,000 Euros saved per 285,000 transactions** while keeping false positive rates operationally acceptable.

---

## Why Precision-Recall Over Accuracy

With only ~0.17% of transactions being fraudulent, a model that predicts "not fraud" every time achieves 99.83% accuracy — and catches nothing. This project evaluates models exclusively on **Precision-Recall AUC**, which measures performance on the minority class that actually matters.

---

## Key Technical Decisions

**Class imbalance handling via SMOTE**  
The raw dataset is heavily skewed. Rather than undersampling legitimate transactions, SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud examples during training which will preserve signal without discarding data.

**Risk scoring over binary classification**  
Rather than a hard fraud/not-fraud label, every transaction receives a continuous risk score between 0 and 1. This mirrors how real-world fraud teams operate: high-risk transactions are reviewed, medium-risk flagged, low-risk approved automatically.

**SHAP explainability**  
Every prediction is accompanied by SHAP feature importance values. In regulated financial environments, models must be auditable — a black-box score alone is insufficient for compliance teams. For this project, though the Neural Network is fueling predictions, the XGBoost model helps explain the reasoning for decisions. SHAP analysis identified the V14 feature as the dominant fraud signal. Transactions with abnormally low V14 values typically will indicate fraud.

**Forecasting fraud velocity**  
The ARIMA/Prophet forecasting layer aggregates transaction data over time to predict when fraud volume is likely to spike. Catching a surge pattern 24–48 hours early allows fraud teams to tighten thresholds proactively.

Prophet MAE: 6.04
**ARIMA MAE**: 3.5

This forecasting layer is a key component in the business scenario as understanding when fraud spikes will happen as opposed to when they happened will greatly help the prevention of fraud. The biggest takeaway from this layer is **volume of data**. To properly forecast utilizing the ARIMA model, there needs to be more data. In a real-world environment, this will be a reality and will allow the model to do its thing.

---

## Tech Stack

```
Python · XGBoost · TensorFlow (Keras) · SHAP · imbalanced-learn (SMOTE)
Pandas · NumPy · Scikit-learn · Statsmodels · Prophet
Streamlit · Matplotlib · Seaborn · JSON
```

---

## Project Structure

```
fraud-risk-detection/
│
├── data/                        # Raw and processed transaction data (raw gitignored)
│
├── notebooks/
│   ├── 01_EDA.ipynb             # Class imbalance analysis, feature distributions
│   ├── 02_preprocessing.ipynb  # SMOTE, feature scaling, train/val/test split
│   ├── 03_modeling.ipynb        # XGBoost vs TensorFlow, PR curves, SHAP values
│   └── 04_forecasting.ipynb     # ARIMA + Prophet fraud volume forecasting
│
├── src/
│   ├── model.py                 # Training logic, predict_proba risk scoring
│   ├── features.py              # Feature engineering pipeline
│   └── forecasting.py           # Time aggregation, model fitting, predictions
│
├── models/                      # Saved model artifacts (.pkl / .pt)
├── app.py                       # Streamlit dashboard — live risk scoring
├── config.yaml                  # Model parameters, thresholds, file paths
├── requirements.txt
└── README.md
```

---

## Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
```

The dashboard accepts transaction feature inputs and returns:
- A **risk score** (0–1)
- A **risk tier** (Low / Medium / High)
- **SHAP waterfall chart** explaining the top contributing features

---

## Data

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 real transactions from European cardholders, with 492 fraud cases (0.172%).

Features V1–V28 are PCA-transformed for confidentiality. `Amount` and `Time` are the only raw features.

---

## Live Demo

🔗 **[Launch Streamlit App](#)** *(link to be added after deployment)*

---

## About

Built by **Corbin Couger** — Data Scientist with 5+ years of experience in predictive modeling, anomaly detection, and forecasting across healthcare, energy, and marketing analytics.

Currently pursuing a Professional Certificate in AI & ML from Purdue University / IBM.

[Portfolio](https://www.couger.me) · [LinkedIn](https://www.linkedin.com/in/corbincouger/) · [corbin.0007@yahoo.com](mailto:corbin.0007@yahoo.com)
