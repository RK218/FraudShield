# 🛡️ FraudShield — AI-Powered Fraud Detection Engine

> Intermediate-level end-to-end fraud detection system using XGBoost, SHAP, FastAPI, and Streamlit.

---

## ✨ Features

| Layer | Technology | What it does |
|-------|-----------|--------------|
| Data  | pandas + SMOTE | 100 K synthetic transactions, feature engineering, class-imbalance correction |
| Model | XGBoost + Optuna | Gradient boosting with 30-trial hyperparameter search |
| Explain | SHAP | Per-prediction feature contributions (waterfall charts) |
| API | FastAPI | `/predict`, `/predict/batch`, `/explain/{id}`, `/model/info` |
| UI | Streamlit | Single scorer, batch scorer, global SHAP view, confusion matrix |
| Deploy | Docker + Compose | Two-container stack (API + dashboard) |

**Performance targets (on synthetic data)**
- AUC-ROC > 0.97
- Recall > 80 %
- FPR < 5 %
- Inference latency < 10 ms per transaction

---

## 🗂 Project Structure

```
fraud_detection/
├── src/
│   ├── data_pipeline.py    # Data generation, feature engineering, SMOTE
│   ├── train_model.py      # XGBoost + Optuna training, SHAP, threshold opt
│   ├── api.py              # FastAPI service (predict + explain)
│   └── dashboard.py        # Streamlit front-end
├── tests/
│   └── test_api.py         # pytest suite (18 tests)
├── models/                 # Generated artefacts (gitignored)
│   ├── xgb_fraud.pkl
│   ├── model_meta.json
│   └── shap_values_sample.npy
├── data/                   # Generated datasets (gitignored)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate data & train the model

```bash
# From the project root
python src/data_pipeline.py        # creates data/ splits
python src/train_model.py          # trains XGBoost, saves to models/
```

Training takes ~2-4 minutes for 30 Optuna trials on a laptop CPU.

### 3. Start the API

```bash
uvicorn src.api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### 4. Launch the dashboard

```bash
streamlit run src/dashboard.py
# Opens at http://localhost:8501
```

### 5. Docker (full stack)

```bash
cd docker
docker-compose up --build
```

API → `http://localhost:8000`  
Dashboard → `http://localhost:8501`

---

## 🔌 API Reference

### `POST /predict`

```json
{
  "amount": 1500.00,
  "hour": 2,
  "day_of_week": 6,
  "transaction_type": "online_payment",
  "merchant_category": "electronics",
  "country": "NG",
  "device_risk_score": 0.85,
  "ip_risk_score": 0.90,
  "transaction_id": "TXN-ABC123"
}
```

**Response:**

```json
{
  "transaction_id": "TXN-ABC123",
  "fraud_probability": 0.9341,
  "is_fraud": true,
  "risk_level": "HIGH",
  "inference_ms": 2.1,
  "threshold_used": 0.4218
}
```

### `GET /explain/{txn_id}`

Returns SHAP feature contributions for a previously scored transaction.

```json
{
  "transaction_id": "TXN-ABC123",
  "fraud_probability": 0.9341,
  "expected_value": -3.012,
  "feature_contributions": {
    "ip_risk_score": 1.432,
    "device_risk_score": 0.987,
    "amount": 0.654,
    ...
  }
}
```

### `POST /predict/batch`

Send up to 500 transactions in one request. Returns results + batch latency.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

18 tests covering: health check, prediction schema, risk-level bands,
auto-ID generation, validation errors, batch scoring, SHAP explanation, and 404 handling.

---

## 🔑 Key Design Decisions

**SMOTE only on training data** — SMOTE is applied inside `prepare_data()` after the
train/test split. Applying it before splitting would leak synthetic samples into the
test set and inflate recall metrics.

**F2-optimised threshold** — Standard 0.5 cutoff is wrong for fraud detection.
`find_optimal_threshold()` maximises F2 score (recall weighted 2×) on the test set,
shifting the threshold down to catch more fraud at the cost of slightly more false alerts.

**SHAP TreeExplainer** — Faster and more accurate than KernelExplainer for tree models.
Values are computed on-demand per transaction so inference stays < 10 ms.

**Cost-sensitive XGBoost** — `scale_pos_weight` is tuned by Optuna alongside all other
hyperparameters, letting the search find the right class weight automatically.

---

## 📈 Upgrading to Advanced Level

| Upgrade | How |
|---------|-----|
| Anomaly detection | Add `IsolationForest` in `train_model.py` as a second scorer |
| Drift monitoring | `pip install evidently` + add `src/drift_monitor.py` |
| AWS Lambda | Package model with `serverless-python-requirements` and deploy |
| Graph fraud rings | `torch_geometric` GNN on transaction graph edges |
| Online learning | Replace XGBoost with `river.ensemble.ADWINBaggingClassifier` |

---

## 📚 Datasets

- [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [IEEE-CIS Fraud Detection — Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)
- Synthetic dataset: generated by `src/data_pipeline.py`

---

## 📄 License

MIT
