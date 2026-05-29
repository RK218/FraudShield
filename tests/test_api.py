"""
test_api.py  —  pytest unit tests for the FastAPI fraud detection service
Run: pytest tests/ -v
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

# ── patch model loading so tests don't need real artefacts ──
import sys, os, types, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Minimal stubs ────────────────────────────────────────────────
class _FakeModel:
    def predict_proba(self, X):
        # always return 80 % fraud probability for deterministic tests
        return np.array([[0.2, 0.8]] * len(X))

class _FakeExplainer:
    expected_value = -3.0
    def shap_values(self, X):
        return np.zeros((len(X), 13))

_fake_meta = {
    "threshold": 0.5,
    "feature_names": [
        "amount","log_amount","device_risk_score","ip_risk_score",
        "risk_composite","hour","day_of_week","is_weekend","is_night",
        "is_high_risk_country","transaction_type_enc",
        "merchant_category_enc","country_enc",
    ],
    "metrics": {"auc_roc":0.98,"recall":0.85,"fpr":0.02,"f1":0.82,
                "tp":100,"fp":10,"fn":15,"tn":9875},
    "best_params": {"n_estimators": 300},
    "shap_importance": {"amount": 0.45, "ip_risk_score": 0.30},
}

import api
api._model     = _FakeModel()
api._meta      = _fake_meta
api._explainer = _FakeExplainer()

from api import app
client = TestClient(app)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def valid_txn():
    return {
        "amount": 350.0,
        "hour": 14,
        "day_of_week": 2,
        "transaction_type": "purchase",
        "merchant_category": "grocery",
        "country": "IN",
        "device_risk_score": 0.2,
        "ip_risk_score": 0.1,
        "transaction_id": "TXN-TEST-001",
    }


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ─────────────────────────────────────────────
# Single prediction
# ─────────────────────────────────────────────

def test_predict_returns_200(valid_txn):
    r = client.post("/predict", json=valid_txn)
    assert r.status_code == 200

def test_predict_schema(valid_txn):
    r = client.post("/predict", json=valid_txn).json()
    assert "fraud_probability" in r
    assert "is_fraud" in r
    assert "risk_level" in r
    assert "inference_ms" in r

def test_predict_high_fraud(valid_txn):
    """Fake model always returns 0.80 → above 0.50 threshold → flagged."""
    r = client.post("/predict", json=valid_txn).json()
    assert r["is_fraud"] is True
    assert r["fraud_probability"] == pytest.approx(0.8, abs=0.01)
    assert r["risk_level"] == "HIGH"

def test_predict_risk_levels(valid_txn):
    """Patch threshold to check risk level bands."""
    original = api._meta["threshold"]
    api._meta["threshold"] = 0.9          # prob 0.8 < 0.9 → MEDIUM
    r = client.post("/predict", json=valid_txn).json()
    assert r["risk_level"] == "MEDIUM"
    api._meta["threshold"] = original

def test_predict_assigns_txn_id(valid_txn):
    """Supplied transaction_id is returned unchanged."""
    r = client.post("/predict", json=valid_txn).json()
    assert r["transaction_id"] == "TXN-TEST-001"

def test_predict_auto_txn_id():
    """Without an ID, one is auto-generated."""
    payload = {
        "amount": 10.0, "hour": 8, "day_of_week": 1,
        "transaction_type": "purchase", "merchant_category": "grocery",
        "country": "US", "device_risk_score": 0.1, "ip_risk_score": 0.05,
    }
    r = client.post("/predict", json=payload).json()
    assert r["transaction_id"].startswith("TXN-")

def test_predict_invalid_country(valid_txn):
    valid_txn["country"] = "ZZ"
    r = client.post("/predict", json=valid_txn)
    assert r.status_code == 422

def test_predict_invalid_amount(valid_txn):
    valid_txn["amount"] = -50
    r = client.post("/predict", json=valid_txn)
    assert r.status_code == 422


# ─────────────────────────────────────────────
# Batch prediction
# ─────────────────────────────────────────────

def test_batch_predict(valid_txn):
    payload = {"transactions": [valid_txn, {**valid_txn, "transaction_id": "TXN-TEST-002"}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    assert data["total_flagged"] == 2       # both flagged by fake model

def test_batch_returns_timing(valid_txn):
    payload = {"transactions": [valid_txn]}
    r = client.post("/predict/batch", json=payload).json()
    assert "batch_inference_ms" in r


# ─────────────────────────────────────────────
# SHAP explanation
# ─────────────────────────────────────────────

def test_explain_after_predict(valid_txn):
    client.post("/predict", json=valid_txn)             # populate store
    r = client.get(f"/explain/{valid_txn['transaction_id']}")
    assert r.status_code == 200
    body = r.json()
    assert "feature_contributions" in body
    assert "expected_value" in body

def test_explain_404_unknown():
    r = client.get("/explain/TXN-DOES-NOT-EXIST")
    assert r.status_code == 404


# ─────────────────────────────────────────────
# Model info
# ─────────────────────────────────────────────

def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "threshold" in data
    assert "metrics" in data
    assert "top_features" in data
