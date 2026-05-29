"""
api.py  —  FastAPI fraud-detection service

Endpoints
---------
GET  /health              → liveness check
POST /predict             → score a single transaction
POST /predict/batch       → score a list of transactions
GET  /explain/{txn_id}    → SHAP explanation for last-seen transaction
GET  /model/info          → model metadata (threshold, metrics, top features)

Run locally:
    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations
import json, time, uuid, logging
from typing import Optional, List
from functools import lru_cache

import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App & CORS
# ─────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring with SHAP explainability",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Model loading (lazy, cached)
# ─────────────────────────────────────────────

_model    = None
_meta     = None
_explainer = None
_explanation_store: dict = {}          # txn_id → last explanation

def load_artefacts():
    global _model, _meta, _explainer
    if _model is None:
        logger.info("Loading model artefacts …")
        _model = joblib.load("models/xgb_fraud.pkl")
        with open("models/model_meta.json") as f:
            _meta = json.load(f)
        _explainer = shap.TreeExplainer(_model)
        logger.info("Model loaded. Threshold = %.4f", _meta["threshold"])

@app.on_event("startup")
async def startup():
    load_artefacts()


# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

MERCHANT_CATS = ["grocery","electronics","travel","dining",
                 "gas_station","online_retail","atm","pharmacy"]
COUNTRIES     = ["IN","US","GB","DE","SG","NG","BR","RU"]
TX_TYPES      = ["purchase","withdrawal","transfer","online_payment"]

class TransactionIn(BaseModel):
    amount:             float  = Field(..., gt=0, description="Transaction amount (USD)")
    hour:               int    = Field(..., ge=0, le=23)
    day_of_week:        int    = Field(..., ge=0, le=6, description="0=Mon … 6=Sun")
    transaction_type:   str    = Field(..., description=f"One of {TX_TYPES}")
    merchant_category:  str    = Field(..., description=f"One of {MERCHANT_CATS}")
    country:            str    = Field(..., description=f"One of {COUNTRIES}")
    device_risk_score:  float  = Field(..., ge=0.0, le=1.0)
    ip_risk_score:      float  = Field(..., ge=0.0, le=1.0)
    transaction_id:     Optional[str] = None

    @validator("transaction_type")
    def check_tx_type(cls, v):
        if v not in TX_TYPES:
            raise ValueError(f"transaction_type must be one of {TX_TYPES}")
        return v

    @validator("merchant_category")
    def check_merchant(cls, v):
        if v not in MERCHANT_CATS:
            raise ValueError(f"merchant_category must be one of {MERCHANT_CATS}")
        return v

    @validator("country")
    def check_country(cls, v):
        if v not in COUNTRIES:
            raise ValueError(f"country must be one of {COUNTRIES}")
        return v


class PredictionOut(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str          # LOW / MEDIUM / HIGH
    inference_ms: float
    threshold_used: float


class BatchIn(BaseModel):
    transactions: List[TransactionIn]


class BatchOut(BaseModel):
    results: List[PredictionOut]
    total_flagged: int
    batch_inference_ms: float


class ExplanationOut(BaseModel):
    transaction_id: str
    fraud_probability: float
    expected_value: float
    feature_contributions: dict   # feature → SHAP value


# ─────────────────────────────────────────────
# Feature engineering (mirrors data_pipeline.py)
# ─────────────────────────────────────────────

TX_TYPE_MAP      = {v: i for i, v in enumerate(TX_TYPES)}
MERCHANT_CAT_MAP = {v: i for i, v in enumerate(MERCHANT_CATS)}
COUNTRY_MAP      = {v: i for i, v in enumerate(COUNTRIES)}
HIGH_RISK        = {"NG", "RU", "BR"}

def featurise(t: TransactionIn) -> np.ndarray:
    log_amount     = np.log1p(t.amount)
    is_weekend     = int(t.day_of_week >= 5)
    is_night       = int(t.hour < 6 or t.hour > 22)
    risk_composite = (t.device_risk_score + t.ip_risk_score) / 2
    is_hr_country  = int(t.country in HIGH_RISK)
    tx_enc         = TX_TYPE_MAP.get(t.transaction_type, 0)
    mc_enc         = MERCHANT_CAT_MAP.get(t.merchant_category, 0)
    co_enc         = COUNTRY_MAP.get(t.country, 0)

    return np.array([[
        t.amount, log_amount, t.device_risk_score, t.ip_risk_score,
        risk_composite, t.hour, t.day_of_week, is_weekend, is_night,
        is_hr_country, tx_enc, mc_enc, co_enc,
    ]], dtype=np.float32)


def risk_level(prob: float, threshold: float) -> str:
    if prob < threshold * 0.5:
        return "LOW"
    if prob < threshold:
        return "MEDIUM"
    return "HIGH"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/model/info")
def model_info():
    load_artefacts()
    return {
        "threshold": _meta["threshold"],
        "metrics":   _meta["metrics"],
        "top_features": dict(
            list(_meta["shap_importance"].items())[:10]
        ),
        "best_params": _meta["best_params"],
    }


@app.post("/predict", response_model=PredictionOut)
def predict(txn: TransactionIn):
    load_artefacts()
    t0    = time.perf_counter()
    X     = featurise(txn)
    prob  = float(_model.predict_proba(X)[0, 1])
    ms    = round((time.perf_counter() - t0) * 1000, 2)

    txn_id = txn.transaction_id or f"TXN-{uuid.uuid4().hex[:8].upper()}"
    threshold = _meta["threshold"]

    # Store for /explain endpoint
    _explanation_store[txn_id] = {"X": X, "prob": prob}

    return PredictionOut(
        transaction_id    = txn_id,
        fraud_probability = round(prob, 4),
        is_fraud          = prob >= threshold,
        risk_level        = risk_level(prob, threshold),
        inference_ms      = ms,
        threshold_used    = threshold,
    )


@app.post("/predict/batch", response_model=BatchOut)
def predict_batch(body: BatchIn):
    load_artefacts()
    t0        = time.perf_counter()
    threshold = _meta["threshold"]
    results   = []

    for txn in body.transactions:
        X     = featurise(txn)
        prob  = float(_model.predict_proba(X)[0, 1])
        txn_id = txn.transaction_id or f"TXN-{uuid.uuid4().hex[:8].upper()}"
        _explanation_store[txn_id] = {"X": X, "prob": prob}
        results.append(PredictionOut(
            transaction_id    = txn_id,
            fraud_probability = round(prob, 4),
            is_fraud          = prob >= threshold,
            risk_level        = risk_level(prob, threshold),
            inference_ms      = 0,
            threshold_used    = threshold,
        ))

    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    return BatchOut(
        results            = results,
        total_flagged      = sum(r.is_fraud for r in results),
        batch_inference_ms = total_ms,
    )


@app.get("/explain/{txn_id}", response_model=ExplanationOut)
def explain(txn_id: str):
    load_artefacts()
    if txn_id not in _explanation_store:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {txn_id} not found. Call /predict first."
        )
    entry = _explanation_store[txn_id]
    X     = entry["X"]

    sv = _explainer.shap_values(X)[0]          # shape: (n_features,)
    feature_names = _meta["feature_names"]
    contribs = {
        feature_names[i]: round(float(sv[i]), 6)
        for i in range(len(feature_names))
    }
    # Sort by absolute impact
    contribs = dict(sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True))

    return ExplanationOut(
        transaction_id      = txn_id,
        fraud_probability   = round(entry["prob"], 4),
        expected_value      = round(float(_explainer.expected_value), 6),
        feature_contributions = contribs,
    )
