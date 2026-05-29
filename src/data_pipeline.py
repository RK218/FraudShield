"""
data_pipeline.py
Generates a synthetic fraud dataset and produces a clean, feature-engineered
DataFrame ready for model training.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib
import os

SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────
# 1. Synthetic data generation
# ─────────────────────────────────────────────

def generate_transactions(n: int = 100_000) -> pd.DataFrame:
    """
    Simulate a realistic bank transaction dataset with ~1 % fraud rate.
    Each row represents one transaction.
    """
    fraud_mask = np.random.rand(n) < 0.01          # 1 % fraud

    merchant_cats = ["grocery", "electronics", "travel", "dining",
                     "gas_station", "online_retail", "atm", "pharmacy"]
    countries     = ["IN", "US", "GB", "DE", "SG", "NG", "BR", "RU"]
    tx_types      = ["purchase", "withdrawal", "transfer", "online_payment"]

    # Base features
    df = pd.DataFrame({
        "transaction_id":   [f"TXN{i:07d}" for i in range(n)],
        "amount":           np.where(
            fraud_mask,
            np.random.exponential(scale=800, size=n).clip(50, 10_000),
            np.random.exponential(scale=150, size=n).clip(1, 5_000)
        ),
        "timestamp":        pd.date_range("2024-01-01", periods=n, freq="30s"),
        "transaction_type": np.random.choice(tx_types, n,
                                             p=[0.50, 0.15, 0.20, 0.15]),
        "merchant_category":np.random.choice(merchant_cats, n),
        "country":          np.where(
            fraud_mask,
            np.random.choice(["NG", "RU", "BR"], n),
            np.random.choice(countries, n, p=[0.35,0.25,0.10,0.10,0.05,0.05,0.05,0.05])
        ),
        "device_risk_score":np.where(
            fraud_mask,
            np.random.beta(5, 2, n),
            np.random.beta(2, 8, n)
        ),
        "ip_risk_score":    np.where(
            fraud_mask,
            np.random.beta(6, 2, n),
            np.random.beta(1, 9, n)
        ),
        "is_fraud":         fraud_mask.astype(int),
    })

    return df


# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["is_night"]    = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

    # Amount bins (log scale)
    df["log_amount"]  = np.log1p(df["amount"])

    # Risk composite
    df["risk_composite"] = (df["device_risk_score"] + df["ip_risk_score"]) / 2

    # High-risk country flag
    high_risk = {"NG", "RU", "BR"}
    df["is_high_risk_country"] = df["country"].isin(high_risk).astype(int)

    # Encode categoricals
    for col in ["transaction_type", "merchant_category", "country"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        joblib.dump(le, f"models/le_{col}.pkl")

    return df


# ─────────────────────────────────────────────
# 3. SMOTE + train/test split
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "amount", "log_amount", "device_risk_score", "ip_risk_score",
    "risk_composite", "hour", "day_of_week", "is_weekend", "is_night",
    "is_high_risk_country",
    "transaction_type_enc", "merchant_category_enc", "country_enc",
]

def prepare_data(df: pd.DataFrame,
                 apply_smote: bool = True,
                 test_size: float = 0.2):
    """
    Returns X_train, X_test, y_train, y_test (numpy arrays).
    SMOTE is applied only on the training fold to prevent leakage.
    """
    X = df[FEATURE_COLS].values
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    if apply_smote:
        sm = SMOTE(random_state=SEED, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[SMOTE] Resampled training set: {X_train.shape[0]:,} rows "
              f"({y_train.sum():,} fraud)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("data",   exist_ok=True)

    print("Generating synthetic transactions …")
    raw = generate_transactions(100_000)
    raw.to_csv("data/transactions_raw.csv", index=False)

    print("Engineering features …")
    processed = engineer_features(raw)
    processed.to_csv("data/transactions_processed.csv", index=False)

    print("Splitting data …")
    X_train, X_test, y_train, y_test = prepare_data(processed)

    # Persist splits for reproducibility
    np.save("data/X_train.npy", X_train)
    np.save("data/X_test.npy",  X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_test.npy",  y_test)
    print("Done. Splits saved to data/")
