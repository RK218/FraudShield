"""
train_model.py
XGBoost with Optuna hyperparameter search, cost-sensitive threshold optimisation,
SHAP explainability, and model persistence.

Run:
    python src/train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import optuna
import shap
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, f1_score
)
from data_pipeline import (
    generate_transactions, engineer_features,
    prepare_data, FEATURE_COLS
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42


# ─────────────────────────────────────────────
# 1. Optuna objective
# ─────────────────────────────────────────────

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "verbosity":          0,
        "objective":          "binary:logistic",
        "eval_metric":        "aucpr",
        "seed":               SEED,
        "n_estimators":       trial.suggest_int("n_estimators", 200, 800),
        "max_depth":          trial.suggest_int("max_depth", 3, 9),
        "learning_rate":      trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
        "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        # class_weight to handle residual imbalance after SMOTE
        "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 1.0, 5.0),
    }
    model = xgb.XGBClassifier(**params, use_label_encoder=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=30,
    )
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


# ─────────────────────────────────────────────
# 2. Threshold optimisation (maximise F2 score)
# ─────────────────────────────────────────────

def find_optimal_threshold(y_true: np.ndarray,
                            y_proba: np.ndarray,
                            beta: float = 2.0) -> float:
    """
    Scan thresholds and return the one maximising F-beta score.
    beta=2 weighs recall twice as heavily as precision (fraud context).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    beta2 = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = ((1 + beta2) * precision * recall /
                  (beta2 * precision + recall))
    f_beta = np.nan_to_num(f_beta)
    best_idx = np.argmax(f_beta[:-1])        # last threshold excluded
    return float(thresholds[best_idx])


# ─────────────────────────────────────────────
# 3. Evaluation helpers
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test, threshold: float, label: str = ""):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    auc   = roc_auc_score(y_test, proba)
    cm    = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "auc_roc":     round(auc, 4),
        "precision":   round(tp / (tp + fp + 1e-9), 4),
        "recall":      round(tp / (tp + fn + 1e-9), 4),
        "fpr":         round(fp / (fp + tn + 1e-9), 4),
        "f1":          round(f1_score(y_test, preds), 4),
        "threshold":   round(threshold, 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }

    print(f"\n{'='*50}")
    print(f"  {label}  (threshold={threshold:.3f})")
    print(f"{'='*50}")
    print(f"  AUC-ROC : {metrics['auc_roc']}")
    print(f"  Recall  : {metrics['recall']}")
    print(f"  FPR     : {metrics['fpr']}")
    print(f"  F1      : {metrics['f1']}")
    print(f"  Confusion matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
    return metrics


# ─────────────────────────────────────────────
# 4. SHAP analysis
# ─────────────────────────────────────────────

def compute_shap(model, X_sample: np.ndarray,
                 feature_names: list) -> dict:
    """
    Returns a dict with mean absolute SHAP values per feature,
    sorted descending.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names
    ).sort_values(ascending=False)

    return {
        "shap_values":      shap_values,
        "expected_value":   float(explainer.expected_value),
        "feature_importance": importance.to_dict(),
    }


# ─────────────────────────────────────────────
# 5. Main training routine
# ─────────────────────────────────────────────

def train(n_trials: int = 30):
    os.makedirs("models", exist_ok=True)

    # ── Load or generate data ──
    if os.path.exists("data/X_train.npy"):
        print("Loading cached splits …")
        X_train = np.load("data/X_train.npy")
        X_test  = np.load("data/X_test.npy")
        y_train = np.load("data/y_train.npy")
        y_test  = np.load("data/y_test.npy")
    else:
        print("Generating data from scratch …")
        raw = generate_transactions(100_000)
        df  = engineer_features(raw)
        X_train, X_test, y_train, y_test = prepare_data(df)

    # Small validation fold from training set for Optuna
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    # ── Hyperparameter search ──
    print(f"\nRunning Optuna ({n_trials} trials) …")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params
    best_params.update({
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "seed": SEED,
        "use_label_encoder": False,
    })
    print(f"Best AUC-PR: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # ── Final model on full training data ──
    print("\nTraining final model …")
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # ── Threshold optimisation ──
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test, y_proba)

    # ── Evaluation ──
    metrics = evaluate(model, X_test, y_test, threshold, "Final XGBoost")

    # ── SHAP (on a random 2 000-row sample for speed) ──
    print("\nComputing SHAP values …")
    idx    = np.random.choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    shap_d = compute_shap(model, X_test[idx], FEATURE_COLS)
    print("\nTop-5 features by SHAP importance:")
    for feat, val in list(shap_d["feature_importance"].items())[:5]:
        print(f"  {feat:<30} {val:.4f}")

    # ── Persist artefacts ──
    joblib.dump(model, "models/xgb_fraud.pkl")
    np.save("models/shap_values_sample.npy", shap_d["shap_values"])
    np.save("models/X_shap_sample.npy",      X_test[idx])

    meta = {
        "threshold":          threshold,
        "expected_shap_value":shap_d["expected_value"],
        "feature_names":      FEATURE_COLS,
        "metrics":            metrics,
        "best_params":        {k: v for k, v in best_params.items()
                               if k not in ("use_label_encoder",)},
        "shap_importance":    shap_d["feature_importance"],
    }
    with open("models/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nModel artefacts saved to models/")
    return model, meta


if __name__ == "__main__":
    train(n_trials=30)
