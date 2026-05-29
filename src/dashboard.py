"""
dashboard.py  —  Streamlit front-end for the fraud detection API

Run:
    streamlit run src/dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FraudShield — AI Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://via.placeholder.com/180x50?text=FraudShield", use_column_width=True)
    st.markdown("---")
    page = st.radio("Navigate", ["🔍 Single Prediction",
                                  "📦 Batch Scoring",
                                  "📊 Model Insights",
                                  "⚙️ Model Info"])
    st.markdown("---")
    st.caption("API: " + API_BASE)

    # API health
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        st.success("API online ✓") if r.json().get("status") == "ok" else st.error("API error")
    except Exception:
        st.error("API offline — start with `uvicorn src.api:app`")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

MERCHANT_CATS = ["grocery","electronics","travel","dining",
                 "gas_station","online_retail","atm","pharmacy"]
COUNTRIES     = ["IN","US","GB","DE","SG","NG","BR","RU"]
TX_TYPES      = ["purchase","withdrawal","transfer","online_payment"]

def risk_badge(level: str):
    colours = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    return colours.get(level, "⚪")

def shap_waterfall(feature_contributions: dict,
                   expected_value: float,
                   fraud_prob: float):
    """Render a SHAP waterfall chart using Plotly."""
    feats  = list(feature_contributions.keys())[:12]
    values = [feature_contributions[f] for f in feats]

    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        measure=["relative"] * len(feats) + ["total"],
        y=feats + ["Prediction"],
        x=values + [sum(values)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#3B8BD4"}},
        increasing={"marker": {"color": "#E24B4A"}},
        totals={"marker": {"color": "#1D9E75"}},
    ))
    fig.update_layout(
        title=f"SHAP Waterfall  (P(fraud) = {fraud_prob:.4f})",
        xaxis_title="SHAP value (impact on model output)",
        height=450,
        margin=dict(l=200, r=20, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────
# PAGE 1 — Single Prediction
# ─────────────────────────────────────────────

if page == "🔍 Single Prediction":
    st.header("🔍 Single Transaction Scorer")
    st.caption("Fill in the transaction details below and click **Score**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Transaction")
        amount      = st.number_input("Amount (USD)", min_value=0.01, value=250.0, step=10.0)
        tx_type     = st.selectbox("Type", TX_TYPES)
        merchant    = st.selectbox("Merchant category", MERCHANT_CATS)
        country     = st.selectbox("Country", COUNTRIES)
        txn_id_input = st.text_input("Transaction ID (optional)", value="")

    with col2:
        st.subheader("Timing")
        hour        = st.slider("Hour of day", 0, 23, 14)
        dow         = st.slider("Day of week (0=Mon)", 0, 6, 2)

    with col3:
        st.subheader("Risk Signals")
        device_risk = st.slider("Device risk score", 0.0, 1.0, 0.2, 0.01)
        ip_risk     = st.slider("IP risk score",     0.0, 1.0, 0.1, 0.01)

    if st.button("🛡️ Score Transaction", type="primary"):
        payload = {
            "amount":            amount,
            "hour":              hour,
            "day_of_week":       dow,
            "transaction_type":  tx_type,
            "merchant_category": merchant,
            "country":           country,
            "device_risk_score": device_risk,
            "ip_risk_score":     ip_risk,
        }
        if txn_id_input:
            payload["transaction_id"] = txn_id_input

        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            result = resp.json()

            txn_id_out = result["transaction_id"]
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transaction ID", txn_id_out)
            c2.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
            c3.metric("Decision", "🚨 FRAUD" if result["is_fraud"] else "✅ Legit")
            c4.metric("Risk level", risk_badge(result["risk_level"]) + " " + result["risk_level"])
            st.caption(f"Latency: {result['inference_ms']} ms  |  "
                       f"Threshold: {result['threshold_used']}")

            # ── SHAP explanation ──
            with st.spinner("Fetching explanation …"):
                exp_resp = requests.get(f"{API_BASE}/explain/{txn_id_out}", timeout=5)
                if exp_resp.status_code == 200:
                    exp = exp_resp.json()
                    st.plotly_chart(
                        shap_waterfall(exp["feature_contributions"],
                                       exp["expected_value"],
                                       exp["fraud_probability"]),
                        use_container_width=True,
                    )
                else:
                    st.warning("Explanation not available.")

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure uvicorn is running.")
        except Exception as e:
            st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# PAGE 2 — Batch Scoring
# ─────────────────────────────────────────────

elif page == "📦 Batch Scoring":
    st.header("📦 Batch Transaction Scorer")
    st.caption("Upload a CSV or generate a random batch to score all at once.")

    mode = st.radio("Input mode", ["Generate random batch", "Upload CSV"])

    if mode == "Generate random batch":
        n = st.slider("Number of transactions", 10, 500, 50)
        if st.button("Generate & Score", type="primary"):
            np.random.seed(None)
            batch = []
            for _ in range(n):
                batch.append({
                    "amount":             float(np.random.exponential(200)),
                    "hour":               int(np.random.randint(0, 24)),
                    "day_of_week":        int(np.random.randint(0, 7)),
                    "transaction_type":   np.random.choice(TX_TYPES),
                    "merchant_category":  np.random.choice(MERCHANT_CATS),
                    "country":            np.random.choice(COUNTRIES),
                    "device_risk_score":  float(np.random.beta(2, 8)),
                    "ip_risk_score":      float(np.random.beta(1, 9)),
                })

            try:
                resp = requests.post(
                    f"{API_BASE}/predict/batch",
                    json={"transactions": batch},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                st.metric("Flagged", data["total_flagged"])
                st.metric("Batch latency", f"{data['batch_inference_ms']} ms")

                df = pd.DataFrame(data["results"])
                st.dataframe(
                    df.style.applymap(
                        lambda x: "background-color:#ffdddd" if x is True else "",
                        subset=["is_fraud"]
                    ),
                    use_container_width=True,
                )

                fig = px.histogram(df, x="fraud_probability", nbins=40,
                                   title="Distribution of fraud probabilities",
                                   color_discrete_sequence=["#534AB7"])
                st.plotly_chart(fig, use_container_width=True)
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API.")

    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.dataframe(df_up.head(5))
            st.caption("CSV preview — ensure column names match the API schema.")


# ─────────────────────────────────────────────
# PAGE 3 — Model Insights
# ─────────────────────────────────────────────

elif page == "📊 Model Insights":
    st.header("📊 Model Insights")
    try:
        info = requests.get(f"{API_BASE}/model/info", timeout=5).json()
        m    = info["metrics"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC",  m["auc_roc"])
        c2.metric("Recall",   m["recall"])
        c3.metric("FPR",      m["fpr"])
        c4.metric("F1",       m["f1"])

        st.markdown("---")
        st.subheader("Top feature importances (SHAP)")
        feats = pd.DataFrame(
            info["top_features"].items(),
            columns=["Feature", "Mean |SHAP|"]
        ).sort_values("Mean |SHAP|", ascending=True)

        fig = px.bar(feats, x="Mean |SHAP|", y="Feature",
                     orientation="h",
                     color="Mean |SHAP|",
                     color_continuous_scale="purples",
                     title="Global SHAP Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion matrix")
        tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
        cm_df = pd.DataFrame(
            {"Predicted Legit": [tn, fn], "Predicted Fraud": [fp, tp]},
            index=["Actual Legit", "Actual Fraud"]
        )
        st.dataframe(cm_df)

    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API.")


# ─────────────────────────────────────────────
# PAGE 4 — Model Info
# ─────────────────────────────────────────────

elif page == "⚙️ Model Info":
    st.header("⚙️ Model Configuration")
    try:
        info = requests.get(f"{API_BASE}/model/info", timeout=5).json()
        st.json(info["best_params"])
        st.caption(f"Decision threshold: **{info['threshold']:.4f}**")
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API.")
