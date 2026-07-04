"""
Anomify — Streamlit demo app
"""
import os
import sys
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

basic_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(basic_path))

from pipelines.inference import AnomifyLiveDetector  

st.set_page_config(page_title="Anomify", page_icon="🛰️", layout="wide")

DATA_PATH = basic_path / "data" / "raw" / "merged.csv"


def get_severity(mse: float, threshold: float) -> tuple[str, str]:
    ratio = mse / max(threshold, 1e-12)
    if ratio < 1:
        return "Normal ✅", "#2ecc71"
    elif ratio < 2:
        return "Low ⚠️", "#f1c40f"
    elif ratio < 5:
        return "Medium 🔶", "#e67e22"
    elif ratio < 10:
        return "High 🔴", "#e74c3c"
    else:
        return "Critical 🚨", "#8e44ad"

def rolling_health(results: pd.DataFrame, window: int = 30) -> pd.Series:
    return (1 - results["is_anomaly"].rolling(window, min_periods=1).mean()) * 100

@st.cache_resource(show_spinner="Loading model, scaler, and preprocessing pipeline...")
def load_detector() -> AnomifyLiveDetector:
    detector = AnomifyLiveDetector()
    models_dir = basic_path / "models"
    detector.engineer.scaler_path = str(models_dir / "scaler.pkl")
    detector.model_path = models_dir / "autoencoder.keras"
    detector.threshold_path = models_dir / "threshold.yaml"
    return detector

st.sidebar.title("🛰️ Anomify")
st.sidebar.caption("Real-Time IoT Anomaly Detection")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🤖 Assistant"])

detector = None
load_error = None
try:
    detector = load_detector()
    st.sidebar.metric("Active anomaly threshold (MSE)", f"{detector.threshold:.6f}")
except Exception as exc:
    load_error = exc
    st.sidebar.error(f"Model failed to load:\n{exc}")

# =============================================================================
# PAGE 1 — DASHBOARD
# =============================================================================
def dashboard_page():
    st.title("📊 Anomify — Live Detection Dashboard")
    st.markdown("Running the local SWaT dataset — cyberattacks mixed with normal operation.")

    if detector is None:
        st.warning(f"Model isn't loaded, so detection can't run.\n\n{load_error}")
        return

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at: {DATA_PATH}\n\nMake sure merged.csv is in data/raw/.")
        return

    num_records = st.slider("Window size", min_value=50, max_value=1000, value=300, step=10)

    progress_bar = st.progress(0.0, text="Scoring stream...")
    try:
        results = detector.analyze_stream(
            str(DATA_PATH),
            num_records=int(num_records),
            progress_callback=lambda p: progress_bar.progress(
                p, text=f"Scoring stream... {int(p*100)}%"
            ),
        )
        st.session_state["results"] = results
    except Exception as exc:
        st.error(f"Detection failed: {exc}")
    finally:
        progress_bar.empty()

    results = st.session_state.get("results")
    if results is None:
        return

    anomalies = results[results["is_anomaly"]]
    threshold = detector.threshold

    peak_mse = results["mse"].max()
    severity_label, severity_color = get_severity(peak_mse, threshold)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records analyzed", len(results))
    m2.metric("Anomalies flagged", len(anomalies))
    m3.metric("Anomaly rate", f"{len(anomalies) / max(len(results), 1):.1%}")
    m4.metric("Peak MSE", f"{peak_mse:.6f}")

    st.divider()

    st.subheader("📈 Reconstruction Error Over Time")
    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(x=results["step"], y=results["mse"], mode="lines", name="MSE", line=dict(color="#133457")))
    fig_mse.add_trace(go.Scatter(x=anomalies["step"], y=anomalies["mse"], mode="markers", name="Anomaly", marker=dict(color="#E45756", size=8, symbol="x")))
    fig_mse.add_hline(y=threshold, line_dash="dash", line_color="gray", annotation_text="Threshold")
    fig_mse.update_layout(height=350, margin=dict(t=10, b=20))
    st.plotly_chart(fig_mse, use_container_width=True)

    st.divider()

    st.subheader("🏥 System Health Score")
    health_series = rolling_health(results, window=30)
    fig_health = go.Figure()
    fig_health.add_trace(go.Scatter(x=results["step"], y=health_series, mode="lines", fill="tozeroy", line=dict(color="#2ecc71")))
    fig_health.add_hline(y=80, line_dash="dot", line_color="orange")
    fig_health.add_hline(y=50, line_dash="dot", line_color="red")
    fig_health.update_layout(yaxis=dict(range=[0, 105]), height=280)
    st.plotly_chart(fig_health, use_container_width=True)

    st.divider()

    st.subheader("🚨 Attack Severity")
    col_sev, col_gauge = st.columns([1, 2])
    with col_sev:
        st.markdown(f"<div style='background:{severity_color};padding:18px;border-radius:10px;text-align:center;color:white;'>{severity_label}</div>", unsafe_allow_html=True)
        st.metric("Peak MSE / Threshold", f"{peak_mse / max(threshold, 1e-12):.1f}×")

    with col_gauge:
        def step_severity(mse):
            r = mse / max(threshold, 1e-12)
            if r < 1: return "Normal"
            elif r < 2: return "Low"
            elif r < 5: return "Medium"
            elif r < 10: return "High"
            else: return "Critical"
        results["severity"] = results["mse"].apply(step_severity)
        sev_counts = results["severity"].value_counts().reindex(["Normal", "Low", "Medium", "High", "Critical"], fill_value=0)
        fig_sev = px.bar(x=sev_counts.index, y=sev_counts.values, color=sev_counts.index, color_discrete_map={"Normal": "#2ecc71", "Low": "#f1c40f", "Medium": "#e67e22", "High": "#e74c3c", "Critical": "#8e44ad"})
        fig_sev.update_layout(showlegend=False, height=260)
        st.plotly_chart(fig_sev, use_container_width=True)

# =============================================================================
# PAGE 2 — ASSISTANT (chatbot)
# =============================================================================
def assistant_page():
    st.title("🤖 Anomify — AI Assistant")
    st.markdown("هنا تقدر تسأل الـ AI Agent عن حالة النظام، الإنذارات اللي اتسجلت، أو تقارير الهجمات.")

    N8N_WEBHOOK_URL = "http://localhost:5678/webhook/anomify-chat"

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "أهلاً بيك! أنا المساعد الذكي، متصل بقاعدة بيانات Anomify. تحب تسأل عن إيه؟"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("إسأل عن الإنذارات، التقارير..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Agent is querying the database..."):
            try:
                payload = {"chatInput": prompt, "sessionId": "anomify_admin"}
                response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=20)
                if response.status_code == 200:
                    bot_reply = response.json().get("output", "تم استلام الطلب.")
                else:
                    bot_reply = f"❌ خطأ من الخادم (n8n): {response.status_code}"
            except Exception as e:
                bot_reply = "❌ لم أتمكن من الاتصال بـ n8n. تأكد من تشغيل الـ Workflow."

        with st.chat_message("assistant"):
            st.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# =============================================================================
# MAIN APP ROUTING
# =============================================================================
if __name__ =="__main__":
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "🤖 Assistant":
        assistant_page()