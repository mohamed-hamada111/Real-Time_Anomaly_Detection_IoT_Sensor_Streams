"""
Anomify — Streamlit demo app (runs on the local SWaT dataset).

Two tabs:
  1. Dashboard  - runs merged.csv through the REAL trained
                  pipeline (preprocess -> feature engineer -> LSTM
                  autoencoder) and plots reconstruction error per step
                  against the anomaly threshold. Also shows:
                    - System Health Score (rolling %)
                    - Attack Severity (how far MSE exceeds threshold)
                    - Top sensors driving each anomaly (per-feature MSE)
  2. Assistant  - chatbot that can answer questions about the project
                  and the current results on screen.

"""
import os
import sys
from pathlib import Path

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
    """Return (label, colour) based on how many times MSE exceeds threshold."""
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
    """
    System Health Score per step: percentage of the last `window` steps that
    were normal, expressed as 0–100.
    """
    return (1 - results["is_anomaly"].rolling(window, min_periods=1).mean()) * 100


#  model loading 

@st.cache_resource(show_spinner="Loading model, scaler, and preprocessing pipeline...")
def load_detector() -> AnomifyLiveDetector:
    detector = AnomifyLiveDetector()
    models_dir = basic_path / "models"
    detector.engineer.scaler_path = str(models_dir / "scaler.pkl")
    detector.model_path = models_dir / "autoencoder.keras"
    detector.threshold_path = models_dir / "threshold.yaml"
    return detector


# sidebar 

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
    st.markdown(
        "Running the local SWaT dataset — "
        "this file contains deliberately launched cyberattacks mixed with normal "
        "operation, used to test detection."
    )

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

    # 1. Summary metrics 
    peak_mse = results["mse"].max()
    severity_label, severity_color = get_severity(peak_mse, threshold)
    health_now = int(rolling_health(results).iloc[-1])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Records analyzed", len(results))
    m2.metric("Anomalies flagged", len(anomalies))
    m3.metric("Anomaly rate", f"{len(anomalies) / max(len(results), 1):.1%}")
    m4.metric("Peak MSE", f"{peak_mse:.6f}")

    st.divider()

    # 2. MSE over time 
    st.subheader("📈 Reconstruction Error Over Time")
    fig_mse = go.Figure()
    fig_mse.add_trace(go.Scatter(
        x=results["step"], y=results["mse"], mode="lines",
        name="Reconstruction error (MSE)", line=dict(color="#133457"),
    ))
    fig_mse.add_trace(go.Scatter(
        x=anomalies["step"], y=anomalies["mse"], mode="markers",
        name="Flagged anomaly", marker=dict(color="#E45756", size=8, symbol="x"),
    ))
    fig_mse.add_hline(
        y=threshold, line_dash="dash", line_color="gray",
        annotation_text="Anomaly threshold", annotation_position="top left",
    )
    fig_mse.update_layout(
        xaxis_title="Step", yaxis_title="Reconstruction error (MSE)",
        height=350, margin=dict(t=10, b=20), legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_mse, use_container_width=True)

    st.divider()

    # 3. System Health Score 
    st.subheader("🏥 System Health Score")
    st.caption("Rolling percentage of normal windows (last 30 steps). 100 = fully healthy, 0 = everything flagged.")

    health_series = rolling_health(results, window=30)
    fig_health = go.Figure()
    fig_health.add_trace(go.Scatter(
        x=results["step"], y=health_series, mode="lines", fill="tozeroy",
        name="Health Score",
        line=dict(color="#2ecc71"),
        fillcolor="rgba(46,204,113,0.15)",
    ))
    fig_health.add_hline(y=80, line_dash="dot", line_color="orange",
                         annotation_text="Warning threshold (80)", annotation_position="top left")
    fig_health.add_hline(y=50, line_dash="dot", line_color="red",
                         annotation_text="Critical threshold (50)", annotation_position="bottom left")
    fig_health.update_layout(
        xaxis_title="Step", yaxis_title="Health Score (0–100)",
        yaxis=dict(range=[0, 105]),
        height=280, margin=dict(t=10, b=20),
    )
    st.plotly_chart(fig_health, use_container_width=True)

    st.divider()

    # 4. Attack Severity 
    st.subheader("🚨 Attack Severity")
    st.caption("Based on how many times the peak MSE exceeds the trained threshold.")

    col_sev, col_gauge = st.columns([1, 2])
    with col_sev:
        st.markdown(
            f"<div style='background:{severity_color};padding:18px 24px;"
            f"border-radius:10px;text-align:center;"
            f"font-size:1.4rem;font-weight:700;color:white;'>"
            f"{severity_label}</div>",
            unsafe_allow_html=True,
        )
        ratio = peak_mse / max(threshold, 1e-12)
        st.metric("Peak MSE / Threshold", f"{ratio:.1f}×", delta=None)
        

    with col_gauge:
        # Severity breakdown per step
        def step_severity(mse):
            r = mse / max(threshold, 1e-12)
            if r < 1:   return "Normal"
            elif r < 2: return "Low"
            elif r < 3: return "Medium"
            elif r < 4: return "High"
            else:        return "Critical"

        results["severity"] = results["mse"].apply(step_severity)
        sev_counts = results["severity"].value_counts().reindex(
            ["Normal", "Low", "Medium", "High", "Critical"], fill_value=0
        )
        fig_sev = px.bar(
            x=sev_counts.index, y=sev_counts.values,
            color=sev_counts.index,
            color_discrete_map={
                "Normal": "#2ecc71", "Low": "#f1c40f",
                "Medium": "#e67e22", "High": "#e74c3c", "Critical": "#8e44ad",
            },
            labels={"x": "Severity Level", "y": "Step count"},
        )
        fig_sev.update_layout(
            showlegend=False, height=260, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_sev, use_container_width=True)

    st.divider()

    # display the table 
    with st.expander("📋 Raw results table"):
        display_cols = ["step", "mse", "threshold", "is_anomaly", "severity", "true_label"]
        st.dataframe(results[[c for c in display_cols if c in results.columns]],
                        use_container_width=True)




# =============================================================================
# PAGE 2 — ASSISTANT (chatbot)
# =============================================================================





if __name__ =="__main__":
    dashboard_page()
