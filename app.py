import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models dictionary
models = joblib.load("clv_regression_models.joblib")

st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Customer Lifetime Value Prediction")

st.markdown("""
Predict the next **90-Day Customer Lifetime Value** using RFM scores.  
The customer segment (**High / Medium / Low Value**) is **automatically detected** from your RFM inputs.
""")

if not models:
    st.error("❌ Error: No models found. Please ensure 'clv_regression_models.joblib' is in the correct directory.")
    st.stop()

# ─────────────────────────────────────────────
# RFM-based segment detection thresholds
# These percentile boundaries were derived from the Online Retail dataset
# Recency: lower = better | Frequency & Monetary: higher = better
# ─────────────────────────────────────────────
# Percentile boundaries (P33 / P66) for each RFM dimension:
RFM_BOUNDARIES = {
    "Recency":   {"p33": 17,    "p66": 75},      # days since last purchase
    "Frequency": {"p33": 21,    "p66": 69},      # number of transactions
    "Monetary":  {"p33": 319,   "p66": 1661},    # total spend
}

def score_rfm(recency, frequency, monetary):
    """
    Assign 1–3 scores for each RFM dimension, then sum → RFM score (3–9).
    Recency: lower days = higher score (3 is best).
    Frequency & Monetary: higher value = higher score.
    """
    # Recency score (inverted: fewer days is better)
    if recency <= RFM_BOUNDARIES["Recency"]["p33"]:
        r_score = 3
    elif recency <= RFM_BOUNDARIES["Recency"]["p66"]:
        r_score = 2
    else:
        r_score = 1

    # Frequency score
    if frequency <= RFM_BOUNDARIES["Frequency"]["p33"]:
        f_score = 1
    elif frequency <= RFM_BOUNDARIES["Frequency"]["p66"]:
        f_score = 2
    else:
        f_score = 3

    # Monetary score
    if monetary <= RFM_BOUNDARIES["Monetary"]["p33"]:
        m_score = 1
    elif monetary <= RFM_BOUNDARIES["Monetary"]["p66"]:
        m_score = 2
    else:
        m_score = 3

    total = r_score + f_score + m_score  # range: 3 – 9
    return r_score, f_score, m_score, total

def detect_segment(rfm_total):
    """Map total RFM score to segment label used when training the models."""
    if rfm_total >= 8:
        return "High Value"
    elif rfm_total >= 5:
        return "Medium Value"
    else:
        return "Low Value"

# ─────────────────────────────────────────────
# Input Fields
# ─────────────────────────────────────────────
st.subheader("Enter Customer RFM Data")

col1, col2, col3 = st.columns(3)

with col1:
    recency = st.number_input(
        "🕐 Recency (Days since last purchase)",
        min_value=0,
        value=30,
        help="Fewer days = more recent customer"
    )

with col2:
    frequency = st.number_input(
        "🔁 Frequency (Number of transactions)",
        min_value=1,
        value=10,
        help="More transactions = more loyal customer"
    )

with col3:
    monetary = st.number_input(
        "💰 Monetary Value (Total spend ₹)",
        min_value=0.0,
        value=1000.0,
        help="Higher spend = more valuable customer"
    )

# ─────────────────────────────────────────────
# Live RFM Score Preview
# ─────────────────────────────────────────────
r_score, f_score, m_score, rfm_total = score_rfm(recency, frequency, monetary)
auto_segment = detect_segment(rfm_total)

with st.expander("🔍 RFM Score Breakdown (auto-calculated)", expanded=True):
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("R Score", f"{r_score} / 3", help="3 = most recent")
    sc2.metric("F Score", f"{f_score} / 3", help="3 = most frequent")
    sc3.metric("M Score", f"{m_score} / 3", help="3 = highest spend")
    sc4.metric("RFM Total", f"{rfm_total} / 9")

    segment_color = {"High Value": "🟢", "Medium Value": "🟡", "Low Value": "🔴"}
    st.info(f"{segment_color[auto_segment]} **Auto-detected Segment: {auto_segment}**  \n"
            f"RFM Total Score: **{rfm_total}**  (High ≥ 8 · Medium 5–7 · Low ≤ 4)")

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if st.button("🚀 Predict Day CLV", use_container_width=True):

    model = models.get(auto_segment)

    if model is None:
        st.error(f"❌ No model found for segment: {auto_segment}")
        st.stop()

    input_df = pd.DataFrame({
        "Recency":   [recency],
        "Frequency": [frequency],
        "Monetary":  [monetary],
    })

    prediction = model.predict(input_df)[0]
    prediction = max(prediction, 0)  # CLV can't be negative

    st.divider()
    res1, res2 = st.columns(2)

    with res1:
        st.success(f"### Predicted 90-Day CLV: ₹{prediction:,.2f}")
        st.caption(f"Prediction made using the **{auto_segment}** Ridge Regression model")

    with res2:
        if auto_segment == "High Value":
            st.success("⭐ **High Value Customer**\nPrioritize retention & upsell campaigns.")
        elif auto_segment == "Medium Value":
            st.warning("👍 **Medium Value Customer**\nNurture with loyalty programs.")
        else:
            st.error("⚠️ **Low Value Customer**\nRe-engage with targeted win-back offers.")

    # RFM summary table
    st.divider()
    st.subheader("📊 Input Summary")
    summary = pd.DataFrame({
        "Metric":      ["Recency (days)", "Frequency", "Monetary (₹)", "R Score", "F Score", "M Score", "RFM Total", "Segment", "Predicted 90-Day CLV"],
        "Value":       [recency, frequency, f"₹{monetary:,.2f}", r_score, f_score, m_score, rfm_total, auto_segment, f"₹{prediction:,.2f}"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)