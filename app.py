import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import load_wine


# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="Wine Classifier Pro",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Premium Styles
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Playfair+Display:wght@700;900&display=swap');

    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: #0a0118;
        background-image:
            radial-gradient(at 0% 0%, rgba(168, 85, 247, 0.25) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.20) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(139, 92, 246, 0.20) 0px, transparent 50%);
        background-attachment: fixed;
        color: #f5f3ff;
    }

    /* Animated glow overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background: radial-gradient(circle at 50% 50%, rgba(168,85,247,0.08), transparent 70%);
        pointer-events: none;
        animation: pulseGlow 8s ease-in-out infinite;
        z-index: 0;
    }

    @keyframes pulseGlow {
        0%, 100% { opacity: 0.4; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
        position: relative;
        z-index: 1;
    }

    /* =========================================================
       HERO — fully centered fix
       ========================================================= */
    .hero-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2rem 1rem 2.5rem;
        margin: 0 auto 1.5rem auto;
        width: 100%;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        background: rgba(168, 85, 247, 0.12);
        border: 1px solid rgba(168, 85, 247, 0.35);
        border-radius: 100px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #d8b4fe;
        margin: 0 auto 1.25rem auto;
        backdrop-filter: blur(10px);
    }

    .hero-badge::before {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 12px #22c55e;
        animation: blink 2s ease-in-out infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    /* Force centered hero title — overrides Streamlit defaults */
    .hero-wrap h1.hero-title,
    h1.hero-title {
        display: block !important;
        width: 100% !important;
        text-align: center !important;
        font-family: 'Playfair Display', serif !important;
        font-size: clamp(2.5rem, 5vw, 4.25rem) !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #f9a8d4 0%, #c4b5fd 35%, #93c5fd 70%, #f9a8d4 100%) !important;
        background-size: 200% 200% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin: 0 auto 0.75rem auto !important;
        padding: 0 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.05 !important;
        animation: gradientShift 6s ease-in-out infinite;
    }

    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .hero-subtitle,
    p.hero-subtitle {
        display: block !important;
        width: 100% !important;
        max-width: 620px !important;
        text-align: center !important;
        color: rgba(245, 243, 255, 0.65) !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        margin: 0 auto !important;
        padding: 0 !important;
        line-height: 1.6 !important;
    }

    .hero-divider {
        display: block;
        width: 60px;
        height: 3px;
        margin: 1.5rem auto 0;
        background: linear-gradient(90deg, transparent, #a855f7, #ec4899, transparent);
        border-radius: 3px;
    }

    /* ---------- Glass Cards ---------- */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(24px) saturate(150%);
        -webkit-backdrop-filter: blur(24px) saturate(150%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 24px;
        padding: 1.75rem;
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }

    /* ---------- Section headers ---------- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #f5f3ff;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 1.5rem 0;
        letter-spacing: -0.01em;
    }

    .section-header-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, rgba(168,85,247,0.25), rgba(236,72,153,0.15));
        border: 1px solid rgba(168,85,247,0.3);
        border-radius: 12px;
        font-size: 1.1rem;
        flex-shrink: 0;
    }

    .section-header-text {
        display: flex;
        flex-direction: column;
        line-height: 1.2;
    }

    .section-header-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f5f3ff;
    }

    .section-header-sub {
        font-size: 0.78rem;
        font-weight: 400;
        color: rgba(245,243,255,0.5);
        letter-spacing: 0.02em;
    }

    /* ---------- Result Card ---------- */
    .result-card {
        background:
            linear-gradient(135deg, rgba(168,85,247,0.18), rgba(236,72,153,0.10)) padding-box,
            linear-gradient(135deg, rgba(168,85,247,0.6), rgba(236,72,153,0.4), rgba(59,130,246,0.4)) border-box;
        border: 1.5px solid transparent;
        backdrop-filter: blur(30px) saturate(160%);
        -webkit-backdrop-filter: blur(30px) saturate(160%);
        border-radius: 28px;
        padding: 2.5rem 2rem 2rem;
        text-align: center;
        box-shadow:
            0 30px 80px rgba(168, 85, 247, 0.25),
            0 10px 30px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255,255,255,0.1);
        margin-bottom: 1.75rem;
        position: relative;
        overflow: hidden;
        animation: cardFadeIn 0.6s ease-out;
    }

    @keyframes cardFadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 50% 0%, rgba(249, 168, 212, 0.15), transparent 40%);
        pointer-events: none;
    }

    .result-label {
        color: rgba(245,243,255,0.55);
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.22em;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
    }

    .result-value {
        font-family: 'Playfair Display', serif;
        font-size: clamp(2.5rem, 4vw, 3.5rem);
        font-weight: 900;
        background: linear-gradient(135deg, #f9a8d4, #c4b5fd, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
        position: relative;
        z-index: 1;
        letter-spacing: -0.02em;
    }

    .result-meta {
        color: rgba(245,243,255,0.75);
        font-size: 0.95rem;
        margin-top: 1rem;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }

    .result-meta b {
        color: #f9a8d4;
        font-weight: 700;
    }

    .result-confidence-bar {
        margin: 1.25rem auto 0;
        max-width: 320px;
        height: 6px;
        background: rgba(255,255,255,0.08);
        border-radius: 100px;
        overflow: hidden;
        position: relative;
        z-index: 1;
    }

    .result-confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #a855f7, #ec4899, #f9a8d4);
        border-radius: 100px;
        box-shadow: 0 0 20px rgba(236,72,153,0.6);
        animation: fillBar 1s ease-out;
    }

    @keyframes fillBar {
        from { width: 0%; }
    }

    /* ---------- Inputs ---------- */
    .stNumberInput {
        margin-bottom: 0.25rem;
    }

    .stNumberInput > label {
        color: rgba(233, 213, 255, 0.9) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 0.35rem !important;
    }

    .stNumberInput input {
        background: rgba(255,255,255,0.04) !important;
        color: #f5f3ff !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 0.65rem 0.9rem !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stNumberInput input:hover {
        border-color: rgba(168,85,247,0.4) !important;
        background: rgba(255,255,255,0.06) !important;
    }

    .stNumberInput input:focus {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 3px rgba(168,85,247,0.18) !important;
        background: rgba(255,255,255,0.07) !important;
    }

    .stNumberInput button {
        background: rgba(168,85,247,0.15) !important;
        border: none !important;
        color: #e9d5ff !important;
        border-radius: 8px !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1.25rem !important;
        border: none !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.01em !important;
        height: auto !important;
        min-height: 48px !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #a855f7 0%, #ec4899 100%) !important;
        color: white !important;
        box-shadow:
            0 10px 30px rgba(168,85,247,0.35),
            inset 0 1px 0 rgba(255,255,255,0.2) !important;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow:
            0 16px 40px rgba(168,85,247,0.55),
            inset 0 1px 0 rgba(255,255,255,0.3) !important;
        filter: brightness(1.08);
    }

    .stButton > button[kind="primary"]:active {
        transform: translateY(0) !important;
    }

    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.05) !important;
        color: #e9d5ff !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: rgba(168,85,247,0.15) !important;
        border-color: rgba(168,85,247,0.4) !important;
        color: #f9a8d4 !important;
        transform: translateY(-2px) !important;
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        backdrop-filter: blur(20px);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #a855f7, #ec4899);
        opacity: 0.6;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(168,85,247,0.3);
        background: rgba(255,255,255,0.06);
        transform: translateY(-2px);
    }

    div[data-testid="stMetricLabel"] {
        color: rgba(245,243,255,0.55) !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f9a8d4 !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
    }

    /* ---------- Dataframe ---------- */
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.03);
        border-radius: 18px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        overflow: hidden;
    }

    /* ---------- Expander ---------- */
    .streamlit-expanderHeader,
    details > summary {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 14px !important;
        color: #e9d5ff !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        transition: all 0.2s ease !important;
    }

    .streamlit-expanderHeader:hover,
    details > summary:hover {
        background: rgba(168,85,247,0.12) !important;
        border-color: rgba(168,85,247,0.3) !important;
    }

    details[open] > summary {
        background: rgba(168,85,247,0.1) !important;
        border-color: rgba(168,85,247,0.3) !important;
    }

    /* ---------- Alerts ---------- */
    .stAlert {
        background: rgba(168,85,247,0.08) !important;
        border: 1px solid rgba(168,85,247,0.25) !important;
        border-radius: 16px !important;
        color: #e9d5ff !important;
        backdrop-filter: blur(20px);
    }

    /* ---------- Form ---------- */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #a855f7, #ec4899);
        border-radius: 10px;
        border: 2px solid transparent;
        background-clip: padding-box;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #c084fc, #f472b6);
        background-clip: padding-box;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: rgba(245,243,255,0.35);
        margin-top: 3rem;
        padding-top: 2rem;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
        border-top: 1px solid rgba(255,255,255,0.06);
    }

    .footer-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        background: rgba(168,85,247,0.1);
        border: 1px solid rgba(168,85,247,0.2);
        border-radius: 100px;
        color: #d8b4fe;
        font-weight: 500;
        margin: 0 0.25rem;
    }

    /* ---------- Hide Streamlit chrome ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}

    hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 1.5rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# Load resources
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    with open("random.pkl", "rb") as f:
        obj = pickle.load(f)

    if not hasattr(obj, "predict"):
        raise TypeError(
            "random.pkl does not contain a trained model. "
            "Run train_model.py to regenerate it."
        )
    return obj


@st.cache_data(show_spinner=False)
def load_feature_names():
    with open("feature_names.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_dataset():
    wine = load_wine()
    return wine.data, list(wine.target_names)


try:
    model = load_model()
    feature_names = load_feature_names()
    wine_data, target_names = load_dataset()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()


# =========================================================
# Pretty feature labels
# =========================================================

PRETTY_NAMES = {
    "alcohol": "Alcohol",
    "malic_acid": "Malic Acid",
    "ash": "Ash",
    "alcalinity_of_ash": "Alcalinity of Ash",
    "magnesium": "Magnesium",
    "total_phenols": "Total Phenols",
    "flavanoids": "Flavanoids",
    "nonflavanoid_phenols": "Nonflavanoid Phenols",
    "proanthocyanins": "Proanthocyanins",
    "color_intensity": "Color Intensity",
    "hue": "Hue",
    "od280/od315_of_diluted_wines": "OD280 / OD315",
    "proline": "Proline",
}

FEATURE_ICONS = {
    "alcohol": "🍷",
    "malic_acid": "🍏",
    "ash": "🌫️",
    "alcalinity_of_ash": "⚗️",
    "magnesium": "✨",
    "total_phenols": "🌿",
    "flavanoids": "🍇",
    "nonflavanoid_phenols": "🌱",
    "proanthocyanins": "🫐",
    "color_intensity": "🎨",
    "hue": "🌈",
    "od280/od315_of_diluted_wines": "🔬",
    "proline": "🧬",
}

def pretty(name):
    return PRETTY_NAMES.get(name, name.replace("_", " ").title())

def icon(name):
    return FEATURE_ICONS.get(name, "🧪")


# =========================================================
# Hero (centered)
# =========================================================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">AI-Powered Analysis</div>
    <h1 class="hero-title">Wine Classifier Pro</h1>
    <p class="hero-subtitle">
        Enter the 13 chemical properties below and let our machine learning model
        identify the exact wine cultivar with precision and confidence.
    </p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# Session state
# =========================================================

if "form_key" not in st.session_state:
    st.session_state.form_key = 0


# =========================================================
# Input form
# =========================================================

st.markdown("""
<div class="section-header">
    <div class="section-header-icon">🧪</div>
    <div class="section-header-text">
        <div class="section-header-title">Chemical Properties</div>
        <div class="section-header-sub">Adjust the 13 features — defaults are set to the training mean</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.form(key=f"wine_form_{st.session_state.form_key}"):
    input_values = {}
    cols_per_row = 3
    rows = (len(feature_names) + cols_per_row - 1) // cols_per_row

    for r in range(rows):
        cols = st.columns(cols_per_row, gap="medium")
        for c in range(cols_per_row):
            i = r * cols_per_row + c
            if i >= len(feature_names):
                break
            fname = feature_names[i]
            mean = float(wine_data[:, i].mean())
            std = float(wine_data[:, i].std())
            lo, hi = mean - 3 * std, mean + 3 * std
            step = 0.01 if std < 1 else (0.1 if std < 10 else 1.0)

            with cols[c]:
                input_values[fname] = st.number_input(
                    label=f"{icon(fname)}  {pretty(fname)}",
                    min_value=float(round(lo, 3)),
                    max_value=float(round(hi, 3)),
                    value=float(round(mean, 3)),
                    step=float(step),
                    format="%.4f",
                    help=f"Training mean: {mean:.2f} | std: {std:.2f}",
                )

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([2, 2, 1], gap="small")
    with b1:
        predict_clicked = st.form_submit_button(
            "🔮  Predict Cultivar",
            type="primary",
            use_container_width=True,
        )
    with b2:
        reset_clicked = st.form_submit_button(
            "♻️  Reset All",
            use_container_width=True,
        )
    with b3:
        load_mean_clicked = st.form_submit_button(
            "📊  Mean",
            use_container_width=True,
        )


if reset_clicked:
    st.session_state.form_key += 1
    st.rerun()

if load_mean_clicked:
    st.session_state.form_key += 1
    st.rerun()


# =========================================================
# Prediction helpers
# =========================================================

def predict(features):
    arr = np.array(features, dtype=float).reshape(1, -1)
    pred = int(model.predict(arr)[0])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr)[0]
    else:
        probs = np.zeros(len(target_names))
        probs[pred] = 1.0
    return pred, probs


# =========================================================
# Results
# =========================================================

if predict_clicked:
    ordered_values = [input_values[f] for f in feature_names]

    try:
        prediction, probabilities = predict(ordered_values)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    confidence = float(probabilities[prediction]) * 100
    predicted_label = target_names[prediction].replace("class_", "Class ")

    # ---- Result card ----
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Cultivar</div>
        <p class="result-value">🍷 {predicted_label}</p>
        <div class="result-meta">
            Confidence: <b>{confidence:.2f}%</b> &nbsp;•&nbsp; {len(target_names)} possible classes
        </div>
        <div class="result-confidence-bar">
            <div class="result-confidence-fill" style="width: {confidence}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- KPI metrics ----
    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("🎯 Cultivar", predicted_label)
    m2.metric("🔢 Class Index", prediction)
    m3.metric("💯 Confidence", f"{confidence:.2f}%")
    m4.metric("📐 Features", len(feature_names))

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Charts ----
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📊</div>
        <div class="section-header-text">
            <div class="section-header-title">Visual Analytics</div>
            <div class="section-header-sub">Probability distribution and feature radar</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    # ---------- Bar chart (fixed Plotly API) ----------
    with c1:
        labels = [t.replace("class_", "Class ") for t in target_names]
        values = [round(float(p) * 100, 2) for p in probabilities]
        palette = ["#a855f7", "#ec4899", "#f472b6"]
        colors = [
            palette[i % len(palette)] if i == prediction else "rgba(255,255,255,0.12)"
            for i in range(len(values))
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=14, family="Inter, sans-serif"),
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.25)", width=1),
                cornerradius=8,
            ),
            hovertemplate="<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text="Class Probabilities",
                x=0.5, xanchor="center",
                font=dict(size=15, color="#f9a8d4", family="Inter, sans-serif"),
            ),
            xaxis=dict(
                title=dict(text=""),
                tickfont=dict(color="rgba(255,255,255,0.9)", size=12),
                gridcolor="rgba(255,255,255,0.04)",
            ),
            yaxis=dict(
                title=dict(
                    text="Probability (%)",
                    font=dict(color="rgba(255,255,255,0.7)", size=12),
                ),
                tickfont=dict(color="rgba(255,255,255,0.9)", size=11),
                range=[0, 118],
                gridcolor="rgba(255,255,255,0.06)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="white", family="Inter, sans-serif"),
            height=430,
            showlegend=False,
            margin=dict(l=50, r=30, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---------- Radar chart ----------
    with c2:
        r_vals = list(ordered_values) + [ordered_values[0]]
        theta_vals = [pretty(f) for f in feature_names] + [pretty(feature_names[0])]
        max_val = max(ordered_values) if max(ordered_values) > 0 else 1

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=theta_vals, fill="toself", name="Input",
            line=dict(color="#ec4899", width=2.5),
            fillcolor="rgba(236,72,153,0.22)",
            hovertemplate="<b>%{theta}</b><br>Value: %{r:.2f}<extra></extra>",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.025)",
                radialaxis=dict(
                    visible=True,
                    range=[0, max_val * 1.2],
                    gridcolor="rgba(255,255,255,0.12)",
                    tickfont=dict(color="rgba(255,255,255,0.5)", size=9),
                    linecolor="rgba(255,255,255,0.1)",
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.12)",
                    tickfont=dict(color="rgba(255,255,255,0.85)", size=10),
                    linecolor="rgba(255,255,255,0.1)",
                ),
            ),
            title=dict(
                text="Feature Radar",
                x=0.5, xanchor="center",
                font=dict(size=15, color="#c4b5fd", family="Inter, sans-serif"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter, sans-serif"),
            height=430,
            showlegend=False,
            margin=dict(l=60, r=60, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---- Feature table ----
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📋</div>
        <div class="section-header-text">
            <div class="section-header-title">Input Summary</div>
            <div class="section-header-sub">Your values compared to the training distribution</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rows = []
    for i, f in enumerate(feature_names):
        mean = float(wine_data[:, i].mean())
        std = float(wine_data[:, i].std()) or 1.0
        z = (ordered_values[i] - mean) / std
        rows.append({
            "Feature": pretty(f),
            "Value": round(ordered_values[i], 4),
            "Training Mean": round(mean, 4),
            "Deviation (σ)": round(z, 3),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=500,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Value": st.column_config.NumberColumn("Value", format="%.4f"),
            "Training Mean": st.column_config.NumberColumn("Training Mean", format="%.4f"),
            "Deviation (σ)": st.column_config.NumberColumn("Deviation (σ)", format="%.3f"),
        },
    )

else:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 2rem;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👆</div>
        <div style="color: rgba(245,243,255,0.75); font-size: 1rem; font-weight: 500;">
            Fill in the chemical properties above and click
            <span style="color: #f9a8d4; font-weight: 700;">Predict Cultivar</span>
            to see the result.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# Help
# =========================================================

with st.expander("ℹ️  About & Help"):
    st.markdown("""
    **Wine Classifier Pro** uses a `RandomForestClassifier` trained on the
    classic *Wine* dataset from scikit-learn (178 samples, 13 chemical features).

    **Cultivars**
    - `Class 0` → Cultivar 1
    - `Class 1` → Cultivar 2
    - `Class 2` → Cultivar 3

    **How to use**
    1. Enter the 13 chemical property values in the form.
    2. Click **Predict Cultivar**.
    3. The model returns the predicted cultivar, class probabilities,
       a radar chart, and a feature summary table.

    **Buttons**
    - **Predict Cultivar** — runs the prediction.
    - **Load Mean** — fills every input with the training-set mean.
    - **Reset All** — clears the form to start over.

    **Deviation (σ)** in the table shows how far each input is from the
    training-set mean, in standard deviations. Values beyond ±2σ are unusual.
    """)


# =========================================================
# Footer
# =========================================================

st.markdown("""
<div class="footer">
    <span class="footer-badge">Streamlit</span>
    <span class="footer-badge">Plotly</span>
    <span class="footer-badge">scikit-learn</span>
    <div style="margin-top: 1rem;">Crafted with precision for wine enthusiasts & data scientists</div>
</div>
""", unsafe_allow_html=True)