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
# Styles
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 50%, #831843 100%);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    .hero-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f9a8d4, #c4b5fd, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        text-align: center;
        color: rgba(255,255,255,0.75);
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .glass-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        margin-bottom: 1.25rem;
    }
    .section-header {
        color: #f9a8d4;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0 0 1rem 0;
        padding-left: 0.75rem;
        border-left: 4px solid #a855f7;
    }
    .result-card {
        background: linear-gradient(135deg, rgba(168,85,247,0.25), rgba(236,72,153,0.15));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 24px;
        padding: 2.25rem 1.5rem;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
        margin-bottom: 1.5rem;
    }
    .result-label {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin-bottom: 0.5rem;
    }
    .result-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f9a8d4, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
    }
    .result-meta {
        color: rgba(255,255,255,0.85);
        font-size: 1.05rem;
        margin-top: 0.75rem;
    }
    /* Inputs */
    .stNumberInput input {
        background: rgba(255,255,255,0.06) !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    .stNumberInput label {
        color: #e9d5ff !important;
        font-weight: 500 !important;
    }
    /* Buttons */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.8rem 1rem !important;
        border: none !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #a855f7, #ec4899) !important;
        color: white !important;
        box-shadow: 0 10px 28px rgba(168,85,247,0.45) !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 36px rgba(168,85,247,0.65) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255,255,255,0.08) !important;
        color: #e9d5ff !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
    }
    /* Metrics */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1rem 1.25rem;
    }
    div[data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.65) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f9a8d4 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        color: #e9d5ff !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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

def pretty(name):
    return PRETTY_NAMES.get(name, name.replace("_", " ").title())


# =========================================================
# Hero
# =========================================================

st.markdown('<h1 class="hero-title">🍷 Wine Classifier Pro</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Enter the 13 chemical properties below and let the model identify the wine cultivar</p>',
    unsafe_allow_html=True,
)


# =========================================================
# Session state for reset
# =========================================================

if "form_key" not in st.session_state:
    st.session_state.form_key = 0


# =========================================================
# Input form
# =========================================================

st.markdown('<div class="section-header">🧪 Chemical Properties</div>', unsafe_allow_html=True)

with st.form(key=f"wine_form_{st.session_state.form_key}"):
    input_values = {}
    cols_per_row = 3
    rows = (len(feature_names) + cols_per_row - 1) // cols_per_row

    for r in range(rows):
        cols = st.columns(cols_per_row)
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
                    label=pretty(fname),
                    min_value=float(round(lo, 3)),
                    max_value=float(round(hi, 3)),
                    value=float(round(mean, 3)),
                    step=float(step),
                    format="%.4f",
                    help=f"Training mean: {mean:.2f} | std: {std:.2f}",
                )

    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        predict_clicked = st.form_submit_button(
            "🔮 Predict Cultivar",
            type="primary",
            use_container_width=True,
        )
    with b2:
        reset_clicked = st.form_submit_button(
            "♻️ Reset All",
            use_container_width=True,
        )
    with b3:
        load_mean_clicked = st.form_submit_button(
            "📊 Load Mean",
            use_container_width=True,
        )


if reset_clicked:
    st.session_state.form_key += 1
    st.rerun()

if load_mean_clicked:
    st.session_state.form_key += 1
    st.rerun()


# =========================================================
# Prediction
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
            Confidence: <b>{confidence:.2f}%</b> &nbsp;•&nbsp;
            {len(target_names)} possible classes
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- KPI metrics ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cultivar", predicted_label)
    m2.metric("Class Index", prediction)
    m3.metric("Confidence", f"{confidence:.2f}%")
    m4.metric("Features", len(feature_names))

    # ---- Charts ----
    st.markdown('<div class="section-header">📊 Visual Analytics</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        labels = [t.replace("class_", "Class ") for t in target_names]
        values = [round(float(p) * 100, 2) for p in probabilities]
        palette = ["#a855f7", "#ec4899", "#f472b6"]
        colors = [
            palette[i % len(palette)] if i == prediction else "rgba(255,255,255,0.15)"
            for i in range(len(values))
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=13),
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.3)", width=1)),
            hovertemplate="<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text="Class Probabilities", x=0.5, xanchor="center",
                       font=dict(size=16, color="#f9a8d4")),
            xaxis=dict(title="Cultivar", tickfont=dict(color="rgba(255,255,255,0.9)"),
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Probability (%)",
                       tickfont=dict(color="rgba(255,255,255,0.9)"),
                       range=[0, 115], gridcolor="rgba(255,255,255,0.08)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=420, showlegend=False,
            margin=dict(l=50, r=30, t=60, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        r_vals = list(ordered_values) + [ordered_values[0]]
        theta_vals = [pretty(f) for f in feature_names] + [pretty(feature_names[0])]
        max_val = max(ordered_values) if max(ordered_values) > 0 else 1

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=theta_vals, fill="toself", name="Input",
            line=dict(color="#ec4899", width=2),
            fillcolor="rgba(236,72,153,0.25)",
        ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(255,255,255,0.03)",
                radialaxis=dict(visible=True, range=[0, max_val * 1.2],
                                gridcolor="rgba(255,255,255,0.15)",
                                tickfont=dict(color="rgba(255,255,255,0.6)", size=10)),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.15)",
                                 tickfont=dict(color="rgba(255,255,255,0.85)", size=11)),
            ),
            title=dict(text="Feature Radar", x=0.5, xanchor="center",
                       font=dict(size=16, color="#c4b5fd")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), height=420, showlegend=False,
            margin=dict(l=60, r=60, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Feature table ----
    st.markdown('<div class="section-header">📋 Input Summary</div>', unsafe_allow_html=True)

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
    st.dataframe(df, use_container_width=True, hide_index=True, height=500)

else:
    st.info("👆 Fill in the chemical properties above and click **Predict Cultivar** to see the result.")


# =========================================================
# Help
# =========================================================

with st.expander("ℹ️ About & Help"):
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


st.markdown(
    '<p style="text-align:center;color:rgba(255,255,255,0.35);margin-top:2rem;font-size:0.85rem;">'
    'Built with Streamlit • Plotly • scikit-learn'
    '</p>',
    unsafe_allow_html=True,
)