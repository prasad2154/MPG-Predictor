# Install dependancies (Put this on terminal) - pip install streamlit numpy pandas plotly scikit-learn>=1.3.0
# to run this code use: streamlit run app.py
#--------------------------------------------------
# If you are facing issue w.r.t cmdlet - Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="MPG Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model.pkl"
FEATURES = ["cyl", "disp", "wt", "hp"]


# -----------------------------
# Custom styling
# -----------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
        }

        .block-container {
            max-width: 1350px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0284c7 100%);
            border-radius: 24px;
            padding: 2rem 2.2rem;
            color: white;
            box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.2rem;
        }

        .hero-card h1 {
            font-size: 2.35rem;
            margin-bottom: 0.35rem;
        }

        .hero-card p {
            color: #dbeafe;
            font-size: 1.02rem;
            margin: 0;
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 20px;
            padding: 1.25rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(10px);
        }

        .result-card {
            background: linear-gradient(135deg, #ecfeff 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 20px;
            padding: 1.2rem 1.4rem;
            text-align: center;
            margin-top: 0.8rem;
        }

        .result-value {
            font-size: 2.75rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1;
        }

        .result-label {
            color: #475569;
            font-size: 0.95rem;
            margin-top: 0.5rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            padding: 0.9rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            min-height: 48px;
            font-weight: 700;
            background: linear-gradient(90deg, #2563eb, #0284c7);
            color: white;
            border: none;
        }

        .stButton > button:hover {
            color: white;
            border: none;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
        }

        .small-note {
            color: #64748b;
            font-size: 0.86rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' was not found. Place model.pkl in the same folder as app.py."
        )

    with open(path, "rb") as file:
        return pickle.load(file)


def predict_mpg(model, input_df: pd.DataFrame) -> float:
    """
    Tries prediction with a DataFrame first, then falls back to a NumPy array.
    This supports most scikit-learn pickle models.
    """
    try:
        prediction = model.predict(input_df)
    except Exception:
        prediction = model.predict(input_df[FEATURES].to_numpy())

    value = float(np.ravel(prediction)[0])

    if not np.isfinite(value):
        raise ValueError("The model returned an invalid prediction.")

    return value


def mpg_category(mpg: float):
    if mpg < 15:
        return "Low efficiency", "🔴"
    if mpg < 25:
        return "Moderate efficiency", "🟠"
    if mpg < 35:
        return "Good efficiency", "🟢"
    return "Excellent efficiency", "🔵"


def create_gauge(mpg: float):
    upper_limit = max(50, int(np.ceil(mpg / 10) * 10))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mpg,
            number={"suffix": " MPG", "font": {"size": 36}},
            title={"text": "Predicted Fuel Efficiency"},
            gauge={
                "axis": {"range": [0, upper_limit]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 15], "color": "#fee2e2"},
                    {"range": [15, 25], "color": "#ffedd5"},
                    {"range": [25, 35], "color": "#dcfce7"},
                    {"range": [35, upper_limit], "color": "#dbeafe"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 4},
                    "thickness": 0.75,
                    "value": mpg,
                },
            },
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=25, r=25, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial"},
    )
    return fig


# -----------------------------
# Session state
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🚗 MPG Predictor")
    st.caption("Machine Learning Prediction Dashboard")

    st.divider()

    st.subheader("Model status")

    try:
        model = load_model(MODEL_PATH)
        st.success("model.pkl loaded successfully")
        model_ready = True
    except Exception as error:
        model = None
        model_ready = False
        st.error(str(error))

    st.divider()

    st.subheader("Input features")
    st.markdown(
        """
        **cyl** — Number of cylinders  
        **disp** — Engine displacement  
        **wt** — Vehicle weight  
        **hp** — Horsepower
        """
    )

    st.divider()

    st.info(
        "The feature order used for prediction is: "
        "`cyl, disp, wt, hp`."
    )

    if st.button("Clear prediction history"):
        st.session_state.history = []
        st.rerun()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1>Vehicle MPG Prediction System</h1>
        <p>
            Enter vehicle specifications and estimate mileage in miles per gallon
            using your trained machine-learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Presets
# -----------------------------
st.subheader("Quick vehicle presets")

preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

presets = {
    "Compact": {"cyl": 4, "disp": 120.1, "wt": 2.465, "hp": 97},
    "Balanced": {"cyl": 4, "disp": 140.8, "wt": 3.150, "hp": 95},
    "Performance": {"cyl": 8, "disp": 275.8, "wt": 3.730, "hp": 180},
    "Heavy Duty": {"cyl": 8, "disp": 351.0, "wt": 3.170, "hp": 264},
}

if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Balanced"

with preset_col1:
    if st.button("🚙 Compact", key="compact"):
        st.session_state.selected_preset = "Compact"
with preset_col2:
    if st.button("⚖️ Balanced", key="balanced"):
        st.session_state.selected_preset = "Balanced"
with preset_col3:
    if st.button("🏎️ Performance", key="performance"):
        st.session_state.selected_preset = "Performance"
with preset_col4:
    if st.button("🚛 Heavy Duty", key="heavy"):
        st.session_state.selected_preset = "Heavy Duty"

selected = presets[st.session_state.selected_preset]

st.caption(f"Current preset: **{st.session_state.selected_preset}**")


# -----------------------------
# Main input and output layout
# -----------------------------
left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Vehicle specifications")

    with st.form("prediction_form"):
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            cyl = st.selectbox(
                "Number of cylinders",
                options=[3, 4, 5, 6, 8, 10, 12],
                index=[3, 4, 5, 6, 8, 10, 12].index(
                    selected["cyl"]
                    if selected["cyl"] in [3, 4, 5, 6, 8, 10, 12]
                    else 4
                ),
                help="Total number of cylinders in the engine.",
            )

        with row1_col2:
            hp = st.number_input(
                "Horsepower (hp)",
                min_value=20.0,
                max_value=1000.0,
                value=float(selected["hp"]),
                step=1.0,
                help="Maximum engine horsepower.",
            )

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            disp = st.number_input(
                "Displacement (disp)",
                min_value=30.0,
                max_value=1000.0,
                value=float(selected["disp"]),
                step=1.0,
                help="Engine displacement, commonly measured in cubic inches.",
            )

        with row2_col2:
            wt = st.number_input(
                "Weight (wt)",
                min_value=0.5,
                max_value=15.0,
                value=float(selected["wt"]),
                step=0.01,
                format="%.3f",
                help="Vehicle weight. Use the same unit and scaling used during model training.",
            )

        st.markdown("#### Input summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Cylinders", int(cyl))
        m2.metric("Displacement", f"{disp:.1f}")
        m3.metric("Weight", f"{wt:.3f}")
        m4.metric("Horsepower", f"{hp:.0f}")

        submitted = st.form_submit_button(
            "Predict MPG",
            disabled=not model_ready,
            width="stretch",
        )

    st.markdown("</div>", unsafe_allow_html=True)


input_df = pd.DataFrame(
    [[cyl, disp, wt, hp]],
    columns=FEATURES,
)

prediction = None
prediction_error = None

if submitted:
    try:
        prediction = predict_mpg(model, input_df)
        category, icon = mpg_category(prediction)

        st.session_state.history.insert(
            0,
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Cylinders": int(cyl),
                "Displacement": round(float(disp), 2),
                "Weight": round(float(wt), 3),
                "Horsepower": round(float(hp), 1),
                "Predicted MPG": round(prediction, 2),
                "Category": category,
            },
        )

        st.session_state.history = st.session_state.history[:20]

    except Exception as error:
        prediction_error = str(error)


with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Prediction result")

    if prediction_error:
        st.error(f"Prediction failed: {prediction_error}")

    elif prediction is not None:
        category, icon = mpg_category(prediction)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-value">{prediction:.2f}</div>
                <div class="result-label">Miles per gallon</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            create_gauge(prediction),
            width='stretch',
            config={"displayModeBar": False},
        )

        st.success(f"{icon} {category}")

        estimated_l_per_100km = 235.214583 / prediction if prediction > 0 else 0
        r1, r2 = st.columns(2)
        r1.metric("Predicted MPG", f"{prediction:.2f}")
        r2.metric("Approx. L/100 km", f"{estimated_l_per_100km:.2f}")

    else:
        st.info("Enter vehicle details and click **Predict MPG**.")
        st.plotly_chart(
            create_gauge(0),
            width='stretch',
            config={"displayModeBar": False},
        )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Input data preview
# -----------------------------
st.markdown("---")
preview_col, insight_col = st.columns([1.1, 0.9], gap="large")

with preview_col:
    st.subheader("Model input preview")
    st.dataframe(
        input_df,
        width='stretch',
        hide_index=True,
    )

with insight_col:
    st.subheader("Prediction notes")
    st.markdown(
        """
        - Higher displacement, horsepower, weight, and cylinder count often reduce fuel efficiency.
        - Keep input units consistent with the data used to train `model.pkl`.
        - This application assumes the model expects exactly four features in this order:
          `cyl`, `disp`, `wt`, `hp`.
        """
    )


# -----------------------------
# Prediction history
# -----------------------------
st.markdown("---")
st.subheader("Recent prediction history")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        width='stretch',
        hide_index=True,
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download history as CSV",
        data=csv_data,
        file_name="mpg_prediction_history.csv",
        mime="text/csv",
        width='stretch',
    )
else:
    st.caption("No predictions have been made yet.")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#64748b; font-size:0.88rem;">
        MPG Prediction Dashboard • Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
