import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="AdVanta - Campaign Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CLEAN PROFESSIONAL CSS ==========
st.markdown("""
<style>

/* ===== MAIN BACKGROUND ===== */
.stApp {
    background-color: #F9FAFB;
}

/* ===== MAIN CONTAINER ===== */
.main > div {
    background: #FFFFFF;
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

/* ===== HEADINGS ===== */
h1 {
    color: #111827;
    font-weight: 700;
}
h2, h3 {
    color: #1F2937;
    font-weight: 600;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: #4F46E5;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #4338CA;
    transform: translateY(-1px);
}

/* ===== EXPANDERS ===== */
.streamlit-expanderHeader {
    background: #F3F4F6;
    border-radius: 10px;
    font-weight: 600;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    padding: 15px;
    border-radius: 12px;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab"] {
    background: #F3F4F6;
    border-radius: 8px;
    padding: 10px 16px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #4F46E5 !important;
    color: white !important;
}

/* ===== DATAFRAME ===== */
.dataframe {
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.title("🚀 AdVanta")
st.markdown("### AI-Powered Advertising Analytics & Optimization Platform")
st.markdown("---")

# ========== DATA IMPORT ==========
st.markdown("## 📂 Data Upload")

uploaded_file = st.file_uploader(
    "Upload Advertising Dataset (CSV)",
    type=["csv"]
)

# ================= FUNCTIONS (UNCHANGED) =================
def adstock(x, decay=0.5):
    result = []
    for i, val in enumerate(x):
        if i == 0:
            result.append(val)
        else:
            result.append(val + decay * result[i-1])
    return result

def clean_ad_data(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna("unknown")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    if "total_revenue" in df.columns:
        if (df["total_revenue"] == 0).all():
            df = df.drop(columns=["total_revenue"])

    return df

def train_prediction_model(df):
    required_cols = ["total_revenue", "fb_spend", "instagram_spend", "tiktok_spend"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        return None, f"Missing columns: {', '.join(missing)}"

    if len(df) < 5:
        return None, "Not enough data"

    df_model = df.copy()
    df_model['fb_adstock'] = adstock(df_model['fb_spend'].values)
    df_model['insta_adstock'] = adstock(df_model['instagram_spend'].values)
    df_model['tiktok_adstock'] = adstock(df_model['tiktok_spend'].values)

    if 'category' in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=['category'], drop_first=True)

    feature_cols = [col for col in df_model.columns if col not in ['total_revenue', 'date']]
    X = df_model[feature_cols]
    y = df_model['total_revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Lasso(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.session_state["r2_score"] = r2
    st.session_state["mae"] = mae
    st.session_state["feature_cols"] = feature_cols
    st.session_state["y_actual_full"] = y
    st.session_state["y_predicted_full"] = model.predict(X)

    return model, None, r2, mae

# ================= CLEAN + TRAIN =================
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    with st.expander("🧹 Clean & Train Model"):
        if st.button("🚀 Run Cleaning & Training", use_container_width=True):
            cleaned_df = clean_ad_data(raw_df)
            st.session_state["cleaned_df"] = cleaned_df

            model, error, r2, mae = train_prediction_model(cleaned_df)

            if model:
                st.session_state["trained_model"] = model
                st.success("✅ Model trained successfully")
                st.metric("R² Score", f"{r2:.3f}")
            else:
                st.error(error)

# ================= FOOTER =================
st.markdown("---")
st.markdown("© 2026 AdVanta • Marketing Intelligence Platform")
