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
    page_title="Adventa - Campaign Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== FUNCTION DEFINITIONS (MOVE THESE HERE, BEFORE ANY CODE THAT USES THEM) ==========
def adstock(x, decay=0.5):
    """Calculate adstock transformation for carryover effect"""
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
        return None, "Not enough data to train model. Need at least 5 rows of data."
    
    try:
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
        st.session_state["test_indices"] = X_test.index.tolist()
        st.session_state["df_analysis"] = df_model
        st.session_state["y_actual_full"] = y
        st.session_state["y_predicted_full"] = model.predict(X)
        
        return model, None, r2, mae
        
    except Exception as e:
        return None, f"Error training model: {str(e)}", None, None

def predict_revenue_lasso(df, model, fb_spend, instagram_spend, tiktok_spend, category_value=None):
    if len(df) > 0:
        last_fb_adstock = adstock(df['fb_spend'].values)[-1] if 'fb_spend' in df.columns else 0
        last_insta_adstock = adstock(df['instagram_spend'].values)[-1] if 'instagram_spend' in df.columns else 0
        last_tiktok_adstock = adstock(df['tiktok_spend'].values)[-1] if 'tiktok_spend' in df.columns else 0
    else:
        last_fb_adstock = last_insta_adstock = last_tiktok_adstock = 0
    
    decay_rate = 0.5
    fb_adstock_pred = fb_spend + decay_rate * last_fb_adstock
    insta_adstock_pred = instagram_spend + decay_rate * last_insta_adstock
    tiktok_adstock_pred = tiktok_spend + decay_rate * last_tiktok_adstock
    
    prediction_row = {
        'fb_adstock': fb_adstock_pred,
        'insta_adstock': insta_adstock_pred,
        'tiktok_adstock': tiktok_adstock_pred,
        'fb_spend': fb_spend,
        'instagram_spend': instagram_spend,
        'tiktok_spend': tiktok_spend
    }
    
    if category_value and 'category' in df.columns:
        unique_cats = df['category'].unique()
        for cat in unique_cats:
            dummy_col = f'category_{cat}'
            if dummy_col in st.session_state.get("feature_cols", []):
                prediction_row[dummy_col] = 1 if cat == category_value else 0
    
    feature_cols = st.session_state.get("feature_cols", [])
    for col in feature_cols:
        if col not in prediction_row:
            prediction_row[col] = 0
    
    features_df = pd.DataFrame([prediction_row])[feature_cols]
    predicted_revenue = model.predict(features_df)[0]
    
    return predicted_revenue

# ========== PROFESSIONAL LIGHT THEME CSS ==========
st.markdown("""
<style>
    /* Your existing CSS here */
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
    <style>
    /* Your header CSS here */
    </style>
    
    <div class="header-container">
        <!-- Your header HTML here -->
    </div>
    <hr style="height: 2px; background: linear-gradient(90deg, #FF4B4B, #FF9068, #FFD166, transparent); border: none; margin-top: -0.8rem;">
""", unsafe_allow_html=True)

# ========== SIDEBAR NAVIGATION ==========
# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Custom sidebar navigation
with st.sidebar:
    st.markdown("### 🚀 ADVENTA")
    st.markdown("---")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.session_state["page"] = "home"
        st.rerun()
    
    if st.button("📊 Dashboard", use_container_width=True):
        if "cleaned_df" in st.session_state:
            st.session_state["page"] = "dashboard"
            st.rerun()
        else:
            st.warning("Please upload data first on the Home page")
    
    if st.button("🎯 Predict", use_container_width=True):
        if "trained_model" in st.session_state:
            st.session_state["page"] = "predict"
            st.rerun()
        else:
            st.warning("Please train the model first on the Home page")
    
    if st.button("📈 Analytics", use_container_width=True):
        if "trained_model" in st.session_state:
            st.session_state["page"] = "analytics"
            st.rerun()
        else:
            st.warning("Please train the model first on the Home page")
    
    st.markdown("---")
    st.caption("v1.0.0 | Analyzer")

# Page routing
if st.session_state["page"] == "home":
    # Welcome page content
    st.markdown("## 🎯 Welcome to ADVENTA")
    st.markdown("""
    ### Your Advertisement Campaign Spend Optimizer
    
    **Features:**
    - 🤖 **Machine Learning Predictions** using Lasso Regression
    - 📊 **Real-time Analytics** with interactive visualizations
    - 💰 **Budget Optimization** recommendations
    - 🔮 **Revenue Forecasting** based on ad spend
    
    ### How it works:
    1. Upload your campaign data (CSV format)
    2. Let our AI train on your historical performance
    3. Get predictions and optimization recommendations
    """)
    
    uploaded_file = st.file_uploader(
        "📁 Upload Campaign Data (CSV)",
        type=["csv"],
        help="Required columns: date, category, fb_spend, instagram_spend, tiktok_spend, total_revenue"
    )
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.session_state["raw_df"] = raw_df
        st.success("✅ File uploaded successfully! Click 'Process Data' to continue.")
        
        if st.button("🚀 Process Data & Train Model", type="primary"):
            with st.spinner("Training AI model..."):
                cleaned_df = clean_ad_data(raw_df)  # Now this works because function is defined above
                st.session_state["cleaned_df"] = cleaned_df
                model, error, r2, mae = train_prediction_model(cleaned_df)  # Now this works too
                if model:
                    st.session_state["trained_model"] = model
                    st.session_state["model_type"] = "lasso"
                    st.session_state["r2_score"] = r2
                    st.session_state["mae"] = mae
                    st.success("✅ Model trained successfully!")
                    st.balloons()
                    st.info("👉 Go to Dashboard or Predict tabs to see results")
                else:
                    st.error(f"Error: {error}")
    
    st.stop()

elif st.session_state["page"] == "dashboard":
    st.markdown("## 📊 Dashboard Overview")
    st.info("Dashboard coming soon! Please upload data and train model on the Home page first.")
    
elif st.session_state["page"] == "predict":
    st.markdown("## 🎯 Campaign Predictor")
    st.info("Predictor coming soon! Please upload data and train model on the Home page first.")
    
elif st.session_state["page"] == "analytics":
    st.markdown("## 📈 Advanced Analytics")
    st.info("Analytics coming soon! Please upload data and train model on the Home page first.")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #64748b;'>
    <p>🚀 Adventa - Advertisement Campaign Spend Optimizer</p>
    <p style='font-size: 12px;'>Powered by Lasso Regression & Adstock Transformation</p>
</div>
""", unsafe_allow_html=True)
