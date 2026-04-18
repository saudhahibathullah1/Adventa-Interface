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

# ========== FUNCTION DEFINITIONS ==========
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
    /* Main background - Light gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
    }
    
    /* Card styling */
    .css-1r6slb0, .css-1v3fvcr {
        background-color: #ffffff;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        padding: 20px;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1e293b;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #64748b;
    }
    
    /* Headers */
    h1 {
        color: #0f172a;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.8rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #3b82f6;
        padding-left: 15px;
    }
    
    h3 {
        color: #1e293b;
        font-weight: 600;
        font-size: 1.3rem;
        margin-top: 0.75rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59,130,246,0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 16px;
        font-size: 1.1rem;
    }
    
    .streamlit-expanderContent {
        background: #ffffff;
        border-radius: 0 0 12px 12px;
        padding: 20px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        color: #475569;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
    }
    
    /* Alert messages */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    /* Success message */
    div[data-testid="stSuccess"] {
        background-color: #f0fdf4;
        border-left-color: #22c55e;
    }
    
    /* Info message */
    div[data-testid="stInfo"] {
        background-color: #eff6ff;
        border-left-color: #3b82f6;
    }
    
    /* Warning message */
    div[data-testid="stWarning"] {
        background-color: #fefce8;
        border-left-color: #eab308;
    }
    
    /* Error message */
    div[data-testid="stError"] {
        background-color: #fef2f2;
        border-left-color: #ef4444;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    
    .dataframe th {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 12px;
    }
    
    /* Input fields */
    .stNumberInput input, .stSelectbox select, .stDateInput input {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        padding: 8px 12px;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.1);
    }
    
    /* Caption text */
    .stCaption {
        color: #64748b;
        font-size: 0.875rem;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    }
</style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    .header-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    .rocket-icon {
        font-size: 3.5rem;
        filter: drop-shadow(0 0 10px rgba(255,75,75,0.5));
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .title-text {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #FF4B4B, #FF9068, #FFD166);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle-text {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #a8c0ff, #3f2b96);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .badge {
        background: rgba(255,75,75,0.2);
        backdrop-filter: blur(10px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-family: monospace;
        color: #FF9068;
        border: 1px solid rgba(255,75,75,0.3);
        margin-left: 15px;
    }
    </style>
    
    <div class="header-container">
        <div class="header-content">
            <div class="rocket-icon">🚀</div>
            <div style="flex: 1;">
                <div style="display: flex; align-items: baseline; gap: 15px;">
                    <div class="title-text">ADVENTA</div>
                    <div class="badge">ALPHA</div>
                </div>
                <div class="subtitle-text">📊 AD CAMPAIGN SPEND OPTIMIZING TOOL</div>
            </div>
        </div>
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
    st.caption("v1.0.0 | AI-Powered")

# ========== PAGE ROUTING ==========
if st.session_state["page"] == "home":
    # Welcome page content
    st.markdown("## 🎯 Welcome to ADVENTA")
    st.markdown("""
    ### Your AI-Powered Campaign Spend Optimizer
    
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
                cleaned_df = clean_ad_data(raw_df)
                st.session_state["cleaned_df"] = cleaned_df
                model, error, r2, mae = train_prediction_model(cleaned_df)
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
    
    if "cleaned_df" not in st.session_state:
        st.warning("⚠️ Please upload and process data on the Home page first.")
    else:
        df = st.session_state["cleaned_df"]
        
        # Key metrics
        total_revenue = df["total_revenue"].sum() if "total_revenue" in df.columns else 0
        total_ad_spend = df[["fb_spend","instagram_spend","tiktok_spend"]].sum().sum() if all(col in df.columns for col in ["fb_spend","instagram_spend","tiktok_spend"]) else 0
        total_campaigns = len(df)
        avg_roi = ((total_revenue - total_ad_spend) / total_ad_spend * 100) if total_ad_spend > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Ad Spend", f"${total_ad_spend:,.2f}")
        with col2:
            st.metric("📊 Total Revenue", f"${total_revenue:,.2f}")
        with col3:
            st.metric("📈 Total Campaigns", total_campaigns)
        with col4:
            st.metric("🎯 Average ROI", f"{avg_roi:.1f}%", delta="Positive" if avg_roi > 0 else "Negative")
        
        # Channel spend breakdown
        st.markdown("### 💰 Channel Spend Breakdown")
        channel_spend = {
            'Facebook': df['fb_spend'].sum() if 'fb_spend' in df.columns else 0,
            'Instagram': df['instagram_spend'].sum() if 'instagram_spend' in df.columns else 0,
            'TikTok': df['tiktok_spend'].sum() if 'tiktok_spend' in df.columns else 0
        }
        
        fig = px.pie(values=list(channel_spend.values()), names=list(channel_spend.keys()), 
                     title="Ad Spend Distribution by Channel", color_discrete_sequence=['#3b82f6', '#8b5cf6', '#ec4899'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif st.session_state["page"] == "predict":
    st.markdown("## 🎯 Campaign Predictor")
    
    if "trained_model" not in st.session_state:
        st.warning("⚠️ Please train the model first on the Home page.")
    else:
        df = st.session_state["cleaned_df"]
        
        st.markdown("### 📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{st.session_state.get('r2_score', 0):.3f}")
        with col2:
            st.metric("MAE", f"${st.session_state.get('mae', 0):,.0f}")
        with col3:
            confidence = "High" if st.session_state.get('r2_score', 0) > 0.8 else "Medium" if st.session_state.get('r2_score', 0) > 0.6 else "Low"
            st.metric("Confidence", confidence)
        
        st.markdown("### 🎯 Make New Predictions")
        st.markdown("Enter your proposed ad spend to forecast revenue")
        
        has_category = 'category' in df.columns
        
        col1, col2, col3 = st.columns(3)
        with col1:
            fb_spend = st.number_input("💰 Facebook Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="fb_input_pred")
        with col2:
            instagram_spend = st.number_input("📸 Instagram Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="insta_input_pred")
        with col3:
            tiktok_spend = st.number_input("🎵 TikTok Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="tiktok_input_pred")
        
        category_value = None
        if has_category:
            categories = df['category'].unique().tolist()
            category_value = st.selectbox("📁 Campaign Category", categories)
        
        if st.button("🔮 Predict Revenue", type="primary", use_container_width=True):
            model = st.session_state["trained_model"]
            
            with st.spinner("Calculating prediction..."):
                predicted_revenue = predict_revenue_lasso(
                    df, model, fb_spend, instagram_spend, tiktok_spend, category_value
                )
            
            total_ad_spend = fb_spend + instagram_spend + tiktok_spend
            roi = ((predicted_revenue - total_ad_spend) / total_ad_spend * 100) if total_ad_spend > 0 else 0
            
            st.markdown("---")
            st.markdown("### 📈 Prediction Results")
            
            col_result1, col_result2 = st.columns([1, 1])
            with col_result1:
                st.metric("💰 Total Ad Spend", f"${total_ad_spend:,.2f}")
                st.metric("📊 Predicted Revenue", f"${predicted_revenue:,.2f}", delta=f"${predicted_revenue - total_ad_spend:,.2f}")
            with col_result2:
                if roi < 0:
                    st.error(f"⚠️ Negative ROI: {roi:.1f}%")
                elif roi > 100:
                    st.success(f"🎉 Excellent ROI: {roi:.1f}%")
                elif roi > 50:
                    st.info(f"✅ Good ROI: {roi:.1f}%")
                else:
                    st.info(f"📈 ROI: {roi:.1f}%")

elif st.session_state["page"] == "analytics":
    st.markdown("## 📈 Advanced Analytics")
    
    if "trained_model" not in st.session_state:
        st.warning("⚠️ Please train the model first on the Home page.")
    else:
        df = st.session_state["cleaned_df"]
        model = st.session_state["trained_model"]
        
        # Create tabs for different analytics views
        tab1, tab2, tab3 = st.tabs(["💰 Revenue Analysis", "📈 Time Series", "🎯 Channel Impact"])
        
        with tab1:
            if "category" in df.columns:
                st.markdown("### Total Revenue by Campaign Category")
                category_revenue = df.groupby("category")["total_revenue"].sum().sort_values(ascending=True)
                
                fig = px.bar(x=category_revenue.values, y=category_revenue.index, 
                            orientation='h', color=category_revenue.values,
                            color_continuous_scale='Blues',
                            text=category_revenue.values)
                fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0),
                                 xaxis_title="Total Revenue ($)",
                                 yaxis_title="Campaign Category")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Category column not found in dataset.")
        
        with tab2:
            if "date" in df.columns:
                st.markdown("### Revenue vs Ad Spend Over Time")
                df['date'] = pd.to_datetime(df['date'])
                df_time = df.copy()
                df_time['total_spend'] = df_time[['fb_spend', 'instagram_spend', 'tiktok_spend']].sum(axis=1)
                df_time = df_time.sort_values('date')
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df_time['date'], y=df_time['total_revenue'],
                                        name='Revenue', line=dict(color='#3b82f6', width=3)),
                             secondary_y=False)
                fig.add_trace(go.Scatter(x=df_time['date'], y=df_time['total_spend'],
                                        name='Ad Spend', line=dict(color='#ef4444', width=3, dash='dash')),
                             secondary_y=True)
                
                fig.update_layout(title="Revenue vs Ad Spend Trend",
                                 xaxis_title="Date",
                                 height=450,
                                 hovermode='x unified')
                fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
                fig.update_yaxes(title_text="Ad Spend ($)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Date column not found for time series analysis.")
        
        with tab3:
            st.markdown("### Channel Impact on Revenue")
            if hasattr(model, 'coef_'):
                feature_cols = st.session_state.get("feature_cols", [])
                coef_df = pd.DataFrame({'Feature': feature_cols, 'Coefficient': model.coef_})
                channel_features = coef_df[coef_df['Feature'].str.contains('adstock|spend', case=False)]
                
                if not channel_features.empty:
                    alias_mapping = {
                        'fb_spend': 'Facebook Spend',
                        'instagram_spend': 'Instagram Spend',
                        'tiktok_spend': 'TikTok Spend',
                        'fb_adstock': 'Facebook AdStock',
                        'insta_adstock': 'Instagram AdStock',
                        'tiktok_adstock': 'TikTok AdStock'
                    }
                    channel_features['Feature'] = channel_features['Feature'].replace(alias_mapping)
                    channel_features = channel_features.sort_values('Coefficient', ascending=True)
                    
                    fig = px.bar(channel_features, x='Coefficient', y='Feature',
                                orientation='h', color='Coefficient',
                                color_continuous_scale='Blues',
                                title="Channel Impact on Revenue")
                    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0),
                                     xaxis_title="Coefficient Value",
                                     yaxis_title="Channel")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("💡 **What is Adstock?** Adstock measures the *carryover effect* of advertising - how past ad spend continues to influence revenue in future days.")
                else:
                    st.info("No channel-specific coefficients found")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #64748b;'>
    <p>🚀 Adventa - AI-Powered Campaign Spend Optimizer</p>
    <p style='font-size: 12px;'>Powered by Lasso Regression & Adstock Transformation</p>
</div>
""", unsafe_allow_html=True)
