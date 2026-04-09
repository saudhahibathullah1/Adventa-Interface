import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Adventa - Ad Spend Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import XGBoost with error handling
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Professional header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Professional button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
    }
    
    /* Metric styling */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

def adstock(x, decay=0.5):
    """Calculate adstock transformation"""
    result = []
    for i, val in enumerate(x):
        if i == 0:
            result.append(val)
        else:
            result.append(val + decay * result[i-1])
    return result

def clean_ad_data(df):
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna("unknown")
    
    # Convert date column if exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
    
    # Drop total_revenue if all zeros
    if "total_revenue" in df.columns:
        if (df["total_revenue"] == 0).all():
            df = df.drop(columns=["total_revenue"])
    
    return df

def train_prediction_model(df):
    """Train XGBoost model with adstock transformation"""
    if not XGBOOST_AVAILABLE:
        return None, "XGBoost not available. Install with: pip install xgboost scikit-learn"
    
    required_cols = ["total_revenue", "fb_spend", "instagram_spend", "tiktok_spend"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    
    if len(df) < 3:
        return None, "Need at least 3 rows of data"
    
    try:
        df_model = df.copy()
        df_model['fb_adstock'] = adstock(df_model['fb_spend'].values)
        df_model['insta_adstock'] = adstock(df_model['instagram_spend'].values)
        df_model['tiktok_adstock'] = adstock(df_model['tiktok_spend'].values)
        
        feature_cols = ['fb_adstock', 'insta_adstock', 'tiktok_adstock']
        X = df_model[feature_cols]
        y = df_model['total_revenue']
        
        model = XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.2,
            subsample=0.5,
            colsample_bytree=0.6,
            random_state=30
        )
        
        model.fit(X, y)
        return model, None
    except Exception as e:
        return None, f"Error: {str(e)}"

def simple_predict(df, fb_spend, instagram_spend, tiktok_spend):
    total_ad_spend_hist = (df['fb_spend'] + df['instagram_spend'] + df['tiktok_spend']).sum()
    total_revenue_hist = df['total_revenue'].sum()
    
    if total_ad_spend_hist > 0:
        roi_ratio = total_revenue_hist / total_ad_spend_hist
    else:
        roi_ratio = 1
    
    new_total_spend = fb_spend + instagram_spend + tiktok_spend
    predicted_revenue = new_total_spend * roi_ratio
    
    return predicted_revenue, roi_ratio

# Professional Header
st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; color:white;">🎈 Adventa</h1>
        <p style="margin:0; opacity:0.9; font-size:1.1rem;">AI-Powered Advertising Spend Optimizer</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for file upload and data info
with st.sidebar:
    st.markdown("### 📁 Data Management")
    uploaded_file = st.file_uploader(
        "Upload Advertising Dataset (CSV)",
        type=["csv"],
        help="Upload a CSV file with columns: date, total_revenue, fb_spend, instagram_spend, tiktok_spend"
    )
    
    st.markdown("---")
    st.markdown("### 📊 About Adventa")
    st.info(
        "Adventa uses advanced machine learning to optimize your ad spend across "
        "Facebook, Instagram, and TikTok. Get accurate revenue predictions and ROI analysis."
    )
    
    st.markdown("### 🎯 Features")
    st.markdown("""
    - ✅ Data cleaning & preprocessing
    - 🎯 Revenue prediction
    - 📈 ROI analysis
    - 📊 Historical trend analysis
    - 🤖 XGBoost ML model
    """)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    
    # Data Overview Section
    with st.expander("📋 Data Overview", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(raw_df))
        with col2:
            st.metric("Total Columns", len(raw_df.columns))
        with col3:
            st.metric("Memory Usage", f"{raw_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.markdown("#### Raw Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)
    
    # Data Cleaning Section
    with st.expander("🧹 Data Cleaning & Preparation", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("✨ Run Data Cleaning & Train Model", use_container_width=True):
                with st.spinner("Cleaning data and training model..."):
                    cleaned_df = clean_ad_data(raw_df)
                    st.session_state["cleaned_df"] = cleaned_df
                    
                    if XGBOOST_AVAILABLE:
                        model, error = train_prediction_model(cleaned_df)
                        if model:
                            st.session_state["trained_model"] = model
                            st.session_state["model_type"] = "xgboost"
                            st.success("✅ Dataset cleaned and XGBoost model trained successfully!")
                        else:
                            st.error(f"❌ Model training failed: {error}")
                            st.session_state["model_type"] = "simple"
                            st.info("ℹ️ Using simplified prediction model as fallback.")
                    else:
                        st.session_state["model_type"] = "simple"
                        st.success("✅ Dataset cleaned successfully!")
                        st.info("ℹ️ XGBoost not available. Using simplified prediction model.")
        with col2:
            if "cleaned_df" in st.session_state:
                csv = st.session_state["cleaned_df"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="💾 Download Cleaned Data",
                    data=csv,
                    file_name="adventa_cleaned_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        if "cleaned_df" in st.session_state:
            st.markdown("#### Cleaned Data Preview")
            st.dataframe(st.session_state["cleaned_df"].head(15), use_container_width=True)
    
    # Main content area with tabs for better organization
    if "cleaned_df" in st.session_state:
        tab1, tab2 = st.tabs(["🎯 Revenue Predictor", "📊 Historical Analysis"])
        
        # Tab 1: Revenue Predictor
        with tab1:
            st.markdown("### 🎯 Revenue Prediction")
            st.markdown("Enter your proposed ad spend to get AI-powered revenue predictions")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fb_spend = st.number_input(
                    "Facebook Ad Spend ($)", 
                    min_value=0.0, 
                    value=1000.0, 
                    step=100.0,
                    help="Amount to spend on Facebook ads"
                )
            with col2:
                instagram_spend = st.number_input(
                    "Instagram Ad Spend ($)", 
                    min_value=0.0, 
                    value=1000.0, 
                    step=100.0,
                    help="Amount to spend on Instagram ads"
                )
            with col3:
                tiktok_spend = st.number_input(
                    "TikTok Ad Spend ($)", 
                    min_value=0.0, 
                    value=1000.0, 
                    step=100.0,
                    help="Amount to spend on TikTok ads"
                )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔮 Predict Revenue", type="primary", use_container_width=True):
                    df = st.session_state["cleaned_df"]
                    model_type = st.session_state.get("model_type", "simple")
                    
                    with st.spinner("Calculating prediction..."):
                        if model_type == "xgboost" and "trained_model" in st.session_state:
                            model = st.session_state["trained_model"]
                            
                            if len(df) > 0:
                                last_fb_adstock = adstock(df['fb_spend'].values)[-1] if 'fb_spend' in df.columns else 0
                                last_insta_adstock = adstock(df['instagram_spend'].values)[-1] if 'instagram_spend' in df.columns else 0
                                last_tiktok_adstock = adstock(df['tiktok_spend'].values)[-1] if 'tiktok_spend' in df.columns else 0
                                
                                decay_rate = 0.5
                                fb_adstock_pred = fb_spend + decay_rate * last_fb_adstock
                                insta_adstock_pred = instagram_spend + decay_rate * last_insta_adstock
                                tiktok_adstock_pred = tiktok_spend + decay_rate * last_tiktok_adstock
                            else:
                                fb_adstock_pred, insta_adstock_pred, tiktok_adstock_pred = fb_spend, instagram_spend, tiktok_spend
                            
                            features = pd.DataFrame([[fb_adstock_pred, insta_adstock_pred, tiktok_adstock_pred]], 
                                                   columns=['fb_adstock', 'insta_adstock', 'tiktok_adstock'])
                            predicted_revenue = model.predict(features)[0]
                            prediction_method = "XGBoost AI Model"
                            show_details = True
                        else:
                            predicted_revenue, roi_ratio = simple_predict(df, fb_spend, instagram_spend, tiktok_spend)
                            prediction_method = "Simplified Model (Historical ROI)"
                            show_details = False
                    
                    # Results Dashboard
                    st.markdown("---")
                    st.markdown("#### 📈 Prediction Results")
                    
                    total_ad_spend = fb_spend + instagram_spend + tiktok_spend
                    roi = ((predicted_revenue - total_ad_spend) / total_ad_spend * 100) if total_ad_spend > 0 else 0
                    profit = predicted_revenue - total_ad_spend
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Investment", f"${total_ad_spend:,.2f}")
                    with col2:
                        st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}", 
                                 delta=f"${profit:+,.2f}")
                    with col3:
                        st.metric("ROI", f"{roi:+.1f}%")
                    with col4:
                        st.metric("Profit Margin", f"{(profit/predicted_revenue*100):+.1f}%" if predicted_revenue > 0 else "N/A")
                    
                    # ROI Gauge Chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = roi,
                        title = {'text': "Return on Investment (ROI)"},
                        delta = {'reference': 0},
                        gauge = {
                            'axis': {'range': [None, 200]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"},
                                {'range': [100, 200], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': roi
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    if roi < 0:
                        st.error("⚠️ **Negative ROI Predicted** - Consider reducing ad spend or reallocating budget to better-performing platforms.")
                    elif roi < 50:
                        st.warning("📊 **Moderate ROI Predicted** - Good but room for improvement. Test different creative strategies.")
                    elif roi < 100:
                        st.success("🎯 **Strong ROI Predicted** - Great performance! Consider scaling successful campaigns.")
                    else:
                        st.balloons()
                        st.success("🏆 **Excellent ROI Predicted!** - Your ad strategy is performing exceptionally well!")
                    
                    st.caption(f"*Prediction generated using {prediction_method}*")
                    
                    if show_details:
                        with st.expander("🔍 Advanced Analytics Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Adstock Values**")
                                st.markdown(f"- Facebook: ${fb_adstock_pred:,.2f}")
                                st.markdown(f"- Instagram: ${insta_adstock_pred:,.2f}")
                                st.markdown(f"- TikTok: ${tiktok_adstock_pred:,.2f}")
                                st.caption("*Adstock accounts for advertising carryover effects*")
                            with col2:
                                if hasattr(model, 'feature_importances_'):
                                    st.markdown("**Platform Impact**")
                                    importance_df = pd.DataFrame({
                                        'Platform': ['Facebook', 'Instagram', 'TikTok'],
                                        'Importance': model.feature_importances_
                                    })
                                    fig = px.bar(importance_df, x='Platform', y='Importance', 
                                                title="Feature Importance")
                                    fig.update_layout(height=250, showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Historical Analysis
        with tab2:
            st.markdown("### 📊 Historical Performance Analysis")
            
            if st.button("🔄 Load Analysis", use_container_width=True):
                df = st.session_state["cleaned_df"]
                
                required_cols = ["total_revenue", "fb_spend", "instagram_spend", "tiktok_spend"]
                missing = [c for c in required_cols if c not in df.columns]
                
                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    # Key Metrics
                    st.markdown("#### Key Performance Indicators")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_revenue = df["total_revenue"].sum()
                    total_ad_spend = df["fb_spend"].sum() + df["instagram_spend"].sum() + df["tiktok_spend"].sum()
                    ad_spend_pct = (total_ad_spend / total_revenue * 100) if total_revenue > 0 else 0
                    total_profit = total_revenue - total_ad_spend
                    
                    with col1:
                        st.metric("Total Revenue", f"${total_revenue:,.2f}")
                    with col2:
                        st.metric("Total Ad Spend", f"${total_ad_spend:,.2f}")
                    with col3:
                        st.metric("Total Profit", f"${total_profit:,.2f}", 
                                 delta=f"{(total_profit/total_revenue*100):+.1f}%" if total_revenue > 0 else "N/A")
                    with col4:
                        st.metric("ROI", f"{(total_profit/total_ad_spend*100):+.1f}%" if total_ad_spend > 0 else "N/A")
                    
                    # Spend vs Revenue Chart
                    st.markdown("#### Ad Spend vs Revenue Analysis")
                    daily_metrics = df.groupby("date").agg({
                        "total_revenue": "sum",
                        "fb_spend": "sum",
                        "instagram_spend": "sum",
                        "tiktok_spend": "sum"
                    }).reset_index()
                    
                    daily_metrics["total_ad_spend"] = daily_metrics["fb_spend"] + daily_metrics["instagram_spend"] + daily_metrics["tiktok_spend"]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_metrics["date"], y=daily_metrics["total_revenue"],
                                            mode='lines+markers', name='Revenue',
                                            line=dict(color='green', width=3)))
                    fig.add_trace(go.Bar(x=daily_metrics["date"], y=daily_metrics["total_ad_spend"],
                                        name='Ad Spend', marker_color='orange'))
                    fig.update_layout(title="Revenue vs Ad Spend Over Time",
                                     xaxis_title="Date",
                                     yaxis_title="Amount ($)",
                                     hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Platform Breakdown
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Ad Spend by Platform")
                        platform_spend = {
                            "Platform": ["Facebook", "Instagram", "TikTok"],
                            "Spend": [df["fb_spend"].sum(), df["instagram_spend"].sum(), df["tiktok_spend"].sum()]
                        }
                        spend_df = pd.DataFrame(platform_spend)
                        fig = px.pie(spend_df, values='Spend', names='Platform', title="Ad Spend Distribution")
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Efficiency Metrics")
                        efficiency_data = {
                            "Platform": ["Facebook", "Instagram", "TikTok"],
                            "Spend per Revenue ($)": [
                                df["fb_spend"].sum() / total_revenue if total_revenue > 0 else 0,
                                df["instagram_spend"].sum() / total_revenue if total_revenue > 0 else 0,
                                df["tiktok_spend"].sum() / total_revenue if total_revenue > 0 else 0
                            ]
                        }
                        efficiency_df = pd.DataFrame(efficiency_data)
                        fig = px.bar(efficiency_df, x='Platform', y='Spend per Revenue ($)',
                                    title="Cost Efficiency (Lower is Better)")
                        fig.update_layout(yaxis_title="Spend per $1 Revenue")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical Summary
                    with st.expander("📈 Statistical Summary"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Revenue Statistics**")
                            st.dataframe(df["total_revenue"].describe().to_frame(), use_container_width=True)
                        with col2:
                            st.markdown("**Ad Spend Statistics**")
                            spend_stats = pd.DataFrame({
                                'Facebook': df["fb_spend"].describe(),
                                'Instagram': df["instagram_spend"].describe(),
                                'TikTok': df["tiktok_spend"].describe()
                            })
                            st.dataframe(spend_stats, use_container_width=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>© 2024 Adventa - AI-Powered Advertising Optimization Platform</p>
        <p style="font-size:0.8rem;">Powered by Machine Learning | Real-time Predictions | Data-Driven Insights</p>
    </div>
""", unsafe_allow_html=True)
