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
    
    h2, h3 {
        color: #1e293b;
        font-weight: 600;
        margin-top: 1rem;
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
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown("# 🚀")
with col2:
    st.markdown("# Adventa")
    st.markdown("### AI-Powered Campaign Spend Optimizer")

st.markdown("---")

# ========== DATA IMPORT ==========
st.markdown("## 📁 Data Import")
st.markdown("Upload your campaign data to get started")

uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=["csv"],
    help="Upload CSV with columns: date, category, fb_spend, instagram_spend, tiktok_spend, total_revenue"
)

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

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    with st.expander("📄 Raw Data Preview", expanded=False):
        st.dataframe(raw_df.head(), use_container_width=True)

    # ---------- CLEAN DATA ----------
    with st.expander("🧹 Data Processing & Model Training", expanded=False):
        clean_button = st.button("🚀 Process Data & Train Model", use_container_width=True)
        
        if clean_button or "cleaned_df" in st.session_state:
            if clean_button:
                with st.spinner("Processing data and training AI model..."):
                    cleaned_df = clean_ad_data(raw_df)
                    st.session_state["cleaned_df"] = cleaned_df
                    model, error, r2, mae = train_prediction_model(cleaned_df)
                    
                    if model:
                        st.session_state["trained_model"] = model
                        st.session_state["model_type"] = "lasso"
                        st.success("✅ Dataset processed and AI model trained successfully!")
                        
                        if r2 >= 0.9:
                            st.balloons()
                            st.success("🎯 Excellent model! R² > 0.9 - Very strong predictive power")
                        elif r2 >= 0.7:
                            st.info("👍 Good model - Ready for predictions")
                        else:
                            st.warning("⚠️ Model could be improved - Consider adding more features or data")
                    else:
                        st.error(f"Model training failed: {error}")
                        st.session_state["model_type"] = "none"
            
            if "cleaned_df" in st.session_state:
                cleaned_df = st.session_state["cleaned_df"]
                
                col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
                with col_metric1:
                    st.metric("Total Rows", len(cleaned_df))
                with col_metric2:
                    st.metric("Total Columns", len(cleaned_df.columns))
                with col_metric3:
                    if "total_revenue" in cleaned_df.columns:
                        st.metric("Total Revenue", f"${cleaned_df['total_revenue'].sum():,.0f}")
                with col_metric4:
                    if "trained_model" in st.session_state:
                        st.metric("Model R²", f"{st.session_state.get('r2_score', 0):.3f}")
                
                st.subheader("Cleaned Data Preview")
                st.dataframe(cleaned_df.head(10), use_container_width=True)
                
                if 'category' in cleaned_df.columns:
                    st.subheader("Category Distribution")
                    category_counts = cleaned_df['category'].value_counts()
                    # Changed to Blues color scale to match theme
                    fig = px.bar(x=category_counts.values, y=category_counts.index, 
                                 orientation='h', color=category_counts.values,
                                 color_continuous_scale='Blues')
                    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                csv = cleaned_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Processed Dataset",
                    data=csv,
                    file_name="adventa_processed_data.csv",
                    mime="text/csv"
                )

# ---------- PREDICT SECTION ----------
    with st.expander("🎯 Predict Campaign Performance", expanded=False):
        if "cleaned_df" not in st.session_state:
            st.warning("⚠️ Please process your data and train the model first.")
        elif st.session_state.get("model_type") != "lasso":
            st.warning("⚠️ Model not trained successfully. Please re-upload and process data.")
        else:
            st.markdown("### 📊 Model Performance Metrics")
            
            col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
            with col_acc1:
                st.metric("🎯 R² Score", f"{st.session_state.get('r2_score', 0):.3f}")
                
            with col_acc2:
                st.metric("📉 MAE", f"${st.session_state.get('mae', 0):,.0f}")
                
            with col_acc3:
                y_actual_full = st.session_state.get("y_actual_full")
                y_predicted_full = st.session_state.get("y_predicted_full")
                if y_actual_full is not None and len(y_actual_full) > 0:
                    non_zero_mask = y_actual_full != 0
                    if np.any(non_zero_mask):
                        mape = np.mean(np.abs((y_actual_full[non_zero_mask] - y_predicted_full[non_zero_mask]) / y_actual_full[non_zero_mask])) * 100
                        st.metric("📊 Accuracy", f"{100 - mape:.1f}%")
                    else:
                        st.metric("📊 Status", "Ready")
                else:
                    st.metric("📊 Status", "Ready")
            with col_acc4:
                confidence_level = "High" if st.session_state.get('r2_score', 0) > 0.8 else "Medium" if st.session_state.get('r2_score', 0) > 0.6 else "Low"
                st.metric("💪 Confidence", confidence_level)
            
            st.markdown("### 📈 Actual vs Predicted Revenue")
            
            df = st.session_state["cleaned_df"]
            
            if "y_predicted_full" not in st.session_state:
                st.info("Recalculating predictions for chart...")
                df_temp = df.copy()
                df_temp['fb_adstock'] = adstock(df_temp['fb_spend'].values)
                df_temp['insta_adstock'] = adstock(df_temp['instagram_spend'].values)
                df_temp['tiktok_adstock'] = adstock(df_temp['tiktok_spend'].values)
                
                if 'category' in df_temp.columns:
                    df_temp = pd.get_dummies(df_temp, columns=['category'], drop_first=True)
                
                feature_cols = [col for col in df_temp.columns if col not in ['total_revenue', 'date']]
                X_full = df_temp[feature_cols]
                y_actual_full = df['total_revenue'].values
                model = st.session_state["trained_model"]
                y_predicted_full = model.predict(X_full)
                
                st.session_state["y_predicted_full"] = y_predicted_full
                st.session_state["y_actual_full"] = y_actual_full
                st.session_state["df_analysis"] = df_temp
            else:
                y_predicted_full = st.session_state["y_predicted_full"]
                y_actual_full = st.session_state["y_actual_full"]
            
            if "date" in df.columns and len(df) > 0:
                pred_df = pd.DataFrame({
                    'date': pd.to_datetime(df['date']),
                    'Actual Revenue': y_actual_full,
                    'Predicted Revenue': y_predicted_full
                })
                pred_df = pred_df.sort_values('date')
                
                max_date = pred_df['date'].max()
                min_date = pred_df['date'].min()
                
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="start_date_pred")
                with col_date2:
                    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="end_date_pred")
                
                if start_date > end_date:
                    st.error("Start date must be before end date")
                    start_date, end_date = end_date, start_date
                
                filtered_df = pred_df[(pred_df['date'] >= pd.to_datetime(start_date)) & (pred_df['date'] <= pd.to_datetime(end_date))]
                
                # Removed markers/dots from the chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['Actual Revenue'],
                                        mode='lines', name='Actual Revenue',
                                        line=dict(color='#3b82f6', width=2)))
                fig.add_trace(go.Scatter(x=filtered_df['date'], y=filtered_df['Predicted Revenue'],
                                        mode='lines', name='Predicted Revenue',
                                        line=dict(color='#8b5cf6', width=2, dash='dash')))
                
                fig.update_layout(
                    title="Model Predictions vs Actual Performance",
                    xaxis_title="Date",
                    yaxis_title="Revenue ($)",
                    hovermode='x unified',
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                if len(filtered_df) >= 2:
                    r2_range = r2_score(filtered_df['Actual Revenue'], filtered_df['Predicted Revenue'])
                    mae_range = mean_absolute_error(filtered_df['Actual Revenue'], filtered_df['Predicted Revenue'])
                    
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("R² (Selected Range)", f"{r2_range:.3f}")
                        st.caption("Proportion of variance explained (0-1, higher is better)")
                    with col_metric2:
                        st.metric("MAE (Selected Range)", f"${mae_range:,.0f}")
                        st.caption("Average prediction error in dollars (lower is better)")
            else:
                st.warning("Date column not found or empty in dataset.")
            
            st.markdown("---")
            st.markdown("### 🎯 Make New Predictions")
            st.markdown("Enter your proposed ad spend to forecast revenue")
            
            has_category = 'category' in df.columns
            
            col1, col2, col3 = st.columns(3)
            with col1:
                fb_spend = st.number_input("💰 Facebook Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="fb_input")
            with col2:
                instagram_spend = st.number_input("📸 Instagram Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="insta_input")
            with col3:
                tiktok_spend = st.number_input("🎵 TikTok Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="tiktok_input")
            
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
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=roi,
                    title={'text': "ROI (%)", 'font': {'size': 24}},
                    delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 200]},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [0, 50], 'color': "#dbeafe"},
                            {'range': [50, 100], 'color': "#bfdbfe"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=0, r=0, t=40, b=0))
                
                col_result1, col_result2 = st.columns([1, 1])
                with col_result1:
                    st.metric("💰 Total Ad Spend", f"${total_ad_spend:,.2f}")
                    st.metric("📊 Predicted Revenue", f"${predicted_revenue:,.2f}", delta=f"${predicted_revenue - total_ad_spend:,.2f}")
                with col_result2:
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                if roi < 0:
                    st.error("⚠️ Negative ROI predicted. Consider adjusting your ad spend allocation.")
                elif roi > 100:
                    st.success("🎉 Excellent ROI predicted! This campaign looks very promising.")
                elif roi > 50:
                    st.info("✅ Good ROI predicted! Solid campaign performance expected.")
                elif roi > 20:
                    st.info("📈 Positive ROI predicted. Good potential for this campaign.")
                else:
                    st.warning("📊 Moderate ROI predicted. Consider optimizing your budget allocation.")
                
                with st.expander("🔍 View Detailed Calculation"):
                    st.markdown("**Adstock Values (Carryover Effect):**")
                    col_detail1, col_detail2, col_detail3 = st.columns(3)
                    with col_detail1:
                        st.metric("Facebook Adstock", f"${fb_spend + 0.5 * adstock(df['fb_spend'].values)[-1] if len(df) > 0 else fb_spend:,.2f}")
                    with col_detail2:
                        st.metric("Instagram Adstock", f"${instagram_spend + 0.5 * adstock(df['instagram_spend'].values)[-1] if len(df) > 0 else instagram_spend:,.2f}")
                    with col_detail3:
                        st.metric("TikTok Adstock", f"${tiktok_spend + 0.5 * adstock(df['tiktok_spend'].values)[-1] if len(df) > 0 else tiktok_spend:,.2f}")
                    
                    if has_category:
                        st.markdown(f"**Selected Category:** {category_value}")
                    
                    # --- NEW SECTION: Investment Allocation Recommendation ---
                    st.markdown("---")
                    st.markdown("### 📊 Optimal Investment Allocation")
                    
                    # Extract coefficients from the trained model
                    model_coef = st.session_state["trained_model"].coef_
                    feature_names = st.session_state["feature_cols"]
                    
                    # Create a mapping for channel-specific coefficients
                    channel_coefs = {}
                    for name, coef in zip(feature_names, model_coef):
                        if 'fb_spend' in name or 'fb_adstock' in name:
                            channel_coefs['Facebook'] = channel_coefs.get('Facebook', 0) + abs(coef)
                        elif 'instagram_spend' in name or 'insta_adstock' in name:
                            channel_coefs['Instagram'] = channel_coefs.get('Instagram', 0) + abs(coef)
                        elif 'tiktok_spend' in name or 'tiktok_adstock' in name:
                            channel_coefs['TikTok'] = channel_coefs.get('TikTok', 0) + abs(coef)
                    
                    # Calculate percentages based on coefficient magnitudes
                    if channel_coefs:
                        total_impact = sum(channel_coefs.values())
                        if total_impact > 0:
                            fb_pct = (channel_coefs.get('Facebook', 0) / total_impact) * 100
                            insta_pct = (channel_coefs.get('Instagram', 0) / total_impact) * 100
                            tiktok_pct = (channel_coefs.get('TikTok', 0) / total_impact) * 100
                            
                            # Display the recommended allocation
                            col_rec1, col_rec2, col_rec3 = st.columns(3)
                            with col_rec1:
                                st.metric("🎯 Facebook", f"{fb_pct:.1f}%", help="Recommended % of total budget based on model coefficients")
                            with col_rec2:
                                st.metric("🎯 Instagram", f"{insta_pct:.1f}%", help="Recommended % of total budget based on model coefficients")
                            with col_rec3:
                                st.metric("🎯 TikTok", f"{tiktok_pct:.1f}%", help="Recommended % of total budget based on model coefficients")
                            
                            # Show actual dollar amounts based on user's total spend
                            st.caption(f"Based on your total spend of **${total_ad_spend:,.2f}**, the optimal allocation would be:")
                            col_dol1, col_dol2, col_dol3 = st.columns(3)
                            with col_dol1:
                                st.info(f"💰 **Facebook:** ${(fb_pct/100) * total_ad_spend:,.2f}")
                            with col_dol2:
                                st.info(f"💰 **Instagram:** ${(insta_pct/100) * total_ad_spend:,.2f}")
                            with col_dol3:
                                st.info(f"💰 **TikTok:** ${(tiktok_pct/100) * total_ad_spend:,.2f}")
                            
                            # Add a comparison with current allocation
                            st.markdown("**Comparison with Your Current Allocation:**")
                            current_allocation = {
                                'Facebook': fb_spend,
                                'Instagram': instagram_spend,
                                'TikTok': tiktok_spend
                            }
                            
                            # Calculate Euclidean distance between current and optimal
                            optimal_allocation = {
                                'Facebook': (fb_pct/100) * total_ad_spend,
                                'Instagram': (insta_pct/100) * total_ad_spend,
                                'TikTok': (tiktok_pct/100) * total_ad_spend
                            }
                            
                            distance = np.sqrt(sum((current_allocation[ch] - optimal_allocation[ch])**2 for ch in current_allocation))
                            
                            if distance < 100:
                                st.success("✅ Your current allocation is very close to the optimal recommendation!")
                            elif distance < 500:
                                st.info("📊 Your current allocation is moderately aligned with optimal recommendations.")
                            else:
                                st.warning("⚠️ Consider reallocating your budget closer to the recommended percentages for better ROI.")
                            
                            # Optional: Show coefficient values for transparency
                            with st.expander("📐 View Model Coefficient Details"):
                                coef_df = pd.DataFrame({
                                    'Channel': list(channel_coefs.keys()),
                                    'Absolute Coefficient Sum': list(channel_coefs.values())
                                }).sort_values('Absolute Coefficient Sum', ascending=False)
                                st.dataframe(coef_df, use_container_width=True)
                                st.caption("Higher coefficient values indicate stronger impact on revenue prediction.")
                        else:
                            st.info("Model coefficients are zero or near-zero. Consider adding more data or features.")
                    else:
                        st.info("Channel-specific coefficients not found in model features.")
                    
                    st.markdown("**Model Interpretation:**")
                    st.markdown("- Lasso Regression automatically selects important features")
                    st.markdown("- Adstock captures delayed/recurring effects of ad spend")
                    st.markdown("- Category dummies account for campaign type differences")
                    
    # ---------- ANALYZE SECTION ----------
    with st.expander("📊 Campaign Analytics Dashboard", expanded=False):
        if "cleaned_df" not in st.session_state or "trained_model" not in st.session_state:
            st.warning("⚠️ Please process your data and train the model first to see analytics.")
        else:
            df = st.session_state["cleaned_df"]
            model = st.session_state["trained_model"]
            
            df_analysis = df.copy()
            df_analysis['fb_adstock'] = adstock(df_analysis['fb_spend'].values)
            df_analysis['insta_adstock'] = adstock(df_analysis['instagram_spend'].values)
            df_analysis['tiktok_adstock'] = adstock(df_analysis['tiktok_spend'].values)
            
            if 'category' in df_analysis.columns:
                df_analysis = pd.get_dummies(df_analysis, columns=['category'], drop_first=True)
            
            feature_cols = [col for col in df_analysis.columns if col not in ['total_revenue', 'date']]

            total_revenue = df["total_revenue"].sum()
            total_ad_spend = df[["fb_spend","instagram_spend","tiktok_spend"]].sum().sum()
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

            tab1, tab2, tab3, tab4 = st.tabs(["💰 Revenue Analysis", "📈 Time Series", "🔥 Channel Performance", "🎯 Channel Contribution"])
            
            with tab1:
                if "category" in df.columns:
                    st.markdown("### Total Revenue by Campaign Category")
                    category_revenue = df.groupby("category")["total_revenue"].sum().sort_values(ascending=True)
                    
                    # Changed to Blues color scale to match theme
                    fig = px.bar(x=category_revenue.values, y=category_revenue.index, 
                                orientation='h', color=category_revenue.values,
                                color_continuous_scale='Blues',
                                text=category_revenue.values)
                    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0),
                                     xaxis_title="Total Revenue ($)",
                                     yaxis_title="Campaign Category",
                                     plot_bgcolor='white',
                                     font=dict(color='black', size=12))
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
                                     hovermode='x unified',
                                     plot_bgcolor='white',
                                     paper_bgcolor='white')
                    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
                    fig.update_yaxes(title_text="Ad Spend ($)", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    df_time['spend_exceeds_revenue'] = df_time['total_spend'] > df_time['total_revenue']
                    if df_time['spend_exceeds_revenue'].any():
                        exceed_df = df_time[df_time['spend_exceeds_revenue']]
                        exceed_dates_list = exceed_df['date'].dt.strftime('%Y-%m-%d').tolist()
                        exceed_dates_str = ', '.join(exceed_dates_list)
                        st.warning(f"⚠️ Ad spend exceeded revenue on: {exceed_dates_str}")
                    else:
                        st.success("✅ Ad spend never exceeded revenue during this period!")
                else:
                    st.info("Date column not found for time series analysis.")
            
            with tab3:
                st.markdown("### Channel Performance Heatmap")
                if "category" in df.columns:
                    if "date" in df.columns:
                        min_date = df["date"].min().date()
                        max_date = df["date"].max().date()
                        
                        col_date1, col_date2 = st.columns(2)
                        with col_date1:
                            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="heatmap_start")
                        with col_date2:
                            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="heatmap_end")
                        
                        filtered_df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
                    else:
                        filtered_df = df.copy()
                    
                    heatmap_data = filtered_df.groupby("category")[["fb_spend","instagram_spend","tiktok_spend"]].sum()
                    
                    if not heatmap_data.empty:
                        heatmap_data_display = heatmap_data.rename(columns={
                            'fb_spend': 'Facebook Spend',
                            'instagram_spend': 'Instagram Spend',
                            'tiktok_spend': 'TikTok Spend'
                        })
                        fig = px.imshow(heatmap_data_display.T, 
                                       text_auto='.0f',
                                       aspect="auto",
                                       color_continuous_scale='Blues',
                                       title="Ad Spend Heatmap")
                        fig.update_layout(height=400,
                                         xaxis_title="Category",
                                         yaxis_title="Channel",
                                         font=dict(color='black', size=12))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No data available for selected date range")
                else:
                    st.info("Category column not found for heatmap.")
            
            with tab4:
                st.markdown("### Channel Contribution Analysis")
                if hasattr(model, 'coef_'):
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
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0),
                                         font=dict(color='black', size=12),
                                         xaxis_title="Coefficient Value",
                                         yaxis_title="Channel")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(
                            channel_features.style.background_gradient(subset=['Coefficient'], cmap='Blues', vmin=-1, vmax=1),
                            use_container_width=True
                        )
                        st.caption("💡 **What is Adstock?** Adstock measures the *carryover effect* of advertising - how past ad spend continues to influence revenue in future days. Higher Adstock means ads have longer-lasting impact.")
                    else:
                        st.info("No channel-specific coefficients found")
                else:
                    st.info("Model coefficients not available")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #64748b;'>
    <p>🚀 Adventa - AI-Powered Campaign Spend Optimizer</p>
    <p style='font-size: 12px;'>Powered by Lasso Regression & Adstock Transformation</p>
</div>
""", unsafe_allow_html=True)
