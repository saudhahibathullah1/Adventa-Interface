import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Adventa - Ad Spend Optimizer",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Professional header styling */
    .professional-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .professional-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .professional-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Card styling for expanders */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
    }
    
    /* Metric card styling */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
    }
    
    /* Success/Error/Warning message styling */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #ced4da;
    }
    
    .stNumberInput > label {
        font-weight: 500;
        color: #495057;
    }
    
    /* File uploader styling */
    .stFileUploader > div > button {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 8px;
    }
    
    /* Subheader styling */
    h1, h2, h3 {
        color: #1e3c72;
    }
    
    /* Divider styling */
    hr {
        margin: 1rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, #e0e0e0, transparent);
    }
    
    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    }
    
    /* Info box styling */
    .stInfo {
        background-color: #d1ecf1;
        border-radius: 8px;
    }
    
    /* Professional footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
        color: #6c757d;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown("""
    <div class="professional-header">
        <h1>🎈 Adventa</h1>
        <p>Optimize your advertisement campaign spend with data-driven insights</p>
    </div>
""", unsafe_allow_html=True)

# Try to import XGBoost with error handling
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not installed. Please install it using: pip install xgboost")

st.markdown("## Data Import - Cleaning & Analyzing")

uploaded_file = st.file_uploader(
    "Upload Advertising Dataset (CSV)",
    type=["csv"],
    help="Upload a CSV file containing your advertising data with columns: date, total_revenue, fb_spend, instagram_spend, tiktok_spend"
)

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
        df["date"] = pd.to_datetime(
            df["date"],
            dayfirst=True,
            errors="coerce"
        )

    # Drop total_revenue if all zeros
    if "total_revenue" in df.columns:
        if (df["total_revenue"] == 0).all():
            df = df.drop(columns=["total_revenue"])

    return df

def train_prediction_model(df):
    """Train XGBoost model with adstock transformation"""
    if not XGBOOST_AVAILABLE:
        return None, "XGBoost is not installed. Please install it using: pip install xgboost scikit-learn"
    
    # Required columns
    required_cols = ["total_revenue", "fb_spend", "instagram_spend", "tiktok_spend"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    
    # Check if we have enough data
    if len(df) < 3:
        return None, "Not enough data to train model. Need at least 3 rows of data."
    
    try:
        # Create adstock features
        df_model = df.copy()
        df_model['fb_adstock'] = adstock(df_model['fb_spend'].values)
        df_model['insta_adstock'] = adstock(df_model['instagram_spend'].values)
        df_model['tiktok_adstock'] = adstock(df_model['tiktok_spend'].values)
        
        # Prepare features and target
        feature_cols = ['fb_adstock', 'insta_adstock', 'tiktok_adstock']
        X = df_model[feature_cols]
        y = df_model['total_revenue']
        
        # Train XGBoost model with optimal hyperparameters
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
        return None, f"Error training model: {str(e)}"

def simple_predict(df, fb_spend, instagram_spend, tiktok_spend):
    """Fallback prediction method using linear regression if XGBoost is not available"""
    # Calculate average revenue per ad spend ratio
    total_ad_spend_hist = (df['fb_spend'] + df['instagram_spend'] + df['tiktok_spend']).sum()
    total_revenue_hist = df['total_revenue'].sum()
    
    if total_ad_spend_hist > 0:
        roi_ratio = total_revenue_hist / total_ad_spend_hist
    else:
        roi_ratio = 1
    
    # Predict based on historical ROI
    new_total_spend = fb_spend + instagram_spend + tiktok_spend
    predicted_revenue = new_total_spend * roi_ratio
    
    return predicted_revenue, roi_ratio

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.markdown("### Raw Data Preview")
    st.dataframe(raw_df.head(), use_container_width=True)

    # ---------- CLEAN DATA ----------
    with st.expander("🧹 Clean Dataset", expanded=False):
        # button inside the expander triggers cleaning
        if st.button("Run Clean Dataset", use_container_width=True):
            cleaned_df = clean_ad_data(raw_df)
            st.session_state["cleaned_df"] = cleaned_df
            
            # Train model after cleaning
            if XGBOOST_AVAILABLE:
                model, error = train_prediction_model(cleaned_df)
                if model:
                    st.session_state["trained_model"] = model
                    st.session_state["model_type"] = "xgboost"
                    st.success("Dataset cleaned and XGBoost model trained successfully ✅")
                else:
                    st.error(f"Model training failed: {error}")
                    # Use simple prediction as fallback
                    st.session_state["model_type"] = "simple"
                    st.info("Using simplified prediction model as fallback.")
            else:
                st.session_state["model_type"] = "simple"
                st.info("XGBoost not available. Using simplified prediction model.")
                st.success("Dataset cleaned successfully ✅")

            # preview only first 15 rows
            st.markdown("#### Cleaned Data Preview (First 15 Rows)")
            st.dataframe(cleaned_df.head(15), use_container_width=True)

            # download full dataset
            csv = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Dataset",
                data=csv,
                file_name="adventa_cleaned_data.csv",
                mime="text/csv",
                use_container_width=True
            )

    # ---------- PREDICT SECTION (Top) ----------
    with st.expander("🎯 Predict Revenue", expanded=False):
        if "cleaned_df" not in st.session_state:
            st.warning("Please clean the dataset first to train the model.")
        else:
            st.write("Enter ad spend values to predict revenue:")
            
            # Create three columns for input fields
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fb_spend = st.number_input("Facebook Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="fb_input")
            with col2:
                instagram_spend = st.number_input("Instagram Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="insta_input")
            with col3:
                tiktok_spend = st.number_input("TikTok Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="tiktok_input")
            
            if st.button("Predict Revenue", type="primary", use_container_width=True):
                df = st.session_state["cleaned_df"]
                model_type = st.session_state.get("model_type", "simple")
                
                if model_type == "xgboost" and "trained_model" in st.session_state:
                    # Use XGBoost model
                    model = st.session_state["trained_model"]
                    
                    # Calculate adstock values using last values from dataset
                    if len(df) > 0:
                        # Get last adstock values from training data
                        last_fb_adstock = adstock(df['fb_spend'].values)[-1] if 'fb_spend' in df.columns else 0
                        last_insta_adstock = adstock(df['instagram_spend'].values)[-1] if 'instagram_spend' in df.columns else 0
                        last_tiktok_adstock = adstock(df['tiktok_spend'].values)[-1] if 'tiktok_spend' in df.columns else 0
                        
                        # Apply adstock transformation with decay
                        decay_rate = 0.5
                        fb_adstock_pred = fb_spend + decay_rate * last_fb_adstock
                        insta_adstock_pred = instagram_spend + decay_rate * last_insta_adstock
                        tiktok_adstock_pred = tiktok_spend + decay_rate * last_tiktok_adstock
                    else:
                        fb_adstock_pred = fb_spend
                        insta_adstock_pred = instagram_spend
                        tiktok_adstock_pred = tiktok_spend
                    
                    # Prepare features for prediction
                    features = pd.DataFrame([[fb_adstock_pred, insta_adstock_pred, tiktok_adstock_pred]], 
                                           columns=['fb_adstock', 'insta_adstock', 'tiktok_adstock'])
                    
                    # Make prediction
                    predicted_revenue = model.predict(features)[0]
                    prediction_method = "XGBoost Model"
                    show_details = True
                    
                else:
                    # Use simple prediction method
                    predicted_revenue, roi_ratio = simple_predict(df, fb_spend, instagram_spend, tiktok_spend)
                    prediction_method = "Simplified Model (Based on Historical ROI)"
                    show_details = False
                
                # Display results in a nice container
                st.markdown("---")
                st.markdown("#### 📈 Prediction Results")
                
                # Create metrics row
                col1, col2, col3 = st.columns(3)
                
                total_ad_spend = fb_spend + instagram_spend + tiktok_spend
                roi = ((predicted_revenue - total_ad_spend) / total_ad_spend * 100) if total_ad_spend > 0 else 0
                
                with col1:
                    st.metric("Total Ad Spend", f"${total_ad_spend:,.2f}")
                with col2:
                    st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}", 
                             delta=f"${predicted_revenue - total_ad_spend:,.2f}")
                with col3:
                    st.metric("ROI", f"{roi:.1f}%", 
                             delta="Positive" if roi > 0 else "Negative",
                             delta_color="normal" if roi > 0 else "inverse")
                
                # Show warning or success message
                if roi < 0:
                    st.error("⚠️ Negative ROI predicted. Consider adjusting your ad spend allocation.")
                elif roi > 100:
                    st.success("🎉 Excellent ROI predicted!")
                elif roi > 50:
                    st.info("✅ Good ROI predicted!")
                
                st.caption(f"*Prediction method: {prediction_method}*")
                
                # Show additional insights
                if model_type == "simple":
                    st.info(f"💡 Based on historical data, every $1 spent on ads generates ${roi_ratio:.2f} in revenue.")
                
                # Show calculation details for XGBoost
                if show_details:
                    with st.expander("🔍 Show calculation details"):
                        st.write("**Adstock Values Used for Prediction:**")
                        st.write(f"Facebook Adstock: ${fb_adstock_pred:,.2f}")
                        st.write(f"Instagram Adstock: ${insta_adstock_pred:,.2f}")
                        st.write(f"TikTok Adstock: ${tiktok_adstock_pred:,.2f}")
                        st.write("*(Adstock accounts for carryover effect from previous spend)*")
                        
                        # Show feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            st.write("**Feature Importance:**")
                            importance_df = pd.DataFrame({
                                'Feature': ['Facebook', 'Instagram', 'TikTok'],
                                'Importance': model.feature_importances_
                            })
                            st.bar_chart(importance_df.set_index('Feature'))

    # ---------- ANALYZE SECTION (Bottom) ----------
    with st.expander("📊 Analyze", expanded=False):
        if st.button("Run Analyze", use_container_width=True):
            if "cleaned_df" not in st.session_state:
                st.warning("Please clean the dataset first.")
            else:
                df = st.session_state["cleaned_df"]

                required_cols = ["total_revenue", "fb_spend", "instagram_spend", "tiktok_spend"]
                missing = [c for c in required_cols if c not in df.columns]

                if missing:
                    st.error(f"Missing columns: {', '.join(missing)}")
                else:
                    total_revenue = df["total_revenue"].sum()
                    total_ad_spend = df["fb_spend"].sum() + df["instagram_spend"].sum() + df["tiktok_spend"].sum()
                    ad_spend_pct = (total_ad_spend / total_revenue * 100) if total_revenue > 0 else 0

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Revenue", f"{total_revenue:,.2f}")
                    with col2:
                        st.metric("Total Ad Spend", f"{total_ad_spend:,.2f}")
                    with col3:
                        st.metric("% of Revenue Spent on Ads", f"{ad_spend_pct:.2f}%")
                    
                    # Show additional metrics
                    if len(df) > 0:
                        avg_daily_revenue = total_revenue / len(df)
                        st.metric("Average Daily Revenue", f"{avg_daily_revenue:,.2f}")
                    
                    # Show spend breakdown
                    st.markdown("#### Ad Spend Breakdown")
                    spend_data = {
                        "Platform": ["Facebook", "Instagram", "TikTok"],
                        "Total Spend": [df["fb_spend"].sum(), df["instagram_spend"].sum(), df["tiktok_spend"].sum()]
                    }
                    spend_df = pd.DataFrame(spend_data)
                    st.bar_chart(spend_df.set_index("Platform"))
                    
                    # Show revenue trend if date column exists
                    if "date" in df.columns:
                        st.markdown("#### Revenue Trend Over Time")
                        revenue_trend = df.groupby("date")["total_revenue"].sum().reset_index()
                        st.line_chart(revenue_trend.set_index("date"))

# Professional Footer
st.markdown("""
    <div class="footer">
        <p>Adventa - Data-Driven Advertising Optimization Platform</p>
        <p style="font-size: 0.8rem;">Powered by Machine Learning | Real-time Predictions | ROI Analysis</p>
    </div>
""", unsafe_allow_html=True)
