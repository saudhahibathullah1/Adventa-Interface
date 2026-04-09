import streamlit as st
import pandas as pd
import numpy as np

# Try to import XGBoost with error handling
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost not installed. Please install it using: pip install xgboost")

st.title("🎈 Adventa")
st.write("Optimize your advertisement campaign spend!")

st.title("Data Import - Cleaning & Analyzing")

uploaded_file = st.file_uploader(
    "Upload Advertising Dataset (CSV)",
    type=["csv"]
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

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    # ---------- CLEAN DATA ----------
    with st.expander("🧹 Clean Dataset", expanded=False):
        # button inside the expander triggers cleaning
        if st.button("Run Clean Dataset"):
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
            st.subheader("Cleaned Data Preview (First 15 Rows)")
            st.dataframe(cleaned_df.head(15))

            # download full dataset
            csv = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Dataset",
                data=csv,
                file_name="adventa_cleaned_data.csv",
                mime="text/csv"
            )

    # Create two columns for Analyze and Predict buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # ---------- ANALYZE ----------
        with st.expander("📊 Analyze", expanded=False):
            if st.button("Run Analyze"):
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

                        st.metric("Total Revenue", f"{total_revenue:,.2f}")
                        st.metric("Total Ad Spend", f"{total_ad_spend:,.2f}")
                        st.metric("% of Revenue Spent on Ads", f"{ad_spend_pct:.2f}%")
                        
                        # Show additional metrics
                        if len(df) > 0:
                            avg_daily_revenue = total_revenue / len(df)
                            st.metric("Average Daily Revenue", f"{avg_daily_revenue:,.2f}")
    
    with col2:
        # ---------- PREDICT ----------
        with st.expander("🎯 Predict Revenue", expanded=False):
            if "cleaned_df" not in st.session_state:
                st.warning("Please clean the dataset first to train the model.")
            else:
                st.write("Enter ad spend values to predict revenue:")
                
                # Input fields for ad spend
                fb_spend = st.number_input("Facebook Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="fb_input")
                instagram_spend = st.number_input("Instagram Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="insta_input")
                tiktok_spend = st.number_input("TikTok Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="tiktok_input")
                
                if st.button("Predict Revenue"):
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
                        
                        # Optional: Show adstock values used
                        show_details = True
                        
                    else:
                        # Use simple prediction method
                        predicted_revenue, roi_ratio = simple_predict(df, fb_spend, instagram_spend, tiktok_spend)
                        prediction_method = "Simplified Model (Based on Historical ROI)"
                        show_details = False
                    
                    # Display results
                    st.success(f"### Predicted Revenue: ${predicted_revenue:,.2f}")
                    st.caption(f"*Prediction method: {prediction_method}*")
                    
                    # Calculate ROI
                    total_ad_spend = fb_spend + instagram_spend + tiktok_spend
                    roi = ((predicted_revenue - total_ad_spend) / total_ad_spend * 100) if total_ad_spend > 0 else 0
                    
                    col_metrics1, col_metrics2 = st.columns(2)
                    with col_metrics1:
                        st.metric("Total Ad Spend", f"${total_ad_spend:,.2f}")
                    with col_metrics2:
                        st.metric("ROI", f"{roi:.1f}%", 
                                 delta="Positive" if roi > 0 else "Negative",
                                 delta_color="normal" if roi > 0 else "inverse")
                    
                    # Show warning if ROI is negative
                    if roi < 0:
                        st.warning("⚠️ Negative ROI predicted. Consider adjusting your ad spend allocation.")
                    elif roi > 100:
                        st.info("🎉 Excellent ROI predicted!")
                    
                    # Show additional insights
                    if model_type == "simple":
                        st.info(f"💡 Based on historical data, every $1 spent on ads generates ${roi_ratio:.2f} in revenue.")
                    
                    # Show calculation details for XGBoost
                    if show_details:
                        with st.expander("Show calculation details"):
                            st.write("**Adstock Values Used for Prediction:**")
                            st.write(f"Facebook Adstock: ${fb_adstock_pred:,.2f}")
                            st.write(f"Instagram Adstock: ${insta_adstock_pred:,.2f}")
                            st.write(f"TikTok Adstock: ${tiktok_adstock_pred:,.2f}")
                            st.write("*(Adstock accounts for carryover effect from previous spend)*")
