import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

st.title("🎈 Adventa")
st.write("Optimize your advertisement campaign spend!")

st.title("Data Import - Cleaning & Analyzing")

uploaded_file = st.file_uploader(
    "Upload Advertising Dataset (CSV)",
    type=["csv"]
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
    """Train Lasso Regression model with adstock transformation and category dummies"""
    
    # Required columns
    required_cols = ["total_revenue", "fb_spend", "instagram_spend", "tiktok_spend"]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        return None, f"Missing columns: {', '.join(missing)}"
    
    # Check if we have enough data
    if len(df) < 5:
        return None, "Not enough data to train model. Need at least 5 rows of data."
    
    try:
        # Create a copy for modeling
        df_model = df.copy()
        
        # Create adstock features for each channel
        df_model['fb_adstock'] = adstock(df_model['fb_spend'].values)
        df_model['insta_adstock'] = adstock(df_model['instagram_spend'].values)
        df_model['tiktok_adstock'] = adstock(df_model['tiktok_spend'].values)
        
        # Create one-hot encoding for 'category' column if it exists
        if 'category' in df_model.columns:
            df_model = pd.get_dummies(df_model, columns=['category'], drop_first=True)
        
        # Define feature columns (exclude target and date)
        feature_cols = [col for col in df_model.columns if col not in ['total_revenue', 'date']]
        X = df_model[feature_cols]
        y = df_model['total_revenue']
        
        # Train-test split (keeping time order with shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train Lasso Regression model
        model = Lasso(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store model info in session state
        st.session_state["r2_score"] = r2
        st.session_state["mae"] = mae
        st.session_state["feature_cols"] = feature_cols
        st.session_state["test_indices"] = X_test.index.tolist()
        
        # Store full transformed data for later use in predict section
        st.session_state["df_analysis"] = df_model
        st.session_state["y_actual_full"] = y
        st.session_state["y_predicted_full"] = model.predict(X)
        
        return model, None, r2, mae
        
    except Exception as e:
        return None, f"Error training model: {str(e)}", None, None

def predict_revenue_lasso(df, model, fb_spend, instagram_spend, tiktok_spend, category_value=None):
    """Make prediction using trained Lasso model with adstock and category"""
    
    # Get last adstock values from historical data
    if len(df) > 0:
        last_fb_adstock = adstock(df['fb_spend'].values)[-1] if 'fb_spend' in df.columns else 0
        last_insta_adstock = adstock(df['instagram_spend'].values)[-1] if 'instagram_spend' in df.columns else 0
        last_tiktok_adstock = adstock(df['tiktok_spend'].values)[-1] if 'tiktok_spend' in df.columns else 0
    else:
        last_fb_adstock = 0
        last_insta_adstock = 0
        last_tiktok_adstock = 0
    
    # Apply adstock with decay (0.5)
    decay_rate = 0.5
    fb_adstock_pred = fb_spend + decay_rate * last_fb_adstock
    insta_adstock_pred = instagram_spend + decay_rate * last_insta_adstock
    tiktok_adstock_pred = tiktok_spend + decay_rate * last_tiktok_adstock
    
    # Create base prediction row
    prediction_row = {
        'fb_adstock': fb_adstock_pred,
        'insta_adstock': insta_adstock_pred,
        'tiktok_adstock': tiktok_adstock_pred
    }
    
    # Add original spend columns (some models may use them)
    prediction_row['fb_spend'] = fb_spend
    prediction_row['instagram_spend'] = instagram_spend
    prediction_row['tiktok_spend'] = tiktok_spend
    
    # Add category dummies if category exists in training
    if category_value and 'category' in df.columns:
        # Get unique categories from training data
        unique_cats = df['category'].unique()
        for cat in unique_cats:
            dummy_col = f'category_{cat}'
            if dummy_col in st.session_state.get("feature_cols", []):
                # Set 1 for selected category, 0 for others
                prediction_row[dummy_col] = 1 if cat == category_value else 0
    
    # Ensure all feature columns are present
    feature_cols = st.session_state.get("feature_cols", [])
    for col in feature_cols:
        if col not in prediction_row:
            prediction_row[col] = 0
    
    # Create DataFrame for prediction
    features_df = pd.DataFrame([prediction_row])[feature_cols]
    
    # Make prediction
    predicted_revenue = model.predict(features_df)[0]
    
    return predicted_revenue

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    # ---------- CLEAN DATA ----------
    with st.expander("🧹 Clean Dataset", expanded=False):
        if st.button("Run Clean Dataset"):
            cleaned_df = clean_ad_data(raw_df)
            st.session_state["cleaned_df"] = cleaned_df
            
            # Train Lasso model after cleaning
            model, error, r2, mae = train_prediction_model(cleaned_df)
            
            if model:
                st.session_state["trained_model"] = model
                st.session_state["model_type"] = "lasso"
                st.success(f"Dataset cleaned and Lasso Regression model trained successfully ✅")
                st.info(f"📊 Model Performance: R² Score = {r2:.4f} | MAE = ${mae:,.2f}")
                
                # Show model quality indicator
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
            
            # Preview cleaned data
            st.subheader("Cleaned Data Preview (First 15 Rows)")
            st.dataframe(cleaned_df.head(15))
            
            # Show category breakdown if available
            if 'category' in cleaned_df.columns:
                st.subheader("📁 Category Distribution")
                category_counts = cleaned_df['category'].value_counts()
                st.dataframe(category_counts)
            
            # Download button
            csv = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Dataset",
                data=csv,
                file_name="adventa_cleaned_data.csv",
                mime="text/csv"
            )

    # ---------- PREDICT SECTION ----------
    with st.expander("🎯 Predict Revenue", expanded=False):
        if "cleaned_df" not in st.session_state:
            st.warning("Please clean the dataset first to train the model.")
        elif st.session_state.get("model_type") != "lasso":
            st.warning("Model not trained successfully. Please re-upload and clean data.")
        else:
            # ========== MODEL ACCURACY INTRODUCTION ==========
            st.subheader("📊 Model Accuracy Overview")
            st.write("Before making predictions, see how well our model performs on historical data:")
            
            # Display model accuracy metrics
            col_acc1, col_acc2, col_acc3 = st.columns(3)
            with col_acc1:
                st.metric("🎯 R² Score", f"{st.session_state.get('r2_score', 0):.4f}", 
                         help="Closer to 1.0 means better predictions")
            with col_acc2:
                st.metric("📉 Mean Absolute Error", f"${st.session_state.get('mae', 0):,.2f}",
                         help="Average prediction error in dollars")
            with col_acc3:
                # Calculate MAPE if possible
                y_actual_full = st.session_state.get("y_actual_full")
                y_predicted_full = st.session_state.get("y_predicted_full")
                if y_actual_full is not None and len(y_actual_full) > 0:
                    mape = np.mean(np.abs((y_actual_full - y_predicted_full) / y_actual_full)) * 100
                    st.metric("📊 Avg. Accuracy", f"{100 - mape:.1f}%",
                             help="Average prediction accuracy percentage")
                else:
                    st.metric("📊 Model Status", "Ready")
            
            st.info("💡 **Interpretation:** Higher R² (closer to 1.0) means more reliable predictions. Lower MAE means smaller prediction errors.")
            
            # ---------- ACTUAL VS PREDICTED CHART ----------
            st.subheader("📈 Actual vs Predicted Revenue Over Time")
            st.write("This chart shows how well our model predictions match actual historical revenue:")
            
            # Get stored transformed data
            df_analysis = st.session_state.get("df_analysis")
            df = st.session_state["cleaned_df"]
            model = st.session_state["trained_model"]
            
            if df_analysis is not None and "date" in df.columns:
                feature_cols = st.session_state.get("feature_cols", [])
                
                if len(feature_cols) > 0:
                    # Use stored predictions or recalculate
                    if "y_predicted_full" in st.session_state:
                        y_predicted = st.session_state["y_predicted_full"]
                        y_actual = st.session_state["y_actual_full"]
                    else:
                        X = df_analysis[feature_cols]
                        y_actual = df["total_revenue"]
                        y_predicted = model.predict(X)
                    
                    # Create dataframe with predictions
                    pred_df = df[["date"]].copy()
                    pred_df["Actual Revenue"] = y_actual.values
                    pred_df["Predicted Revenue"] = y_predicted
                    pred_df["date"] = pd.to_datetime(pred_df["date"])
                    pred_df = pred_df.sort_values("date")
                    
                    # Date range selector (default to last 6 months)
                    min_date = pred_df["date"].min()
                    max_date = pred_df["date"].max()
                    default_start = max_date - pd.Timedelta(days=182)
                    
                    col_date1, col_date2 = st.columns(2)
                    with col_date1:
                        start_date = st.date_input("Start Date", value=default_start, min_value=min_date, max_value=max_date, key="pred_start_date")
                    with col_date2:
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="pred_end_date")
                    
                    # Filter based on selected dates
                    mask = (pred_df["date"] >= pd.to_datetime(start_date)) & (pred_df["date"] <= pd.to_datetime(end_date))
                    pred_df_filtered = pred_df[mask]
                    
                    if not pred_df_filtered.empty:
                        st.line_chart(pred_df_filtered.set_index("date"))
                        st.caption(f"📅 Showing data from {start_date} to {end_date}")
                        
                        # Calculate accuracy on selected range
                        if len(pred_df_filtered) >= 2:
                            r2_range = r2_score(pred_df_filtered["Actual Revenue"], pred_df_filtered["Predicted Revenue"])
                            mae_range = mean_absolute_error(pred_df_filtered["Actual Revenue"], pred_df_filtered["Predicted Revenue"])
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("R² Score (selected period)", f"{r2_range:.4f}")
                            with col_b:
                                st.metric("MAE (selected period)", f"${mae_range:,.2f}")
                            
                            # Confidence indicator
                            if r2_range >= 0.8:
                                st.success("✅ **High confidence:** Model predictions are very reliable for this period.")
                            elif r2_range >= 0.6:
                                st.info("📊 **Moderate confidence:** Predictions are reasonably reliable.")
                            else:
                                st.warning("⚠️ **Low confidence:** Consider retraining model with more data.")
                    else:
                        st.warning("No data available for selected date range.")
                    
                    st.markdown("---")
            
            # ========== PREDICTION INPUT SECTION ==========
            st.subheader("🎯 Make New Predictions")
            st.write("Enter ad spend values to predict revenue:")
            
            # Get category options if available
            df = st.session_state["cleaned_df"]
            has_category = 'category' in df.columns
            
            # Input columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fb_spend = st.number_input("Facebook Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="fb_input")
            with col2:
                instagram_spend = st.number_input("Instagram Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="insta_input")
            with col3:
                tiktok_spend = st.number_input("TikTok Spend ($)", min_value=0.0, value=1000.0, step=100.0, key="tiktok_input")
            
            # Category selector (if available)
            category_value = None
            if has_category:
                categories = df['category'].unique().tolist()
                category_value = st.selectbox("Campaign Category", categories)
            
            if st.button("Predict Revenue", type="primary"):
                model = st.session_state["trained_model"]
                
                # Make prediction
                predicted_revenue = predict_revenue_lasso(
                    df, model, fb_spend, instagram_spend, tiktok_spend, category_value
                )
                
                # Calculate metrics
                total_ad_spend = fb_spend + instagram_spend + tiktok_spend
                roi = ((predicted_revenue - total_ad_spend) / total_ad_spend * 100) if total_ad_spend > 0 else 0
                
                # Display results
                st.markdown("---")
                st.subheader("📈 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Ad Spend", f"${total_ad_spend:,.2f}")
                with col2:
                    st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}", 
                             delta=f"${predicted_revenue - total_ad_spend:,.2f}")
                with col3:
                    st.metric("ROI", f"{roi:.1f}%", 
                             delta="Positive" if roi > 0 else "Negative",
                             delta_color="normal" if roi > 0 else "inverse")
                
                # ROI feedback
                if roi < 0:
                    st.error("⚠️ Negative ROI predicted. Consider adjusting your ad spend allocation.")
                elif roi > 100:
                    st.success("🎉 Excellent ROI predicted!")
                elif roi > 50:
                    st.info("✅ Good ROI predicted!")
                
                # Show model quality badge with confidence note
                r2_score_val = st.session_state.get("r2_score", 0)
                confidence_text = "High confidence" if r2_score_val >= 0.8 else "Moderate confidence" if r2_score_val >= 0.6 else "Low confidence"
                st.caption(f"🤖 Lasso Regression Model • R² Score: {r2_score_val:.3f} • {confidence_text}")
                
                # Show details
                with st.expander("🔍 Show calculation details"):
                    st.write("**Adstock Values Used (carryover effect):**")
                    st.write(f"Facebook Adstock: ${fb_spend + 0.5 * adstock(df['fb_spend'].values)[-1] if len(df) > 0 else fb_spend:,.2f}")
                    st.write(f"Instagram Adstock: ${instagram_spend + 0.5 * adstock(df['instagram_spend'].values)[-1] if len(df) > 0 else instagram_spend:,.2f}")
                    st.write(f"TikTok Adstock: ${tiktok_spend + 0.5 * adstock(df['tiktok_spend'].values)[-1] if len(df) > 0 else tiktok_spend:,.2f}")
                    
                    if has_category:
                        st.write(f"**Selected Category:** {category_value}")
                    
                    st.write("**Model Interpretation:**")
                    st.write("- Lasso Regression automatically selects important features")
                    st.write("- Adstock captures delayed/recurring effects of ad spend")
                    st.write("- Category dummies account for campaign type differences")

# ---------- ANALYZE SECTION ----------
with st.expander("📊 Analyze", expanded=False):
    if st.button("Run Analyze"):
        if "cleaned_df" not in st.session_state or "trained_model" not in st.session_state:
            st.warning("Please clean the dataset and train the model first.")
        else:
            df = st.session_state["cleaned_df"]
            model = st.session_state["trained_model"]
            
            # Recreate the transformed dataset for predictions
            df_analysis = df.copy()
            
            # Apply adstock transformations
            df_analysis['fb_adstock'] = adstock(df_analysis['fb_spend'].values)
            df_analysis['insta_adstock'] = adstock(df_analysis['instagram_spend'].values)
            df_analysis['tiktok_adstock'] = adstock(df_analysis['tiktok_spend'].values)
            
            # Apply category dummies if category exists
            if 'category' in df_analysis.columns:
                df_analysis = pd.get_dummies(df_analysis, columns=['category'], drop_first=True)
            
            # Get feature columns (excluding target and date)
            feature_cols = [col for col in df_analysis.columns if col not in ['total_revenue', 'date']]

            # ---------- KPI CARDS ----------
            total_revenue = df["total_revenue"].sum()
            total_ad_spend = df[["fb_spend","instagram_spend","tiktok_spend"]].sum().sum()
            total_campaigns = len(df)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Total Ad Spend", f"${total_ad_spend:,.2f}")
                if st.checkbox("View spend by channel", key="show_channel_spend"):
                    channel_spend = df[["fb_spend","instagram_spend","tiktok_spend"]].sum().to_frame(name="Total Spend")
                    st.dataframe(channel_spend)
            with col2:
                st.metric("📊 Total Revenue", f"${total_revenue:,.2f}")
                if st.checkbox("View revenue timeline", key="show_revenue_timeline"):
                    if "date" in df.columns:
                        revenue_trend = df.groupby("date")["total_revenue"].sum().reset_index()
                        st.line_chart(revenue_trend.set_index("date"))
                if st.checkbox("View revenue by category", key="show_revenue_category"):
                    if "category" in df.columns:
                        category_revenue = df.groupby("category")["total_revenue"].sum().sort_values(ascending=False)
                        st.dataframe(category_revenue)
            with col3:
                st.metric("📈 Total Campaigns", total_campaigns)

            # ---------- HEATMAP: Ad Spend by Category and Channel ----------
            st.subheader("Heatmap – Ad Spend by Campaign Category & Channel")
            if "category" in df.columns:
                # Option to filter by timeframe
                timeframe = st.selectbox("Select timeframe for heatmap", 
                                         ["All time","Past Week","Past Month","Past 6 Months","Past Year"],
                                         key="heatmap_timeframe")
                filtered_df = df.copy()
                if "date" in df.columns:
                    # Convert date to datetime if not already
                    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
                    today = pd.Timestamp.now()
                    
                    if timeframe == "Past Week":
                        filtered_df = filtered_df[filtered_df["date"] >= today - pd.Timedelta(days=7)]
                    elif timeframe == "Past Month":
                        filtered_df = filtered_df[filtered_df["date"] >= today - pd.Timedelta(days=30)]
                    elif timeframe == "Past 6 Months":
                        filtered_df = filtered_df[filtered_df["date"] >= today - pd.Timedelta(days=182)]
                    elif timeframe == "Past Year":
                        filtered_df = filtered_df[filtered_df["date"] >= today - pd.Timedelta(days=365)]

                # Pivot table for heatmap (showing ad spend, not revenue)
                heatmap_data = filtered_df.groupby("category")[["fb_spend","instagram_spend","tiktok_spend"]].sum()
                
                if not heatmap_data.empty:
                    st.dataframe(heatmap_data)

                    # Optional: colored heatmap
                    try:
                        import seaborn as sns
                        import matplotlib.pyplot as plt
                        
                        fig, ax = plt.subplots(figsize=(8,4))
                        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
                        plt.title("Ad Spend Heatmap by Category & Channel")
                        st.pyplot(fig)
                    except ImportError:
                        st.info("Install seaborn and matplotlib for heatmap visualization: pip install seaborn matplotlib")
                else:
                    st.info("No data available for selected timeframe")

            # ---------- CHANNEL CONTRIBUTION (Feature Importance) ----------
            st.subheader("📊 Channel Contribution Analysis")
            
            # Extract coefficients from Lasso model
            if hasattr(model, 'coef_'):
                # Get feature names and coefficients
                coef_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Coefficient': model.coef_
                })
                # Filter for adstock and spend features only
                channel_features = coef_df[coef_df['Feature'].str.contains('adstock|spend', case=False)]
                
                if not channel_features.empty:
                    # Show top contributing channels
                    st.write("**Impact of each channel on revenue:**")
                    st.dataframe(channel_features.sort_values('Coefficient', ascending=False))
                    
                    # Visualize coefficients
                    st.bar_chart(channel_features.set_index('Feature')['Coefficient'])
                else:
                    st.info("No channel-specific coefficients found")
