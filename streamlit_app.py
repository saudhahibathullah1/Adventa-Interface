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
        
        # Display model accuracy metrics in a single row
        col_acc1, col_acc2, col_acc3 = st.columns(3)
        with col_acc1:
            st.metric("🎯 R² Score", f"{st.session_state.get('r2_score', 0):.4f}")
        with col_acc2:
            st.metric("📉 Mean Absolute Error", f"${st.session_state.get('mae', 0):,.2f}")
        with col_acc3:
            y_actual_full = st.session_state.get("y_actual_full")
            y_predicted_full = st.session_state.get("y_predicted_full")
            if y_actual_full is not None and len(y_actual_full) > 0:
                # Avoid division by zero
                non_zero_mask = y_actual_full != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_actual_full[non_zero_mask] - y_predicted_full[non_zero_mask]) / y_actual_full[non_zero_mask])) * 100
                    st.metric("📊 Accuracy", f"{100 - mape:.1f}%")
                else:
                    st.metric("📊 Status", "Ready")
            else:
                st.metric("📊 Status", "Ready")
        
        # ---------- COMPACT ACTUAL VS PREDICTED (LAST 6 MONTHS ONLY) ----------
        st.subheader("📈 Actual vs Predicted (Last 6 Months)")
        
        # Get stored data
        df = st.session_state["cleaned_df"]
        
        # Debug: Check if we have the required data
        if "y_predicted_full" not in st.session_state:
            st.info("Recalculating predictions for chart...")
            # Recreate transformed dataset
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
            
            # Store for next time
            st.session_state["y_predicted_full"] = y_predicted_full
            st.session_state["y_actual_full"] = y_actual_full
            st.session_state["df_analysis"] = df_temp
        else:
            y_predicted_full = st.session_state["y_predicted_full"]
            y_actual_full = st.session_state["y_actual_full"]
        
        # Create dataframe with predictions
        if "date" in df.columns and len(df) > 0:
            pred_df = pd.DataFrame({
                'date': pd.to_datetime(df['date']),
                'Actual Revenue': y_actual_full,
                'Predicted Revenue': y_predicted_full
            })
            pred_df = pred_df.sort_values('date')
            
            # Filter for last 6 months only
            max_date = pred_df['date'].max()
            six_months_ago = max_date - pd.Timedelta(days=182)
            pred_df_last_6m = pred_df[pred_df['date'] >= six_months_ago]
            
            # Show data info for debugging
            st.caption(f"📊 Total data points: {len(pred_df)} | Last 6 months: {len(pred_df_last_6m)} points")
            
            if not pred_df_last_6m.empty:
                # Make chart smaller using height parameter
                st.line_chart(pred_df_last_6m.set_index('date')[['Actual Revenue', 'Predicted Revenue']], height=300)
                st.caption(f"📅 Last 6 months: {pred_df_last_6m['date'].min().strftime('%Y-%m-%d')} to {pred_df_last_6m['date'].max().strftime('%Y-%m-%d')}")
                
                # Calculate accuracy on last 6 months
                if len(pred_df_last_6m) >= 2:
                    from sklearn.metrics import r2_score, mean_absolute_error
                    r2_6m = r2_score(pred_df_last_6m['Actual Revenue'], pred_df_last_6m['Predicted Revenue'])
                    mae_6m = mean_absolute_error(pred_df_last_6m['Actual Revenue'], pred_df_last_6m['Predicted Revenue'])
                    
                    # Show accuracy metrics in a compact row
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("📊 R² (Last 6M)", f"{r2_6m:.4f}", 
                                 help="Closer to 1.0 = better predictions")
                    with col_metric2:
                        st.metric("💰 MAE (Last 6M)", f"${mae_6m:,.2f}",
                                 help="Average prediction error")
                    
                    # Confidence indicator
                    if r2_6m >= 0.8:
                        st.success("✅ **High confidence** - Model is reliable for recent data")
                    elif r2_6m >= 0.6:
                        st.info("📊 **Moderate confidence** - Predictions are reasonably reliable")
                    else:
                        st.warning("⚠️ **Low confidence** - Consider retraining with more recent data")
                else:
                    st.info("Need at least 2 data points in last 6 months to calculate accuracy.")
            else:
                st.warning(f"No data available in the last 6 months. Earliest date: {pred_df['date'].min().strftime('%Y-%m-%d')}, Latest: {max_date.strftime('%Y-%m-%d')}")
        else:
            st.warning("Date column not found or empty in dataset.")
        
        st.markdown("---")
        
        # ========== PREDICTION INPUT SECTION ==========
        st.subheader("🎯 Make New Predictions")
        st.write("Enter ad spend values to predict revenue:")
        
        # Get category options if available
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
            
            # Show model quality badge
            r2_score_val = st.session_state.get("r2_score", 0)
            st.caption(f"🤖 Lasso Regression • R²: {r2_score_val:.3f}")
            
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
        with col2:
            st.metric("📊 Total Revenue", f"${total_revenue:,.2f}")
        with col3:
            st.metric("📈 Total Campaigns", total_campaigns)

        # ---------- REVENUE BY CATEGORY (Horizontal Bar Chart) ----------
        if "category" in df.columns:
            st.subheader("📊 Total Revenue by Campaign Category")
            category_revenue = df.groupby("category")["total_revenue"].sum().sort_values(ascending=True)
            
            # Create horizontal bar chart using matplotlib
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(category_revenue.index, category_revenue.values, color='skyblue')
                ax.set_xlabel('Total Revenue ($)')
                ax.set_ylabel('Campaign Category')
                ax.set_title('Total Revenue by Campaign Category')
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, category_revenue.values)):
                    ax.text(value, bar.get_y() + bar.get_height()/2, 
                           f'${value:,.0f}', 
                           va='center', ha='left', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            except ImportError:
                # Fallback to st.bar_chart if matplotlib not available
                st.bar_chart(category_revenue)
        else:
            st.info("Category column not found in dataset.")

        # ---------- TIME SERIES CHART: Total Revenue vs Total Spend ----------
        if "date" in df.columns:
            st.subheader("📈 Total Revenue vs Total Ad Spend Over Time")
            
            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Calculate total spend per date
            df_time = df.copy()
            df_time['total_spend'] = df_time[['fb_spend', 'instagram_spend', 'tiktok_spend']].sum(axis=1)
            
            # Sort by date
            df_time = df_time.sort_values('date')
            
            # Create time series chart
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot both lines
                ax.plot(df_time['date'], df_time['total_revenue'], 
                       label='Total Revenue', color='green', linewidth=2, marker='o', markersize=4)
                ax.plot(df_time['date'], df_time['total_spend'], 
                       label='Total Ad Spend', color='red', linewidth=2, marker='s', markersize=4)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Amount ($)')
                ax.set_title('Total Revenue vs Total Ad Spend Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Optional: Add a note about periods where spend exceeded revenue
                df_time['spend_exceeds_revenue'] = df_time['total_spend'] > df_time['total_revenue']
                if df_time['spend_exceeds_revenue'].any():
                    exceed_dates = df_time[df_time['spend_exceeds_revenue']]['date'].dt.strftime('%Y-%m-%d').tolist()
                    st.warning(f"⚠️ Ad spend exceeded revenue on: {', '.join(exceed_dates)}")
                else:
                    st.success("✅ Ad spend never exceeded revenue during this period!")
                    
            except ImportError:
                # Fallback to st.area_chart for simple visualization
                chart_data = df_time[['date', 'total_revenue', 'total_spend']].set_index('date')
                st.line_chart(chart_data)
        else:
            st.info("Date column not found for time series analysis.")

        # ---------- HEATMAP: Ad Spend by Category and Channel ----------
        st.subheader("Heatmap – Ad Spend by Campaign Category & Channel")
        if "category" in df.columns:
            # Date range selector for heatmap
            if "date" in df.columns:
                # Convert date to datetime
                df["date"] = pd.to_datetime(df["date"])
                min_date = df["date"].min().date()
                max_date = df["date"].max().date()
                
                # Create date range selector
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key="heatmap_start")
                with col_date2:
                    end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key="heatmap_end")
                
                # Filter data based on date range
                filtered_df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
                
                # Show selected date range info
                st.caption(f"📅 Showing data from {start_date} to {end_date}")
            else:
                filtered_df = df.copy()
                st.caption("📅 No date column found - showing all data")
        
            # Pivot table for heatmap (showing ad spend, not revenue)
            heatmap_data = filtered_df.groupby("category")[["fb_spend","instagram_spend","tiktok_spend"]].sum()
            
            if not heatmap_data.empty:
                # Show only heatmap, no extra table
                try:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(8,4))
                    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
                    plt.title(f"Ad Spend Heatmap by Category & Channel")
                    st.pyplot(fig)
                except ImportError:
                    st.info("Install seaborn and matplotlib for heatmap visualization: pip install seaborn matplotlib")
            else:
                st.info(f"No data available for selected date range: {start_date} to {end_date}")
        else:
            st.info("Category column not found for heatmap.")
            
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
                # Create alias mapping
                alias_mapping = {
                    'fb_spend': 'Facebook Spend',
                    'instagram_spend': 'Instagram Spend',
                    'tiktok_spend': 'TikTok Spend',
                    'fb_adstock': 'Facebook AdStock',
                    'insta_adstock': 'Instagram AdStock',
                    'tiktok_adstock': 'TikTok AdStock'
                }
                
                # Apply aliases
                channel_features['Feature'] = channel_features['Feature'].replace(alias_mapping)
                
                # Sort by coefficient
                channel_features = channel_features.sort_values('Coefficient', ascending=False)
                
                # Display table with color gradient on Coefficient column
                st.dataframe(
                    channel_features.style.background_gradient(subset=['Coefficient'], cmap='RdYlGn', vmin=-1, vmax=1),
                    use_container_width=True
                )
                
                # Simple Adstock definition
                st.caption("💡 **What is Adstock?** Adstock measures the *carryover effect* of advertising - how past ad spend continues to influence revenue in future days. Higher Adstock means ads have longer-lasting impact.")
            else:
                st.info("No channel-specific coefficients found")
        else:
            st.info("Model coefficients not available")
