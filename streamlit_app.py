import streamlit as st
import pandas as pd

st.title("🎈 Adventa")
st.write("Hello world!")

st.title("Data Import - Cleaning & Analyzing")

uploaded_file = st.file_uploader(
    "Upload Advertising Dataset (CSV)",
    type=["csv"]
)

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
