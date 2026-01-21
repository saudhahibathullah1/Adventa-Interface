import streamlit as st
import pandas as pd

st.title("🎈 Adventa")
st.write("Hello world!")

st.title("AdVanta – Data Import & Cleaning")

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

    col1, col2 = st.columns(2)

    # -------- CLEAN BUTTON --------
    with col1:
        if st.button("🧹 Clean Dataset"):
    cleaned_df = clean_ad_data(raw_df)

    # ✅ store full dataset
    st.session_state["cleaned_df"] = cleaned_df

    st.success("Dataset cleaned successfully ✅")

    # 👀 preview ONLY (separate variable)
    preview_df = cleaned_df.head(5)

    st.subheader("Cleaned Data Preview (First 5 Rows)")
    st.dataframe(preview_df)

    # ⬇️ download FULL dataset
    csv = st.session_state["cleaned_df"].to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="adventa_cleaned_data.csv",
        mime="text/csv"
    )

    # -------- ANALYZE BUTTON --------
    with col2:
        if st.button("📊 Analyze"):
            df = clean_ad_data(raw_df)

            required_cols = [
                "total_revenue",
                "fb_spend",
                "instagram_spend",
                "tiktok_spend"
            ]

            missing = [c for c in required_cols if c not in df.columns]

            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                total_revenue = df["total_revenue"].sum()

                total_ad_spend = (
                    df["fb_spend"].sum()
                    + df["instagram_spend"].sum()
                    + df["tiktok_spend"].sum()
                )

                ad_spend_pct = (
                    (total_ad_spend / total_revenue) * 100
                    if total_revenue > 0 else 0
                )

                st.metric("Total Revenue", f"{total_revenue:,.2f}")
                st.metric("Total Ad Spend", f"{total_ad_spend:,.2f}")
                st.metric(
                    "% of Revenue Spent on Ads",
                    f"{ad_spend_pct:.2f}%"
                )
