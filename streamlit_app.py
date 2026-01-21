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
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop total_revenue if all zeros
    if "total_revenue" in df.columns:
        if (df["total_revenue"] == 0).all():
            df = df.drop(columns=["total_revenue"])

    return df

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    if st.button("Clean Dataset"):
        cleaned_df = clean_ad_data(raw_df)

        st.success("Dataset cleaned successfully ✅")

        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df.head())

        # ✅ Correct indentation starts here
        csv = cleaned_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="adventa_cleaned_data.csv",
            mime="text/csv"
        )
