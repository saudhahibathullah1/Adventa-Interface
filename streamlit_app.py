import streamlit as st
import pandas as pd

st.title('🎈 Adventa')

st.write('Hello world!')

st.title("AdVanta – Data Import & Cleaning")

uploaded_file = st.file_uploader(
    "Upload Advertising Dataset (CSV)",
    type=["csv"]
)
