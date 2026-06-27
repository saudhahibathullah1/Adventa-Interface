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

# ========== FUNCTION DEFINITIONS ==========
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
    
    h2 {
        color: #1e293b;
        font-weight: 700;
        font-size: 1.8rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        padding-left: 0px;
    }
    
    h3 {
        color: #1e293b;
        font-weight: 600;
        font-size: 1.3rem;
        margin-top: 0.75rem;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
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
        font-size: 1.1rem;
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
    
    /* Section highlight animation */
    @keyframes highlight {
        0% { background-color: rgba(59,130,246,0.2); }
        100% { background-color: transparent; }
    }
    
    .section-highlight {
        animation: highlight 1.5s ease-out;
    }
</style>

<script>
    // Function to scroll to section with offset for header
    function scrollToSection(sectionId) {
        const element = document.getElementById(sectionId);
        if (element) {
            const offset = 80;
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - offset;
            
            window.scrollTo({
                top: offsetPosition,
                behavior: "smooth"
            });
            
            // Add highlight class
            element.classList.add('section-highlight');
            setTimeout(() => {
                element.classList.remove('section-highlight');
            }, 1500);
        }
    }
</script>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    .header-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    
    .rocket-icon {
        font-size: 3.5rem;
        filter: drop-shadow(0 0 10px rgba(255,75,75,0.5));
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .title-text {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #FF4B4B, #FF9068, #FFD166);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle-text {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #a8c0ff, #3f2b96);
        background-clip: text;
        -webkit-background-clip: text;
        color: transparent;
        font-size: 1rem;
        font-weight: 500;
        margin: 0;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .badge {
        background: rgba(255,75,75,0.2);
        backdrop-filter: blur(10px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-family: monospace;
        color: #FF9068;
        border: 1px solid rgba(255,75,75,0.3);
        margin-left: 15px;
    }
    
    .nav-button {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        padding: 8px 16px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background: rgba(255,255,255,0.1);
        border-color: rgba(255,75,75,0.5);
    }
    </style>
    
    <div class="header-container">
        <div class="header-content">
            <div class="rocket-icon">🚀</div>
            <div style="flex: 1;">
                <div style="display: flex; align-items: baseline; gap: 15px;">
                    <div class="title-text">ADVENTA</div>
                    <div class="badge">ALPHA</div>
                </div>
                <div class="subtitle-text">📊 AD CAMPAIGN SPEND OPTIMIZING TOOL</div>
            </div>
        </div>
    </div>
    <hr style="height: 2px; background: linear-gradient(90deg, #FF4B4B, #FF9068, #FFD166, transparent); border: none; margin-top: -0.8rem;">
""", unsafe_allow_html=True)

# ========== WELCOME SECTION ==========
st.markdown('<div id="welcome-section"></div>', unsafe_allow_html=True)
st.markdown("## 🎯 Welcome to ADVENTA")
st.markdown("""
### Your Advertisement Campaign Spend Optimizer

**Features:**
- 🤖 **Machine Learning Predictions** using Lasso Regression
- 📊 **Real-time Analytics** with interactive visualizations
- 💰 **Budget Optimization** recommendations
- 🔮 **Revenue Forecasting** based on ad spend

### How it works:
1. Upload your campaign data (CSV format)
2. Let our tool train on your historical performance
3. Get predictions and optimization recommendations
""")

st.markdown("---")

# ========== SIDEBAR NAVIGATION ==========
with st.sidebar:
    st.markdown("### 🚀 ADVENTA")
    st.markdown("---")
    
    # Navigation buttons with HTML links for smooth scrolling
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 8px;">
        <a href="#" style="text-decoration: none;">
            <button style="width: 100%; padding: 10px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: 600;">
                🏠 Home
            </button>
        </a>
        <a href="#data-import" style="text-decoration: none;">
            <button style="width: 100%; padding: 10px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: 600;">
                📁 Data Import
            </button>
        </a>
        <a href="#predict-section" style="text-decoration: none;">
            <button style="width: 100%; padding: 10px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: 600;">
                🎯 Predict
            </button>
        </a>
        <a href="#analytics-section" style="text-decoration: none;">
            <button style="width: 100%; padding: 10px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 10px; cursor: pointer; font-weight: 600;">
                📊 Analytics
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.caption("v1.0.0 | Analyzer")

# ========== DATA IMPORT SECTION ==========
st.markdown('<div id="data-import"></div>', unsafe_allow_html=True)
st.markdown("## 📁 Data Import")
st.markdown("Upload your campaign data to get started")

# ========== DEMO DATA SECTION ==========
# Add this right after the "Data Import" header and before the file uploader

st.markdown("### 📥 Download Demo Dataset")
st.markdown("Don't have data? Download our sample dataset to test the tool:")

# Your actual demo data as a CSV string
demo_csv_data = """date,campaign_ID,category,fb_spend,instagram_spend,tiktok_spend,total_revenue
1/1/2024,1466,Home,2185.43,3817.64,2615.58,10823.83
1/1/2024,1663,Home,761.38,3504.85,2183.68,8478.24
1/1/2024,1276,Electronics,592.63,3888.67,2947.06,9190.25
1/1/2024,1560,Home,1869.09,2241.6,1625.42,7405.63
1/1/2024,1686,Clothing,3253.34,816.13,1164.08,6986.2
8/1/2024,1130,Electronics,1398.53,2202.67,2154.97,6477.42
8/1/2024,1564,Clothing,3233.95,930.94,414.67,6258.62
8/1/2024,1508,Beauty,3579.05,1928.56,602.73,9072.7
8/1/2024,1389,Clothing,654.75,3664.49,1053.97,6378.17
8/1/2024,1729,Beauty,2960.2,983.96,3399.63,10345
15/1/2024,1337,Home,4727.75,3610.86,2173.07,13888.88
15/1/2024,1379,Electronics,4229.32,1619.99,1127.08,9010.82
15/1/2024,1674,Home,1134.16,3268.13,446.02,6776.09
15/1/2024,1040,Beauty,3970.72,573.97,1382.94,8805.64
15/1/2024,1727,Electronics,4383.97,2606.2,1291.96,11425.5
22/1/2024,1960,Home,3783.23,2658.96,3127.8,12679.66
22/1/2024,1870,Home,1038.17,2939.01,2710.59,9378.39
22/1/2024,1460,Home,2852.3,1881.9,283.88,7234.67
22/1/2024,1406,Home,641.43,2654.72,1237.37,6349.89
22/1/2024,1159,Home,2346.72,3095.54,955.03,8699.82
29/1/2024,1645,Beauty,1803.88,896.52,3268,8043.62
29/1/2024,1330,Beauty,4116.52,990.31,3145.44,11815.8
29/1/2024,1957,Electronics,4133.48,3615.54,1249.41,12271.4
29/1/2024,1455,Home,4181.07,3484.7,222.94,10885.33
29/1/2024,1373,Beauty,2378.35,1121.8,595.56,5967.01
5/2/2024,1496,Clothing,2834.56,2901.17,1399.98,9620.31
5/2/2024,1080,Home,4831.01,1231.59,1840.92,10520.42
5/2/2024,1996,Beauty,3243.04,2159.91,369.88,7620.01
5/2/2024,1415,Home,4587.2,1186.38,678.15,9406.78
5/2/2024,1766,Clothing,3524.61,3117.99,984.2,10517.65
12/2/2024,1608,Electronics,2155.02,2639.53,2290.65,8972.38
12/2/2024,1659,Beauty,1943.51,990.12,334.56,4642.34
12/2/2024,1819,Electronics,3549.04,361.37,1889.91,8385.78
12/2/2024,1640,Home,3609.22,1730.92,3291.21,11002.04
12/2/2024,1763,Beauty,2034.8,719.85,3251.49,7711.42
19/2/2024,1001,Electronics,4177.5,2354.24,1947.85,11942.24
19/2/2024,1098,Home,918.96,3619.7,3171.38,9169.69
19/2/2024,1119,Electronics,3766.8,3619.31,3127.39,14703.53
19/2/2024,1226,Beauty,3389.14,611.32,733.37,6871.61
19/2/2024,1895,Clothing,956.62,2754.96,216.7,5700.12
26/2/2024,1800,Clothing,2969.3,2860.01,2351.47,11360.98
26/2/2024,1134,Home,1964.3,3062.02,2343.79,9262.38
26/2/2024,1749,Clothing,3459.26,2402.74,509.13,8238.47
26/2/2024,1282,Home,4878.55,1754.46,3143.75,12985.23
26/2/2024,1828,Beauty,4076.65,2159.76,2103.78,10744
4/3/2024,1016,Beauty,1763.48,389.97,2330.06,5930.99
4/3/2024,1180,Home,4732.06,3829.54,3219.05,15877.29
4/3/2024,1980,Beauty,4836.29,3456.13,1171.68,13523.59
4/3/2024,1569,Home,4330.12,1472.61,759.33,9675.82
4/3/2024,1853,Home,3065.28,659.55,2229.52,8177
11/3/2024,1810,Clothing,1130.38,2217.82,3095.33,8729.36
11/3/2024,1866,Clothing,2117.71,1386.29,2870.89,8956.91
11/3/2024,1320,Beauty,4401.83,3678.99,1887.43,14369.03
11/3/2024,1470,Home,3658.85,3244.43,3137.02,13474.11
11/3/2024,1574,Clothing,2190.12,647.73,2108.32,6637.39
18/3/2024,1691,Clothing,1789.44,2486.08,300.65,6009.51
18/3/2024,1619,Electronics,4201.7,1632.71,619.3,9948.42
18/3/2024,1349,Home,2557.69,3415.49,841.45,9350.69
18/3/2024,1786,Beauty,3647.8,811.91,638.06,7239.26
18/3/2024,1829,Home,3848.19,1228.18,808.3,8095.77
25/3/2024,1095,Clothing,2427.42,2847.45,392.04,7755.53
25/3/2024,1896,Beauty,922.43,976.6,3284.23,7010.16
25/3/2024,1085,Electronics,2825.13,2731.31,1637.72,9161.88
25/3/2024,1093,Clothing,2038.46,639.66,510.72,4616.6
25/3/2024,1611,Electronics,4907.8,948.72,256.63,9252.54
1/4/2024,1757,Electronics,2591.03,2704.16,358.59,8132.21
1/4/2024,1324,Electronics,4490.06,1265.31,250.5,8681.5
1/4/2024,1496,Beauty,1233.2,561.1,2319.98,5741.17
1/4/2024,1928,Electronics,3135.99,3778.85,2099.06,12928.37
1/4/2024,1416,Beauty,2955.28,3783.42,1474.14,10501.43
8/4/2024,1848,Clothing,4574.08,1024.43,428.89,8848.04
8/4/2024,1417,Home,1935.39,3426.04,276.8,7377.75
8/4/2024,1419,Electronics,1768.35,737.21,2499.23,7059.77
8/4/2024,1456,Beauty,4115.66,1343.53,785.55,9016.39
8/4/2024,1663,Home,4130.76,3964.87,1561.64,13863.27
15/4/2024,1524,Home,4688.41,3476.13,1615.68,13194.61
15/4/2024,1341,Electronics,3895.44,681.56,3178.42,11196.03
15/4/2024,1232,Home,4529.85,1740.05,235.76,9350.25
15/4/2024,1366,Beauty,910.79,1481.46,3335.2,7342.82
15/4/2024,1890,Electronics,2518,1384.88,1284.59,7416.43
22/4/2024,1934,Clothing,3885.69,3228.84,2805.74,13386.79
22/4/2024,1234,Home,2972.88,1933.66,3129.42,10279.75
22/4/2024,1518,Clothing,1026.8,829.07,2712.99,5462.47
22/4/2024,1633,Clothing,3654.36,569.22,2912.14,9837.94
22/4/2024,1499,Electronics,866.07,613.9,3455.91,5520.22
29/4/2024,1069,Beauty,4762.62,3948.2,2686.15,14796.24
29/4/2024,1038,Clothing,875.75,3175.44,2042.73,8321.8
29/4/2024,1057,Beauty,2716.81,342.01,1746.58,6535.75
29/4/2024,1189,Electronics,1034.68,734.85,2342.39,5577.87
29/4/2024,1933,Electronics,2186.92,1357.13,3066.38,9490.67
6/5/2024,1228,Beauty,4834.5,344.97,3400.6,11892.98
6/5/2024,1232,Electronics,2992.34,3886.42,1926.22,11858.62
6/5/2024,1952,Home,3630.87,1981.8,2270.94,11183.93
6/5/2024,1971,Beauty,1764.33,3816.52,3137.87,11465.09
6/5/2024,1177,Beauty,3290.6,1326.31,820.8,6583.12
13/5/2024,1501,Home,849.81,3905.26,3454.5,10234.08
13/5/2024,1557,Home,2912.43,1445.25,2885.52,9260.89
13/5/2024,1814,Electronics,4201.42,3814.26,2594.87,14620.47
13/5/2024,1223,Home,2382.09,3751.1,3058.01,12708.9
13/5/2024,1853,Home,1176.88,2498.28,1456.94,6526.82
20/5/2024,1822,Clothing,4289.54,3401.82,1746.69,12155.49
20/5/2024,1770,Electronics,3612.71,2277.08,2674.71,11289.23
20/5/2024,1456,Clothing,3133.17,2986.65,2698.37,11145.08
20/5/2024,1448,Electronics,1631.49,1316.51,883.85,4900.7
20/5/2024,1437,Clothing,3906.5,473.52,1086.62,7778.45
27/5/2024,1235,Electronics,4241.17,1438.78,2894.07,11492.04
27/5/2024,1228,Electronics,897.84,3229.73,2146.85,6817.23
27/5/2024,1703,Beauty,3377.13,3278.67,3180.4,12816.9
27/5/2024,1993,Beauty,4912.08,2549.93,2300.93,13009.85
27/5/2024,1541,Home,2963.51,1968.37,3204.56,10907.82
3/6/2024,1784,Beauty,2856.21,2881.27,2828.36,12030.08
3/6/2024,1449,Electronics,798.06,469.69,2248.66,4434.05
3/6/2024,1847,Home,1441.09,2444.7,1327.16,5844.9
3/6/2024,1899,Beauty,2301.35,2881.37,794.22,9166.94
3/6/2024,1050,Home,2352.48,3534.98,1900.28,10442.55
10/6/2024,1533,Electronics,4198.06,1576.81,1347.14,10139.99
10/6/2024,1793,Electronics,2969.22,2277.37,1374.77,8454.89
10/6/2024,1106,Home,2665.15,2844.8,1888.47,9381.97
10/6/2024,1691,Home,2197.79,309.6,3065.39,7468.39
10/6/2024,1027,Home,2914.66,3718.95,979.18,9994.03
17/6/2024,1362,Home,2890.7,2965.91,405.73,9112.03
17/6/2024,1882,Home,637.12,3508.7,1368.68,7394.92
17/6/2024,1085,Clothing,971.91,3028.4,801.54,6951.43
17/6/2024,1024,Beauty,2909.01,1162.9,1331.66,7584.95
17/6/2024,1622,Beauty,2097.97,2700.64,1782.62,9578.34
24/6/2024,1612,Clothing,3139.41,2388.5,1449.95,9831.93
24/6/2024,1376,Electronics,4548.41,2547.95,1006.37,10303.03
24/6/2024,1623,Beauty,533.9,1133.73,1405.68,3950.78
24/6/2024,1074,Beauty,4328.68,625.18,2859.35,10823.96
24/6/2024,1997,Home,523.85,3907.75,1819.47,8503.15
1/7/2024,1644,Clothing,4193.88,2958.29,1965.62,13116.33
1/7/2024,1531,Electronics,4855.97,2930.52,858.37,12080.79
1/7/2024,1510,Beauty,2884.28,2916.75,2733.67,11405.07
1/7/2024,1009,Electronics,1942.89,2497.37,1418.46,7425.22
1/7/2024,1612,Electronics,785.19,3962.85,1263.77,7545.03
8/7/2024,1418,Clothing,3921.03,2503.86,1756.2,11624.13
8/7/2024,1458,Electronics,980.15,3447.69,2661.72,9488.27
8/7/2024,1319,Clothing,1422.51,1703.15,3257.28,8184.33
8/7/2024,1290,Electronics,716.43,3191.6,2932.2,9123.79
8/7/2024,1573,Beauty,1338.82,1172.06,2291.4,6600.57
15/7/2024,1799,Home,1922.88,2476.73,2453.84,9934.87
15/7/2024,1992,Electronics,3308.46,2297.19,1647.86,9727.28
15/7/2024,1578,Home,2099.13,1748.48,1955.13,6623.34
15/7/2024,1622,Electronics,3787.47,2866.76,750.21,11024.49
15/7/2024,1288,Clothing,2729.41,3043.41,2091.4,11045.27
22/7/2024,1841,Electronics,1420.45,2942.04,1830.14,8511.88
22/7/2024,1014,Electronics,963.12,2284.98,1450.11,7283.89
22/7/2024,1684,Clothing,2929.37,2099.52,1549.55,8516.07
22/7/2024,1659,Home,554.91,2514.24,2066.18,6939.34
22/7/2024,1022,Home,4815.84,1567.34,950.26,11185.41
29/7/2024,1682,Beauty,1795.68,2575.32,3209.11,10326.29
29/7/2024,1804,Beauty,959.96,3699.38,2807.28,9149.61
29/7/2024,1419,Clothing,3431.15,3153.61,1435.64,10655.48
29/7/2024,1370,Beauty,4268.71,1528.94,1227.77,9616.32
29/7/2024,1665,Electronics,3235.52,1703.43,2656.02,10563.07
5/8/2024,1805,Clothing,1014.2,1833.67,3049.61,7249.4
5/8/2024,1762,Home,2595.65,2079.1,3230.9,10251.65
5/8/2024,1500,Beauty,1617.02,2437.24,746.19,6530.7
5/8/2024,1048,Home,1901.63,3187.83,1116.04,7999.64
5/8/2024,1341,Beauty,4889.94,1998.26,2039.11,12052.03
12/8/2024,1434,Home,2907.81,982.51,1188.67,6629.95
12/8/2024,1304,Electronics,4099.02,1592.73,1745.17,9507.3
12/8/2024,1193,Clothing,2199.77,3395.29,2138.6,10799.85
12/8/2024,1830,Beauty,2483.43,1782.82,2046.81,7998.67
12/8/2024,1877,Clothing,1318.68,3488.61,3322.18,9796.07
19/8/2024,1177,Home,2339.3,393.93,715.3,4490.51
19/8/2024,1327,Clothing,3465.16,400.26,932.51,7197.93
19/8/2024,1603,Home,2026.72,2406.94,3128.62,9358.92
19/8/2024,1911,Beauty,3743.66,3279.67,3462.8,14659.32
19/8/2024,1822,Beauty,4749.84,822,1541.09,10252.83
26/8/2024,1081,Clothing,891.16,2642.99,2628.49,8091.97
26/8/2024,1385,Home,4597.81,2611.62,1308.36,12296.86
26/8/2024,1225,Electronics,3156.33,1291.84,2259.69,9748.86
26/8/2024,1279,Electronics,1825.1,3809.28,2719.9,9868.66
26/8/2024,1114,Electronics,4408.11,2103.5,3152.02,12961.28
2/9/2024,1358,Electronics,1709.05,2304.05,2290.48,8980.44
2/9/2024,1726,Electronics,1127.1,3389.24,3448.53,9590.67
2/9/2024,1355,Electronics,582.76,3682.91,588.58,6418.63
2/9/2024,1873,Clothing,1733.25,2350.46,2349.69,8240.53
2/9/2024,1167,Clothing,1115.99,3630.07,3083.84,10357.15
9/9/2024,1475,Clothing,3202.33,2760.64,778.73,9247.75
9/9/2024,1362,Electronics,2835.13,473.77,748.74,6170.38
9/9/2024,1511,Beauty,872.59,2531.66,1009.65,5049.62
9/9/2024,1097,Home,3735.71,1399.35,2069.14,9506.47
9/9/2024,1490,Electronics,3486.52,3766.27,2617.49,13570.7
16/9/2024,1153,Clothing,2733.65,2508.32,1303,8165.3
16/9/2024,1041,Electronics,979.69,578.01,2603.02,6332.09
16/9/2024,1171,Home,1608.81,3330.68,2838.07,10125.15
16/9/2024,1317,Beauty,1724.65,2483.85,1391.21,6825.76
16/9/2024,1726,Electronics,4428.26,3009.23,2861.65,13875.25
23/9/2024,1921,Electronics,3615.24,3442.02,1023.9,10362.89
23/9/2024,1710,Electronics,4748.27,445.88,2528.4,10707.18
23/9/2024,1497,Home,1312.59,2401.4,3221.11,9179.36
23/9/2024,1934,Home,4749.2,2054.59,3044.74,13341.82
23/9/2024,1187,Electronics,1935.95,3366.99,322.13,6964.99
30/9/2024,1182,Home,846.29,2876.27,1321.59,6845.01
30/9/2024,1721,Electronics,794.1,1466.57,1980.32,5148.78
30/9/2024,1310,Home,4486.9,2578.69,968.77,11766.94
30/9/2024,1738,Home,4415.44,378.7,3086.52,11175.99
30/9/2024,1476,Clothing,4990.7,1597.63,2731.72,12864.54
7/10/2024,1388,Electronics,2659.44,2621.77,3083.13,10256.31
7/10/2024,1669,Clothing,1574.5,708.75,1370.25,4048.13
7/10/2024,1343,Beauty,1833.39,1164.35,338.91,3703.59
7/10/2024,1938,Beauty,1482.14,3814.86,2794.94,9932.76
7/10/2024,1152,Home,2379.11,3552.74,3317.62,13323.23
14/10/2024,1574,Clothing,4960.26,1157.19,3311.01,12586
14/10/2024,1752,Clothing,3234.82,2196.95,961.21,8602.95
14/10/2024,1046,Home,4008.13,1595.46,390.88,8255.89
14/10/2024,1689,Clothing,4477.04,3732.68,3483.2,15561.18
14/10/2024,1997,Electronics,3632.09,869.41,2892.25,9462.28
21/10/2024,1616,Beauty,1507.18,2286.81,2156.7,7401.61
21/10/2024,1920,Clothing,1695.2,779.21,3132.87,7397.57
21/10/2024,1983,Beauty,4379.57,3295.21,2362.3,13262.61
21/10/2024,1601,Clothing,2177.1,1261.09,2587.29,8062.18
21/10/2024,1348,Home,864.71,1114.68,2454.75,3951.44
28/10/2024,1858,Electronics,4211.06,1586.89,2437.45,10869.38
28/10/2024,1570,Beauty,1701.63,3550.93,2831.51,11032.96
28/10/2024,1169,Beauty,3687.63,3396.95,2501.66,13160.31
28/10/2024,1669,Home,3283.75,3085.05,723.4,9619.11
28/10/2024,1988,Beauty,1223.42,3326.48,2946.04,9278.02
4/11/2024,1222,Clothing,528.74,1362.04,2235.86,5259.12
4/11/2024,1325,Home,3353.03,2297.95,2773.49,11390.05
4/11/2024,1221,Electronics,3924.63,2302.69,3377.87,13538.38
4/11/2024,1241,Beauty,961.29,3767.75,2470.02,8583.29
4/11/2024,1473,Beauty,1854.34,2920.24,422.26,6179.55
11/11/2024,1284,Beauty,705.84,3524.69,3412.51,10218.38
11/11/2024,1946,Home,3873.43,781.32,2702.27,10003.07
11/11/2024,1284,Beauty,3574.83,1949.84,1102.97,8202.98
11/11/2024,1883,Beauty,2417.82,1970.13,739.96,7400.17
11/11/2024,1538,Home,870.71,2817.85,2359.89,8294.79
18/11/2024,1954,Beauty,4778.89,858.91,1626.7,10179.66
18/11/2024,1256,Clothing,2289.17,1314.6,3447.13,9595.92
18/11/2024,1523,Clothing,4523.45,1150.83,903.25,9402.26
18/11/2024,1429,Electronics,4389.61,2050.88,3395.04,13209.91
18/11/2024,1386,Electronics,4408.8,3173.41,2744.04,14261.45
25/11/2024,1898,Beauty,1090.6,420.35,3238.8,6372.67
25/11/2024,1739,Electronics,4084.42,2081.63,587.12,9724.78
25/11/2024,1500,Electronics,1402.36,2118.9,411.89,4973.68
25/11/2024,1989,Beauty,3156.19,1903.01,3448.36,11286.73
25/11/2024,1010,Electronics,2279.95,3906.39,2582.23,11007.96
2/12/2024,1552,Electronics,3906.77,3824.8,3195.8,15137.28
2/12/2024,1309,Clothing,962.57,2412.35,260.61,5315.65
2/12/2024,1925,Electronics,1446.57,1580.26,2760.16,8047.96
2/12/2024,1165,Home,4210.31,1623.86,1262.91,9987.77
2/12/2024,1871,Home,1403.87,2864.82,2910.75,9373.85
9/12/2024,1072,Home,1058.99,3616.02,2162.76,8747.61
9/12/2024,1083,Clothing,3258.25,1405.69,3286.17,10731.3
9/12/2024,1207,Beauty,3255.16,2092.27,2380.71,10720.6
9/12/2024,1011,Home,3188.64,3200.18,1786.98,11930.39
9/12/2024,1146,Home,3426.32,2073.33,2231.95,10283
16/12/2024,1811,Clothing,2615.82,3053.83,3077.05,11304.67
16/12/2024,1311,Electronics,593.88,2717.07,1296.64,5938.05
16/12/2024,1206,Electronics,2719.75,601.65,1530.76,6492.53
16/12/2024,1953,Electronics,4282.04,944.52,3370.58,10989.65
16/12/2024,1561,Electronics,4308.1,2626.43,2626.66,13042.8
23/12/2024,1865,Beauty,2812.98,2475.82,297,8584.55
23/12/2024,1642,Electronics,3227.85,2511.58,2816.76,11042.84
23/12/2024,1793,Beauty,2596.16,3393.19,978.11,8797.6
23/12/2024,1292,Electronics,2323.9,2526.86,2287.38,9630.94
23/12/2024,1380,Home,811.1,1917.51,521.4,3172.75
30/12/2024,1856,Electronics,3168.13,2443.84,2386.65,10927.28
30/12/2024,1448,Beauty,2295.71,3474.55,229.8,7879.43
30/12/2024,1629,Clothing,3194.21,1767.64,982.19,8484.45
30/12/2024,1360,Electronics,1989,1960.48,2408.99,8273.16
30/12/2024,1283,Electronics,2522.45,2475.17,3330.33,10626.78
6/1/2025,1304,Electronics,3224.92,1619.91,1960.59,8643.64
6/1/2025,1915,Clothing,3674.24,1951.56,3051.91,12502.68
6/1/2025,1281,Electronics,4940.12,754.99,962.73,10700.84
6/1/2025,1275,Electronics,3380.01,1131.39,794.03,8351.54
6/1/2025,1216,Clothing,2659.54,2838.08,257.35,8373.57
13/1/2025,1293,Beauty,4878.94,2446.42,1038.94,10684.71
13/1/2025,1945,Clothing,1065.84,791.04,673.69,3582.04
13/1/2025,1357,Electronics,3798.05,900.81,821.55,7493.59
13/1/2025,1118,Clothing,4835.06,1244.63,2383.39,11972.62
13/1/2025,1880,Beauty,2184.34,364.81,1005.69,5049.48
20/1/2025,1000,Home,1008.99,3243.39,3231.64,9691.24
20/1/2025,1961,Clothing,4733.94,3359.3,2401.01,13423.74
20/1/2025,1088,Beauty,3222.29,1477.01,3059.25,10896.38
20/1/2025,1101,Clothing,3885.92,634.78,1152.66,8188.49
20/1/2025,1396,Electronics,4665.54,1921.24,3174.54,12912.69
27/1/2025,1724,Home,1927.38,3514.93,1439.72,9677.94
27/1/2025,1914,Beauty,3335.94,2556.38,2059.35,10131.02
27/1/2025,1266,Home,2968.25,2507,2000.77,9960.88
27/1/2025,1776,Home,1258,2677.53,2705.84,8597.45
27/1/2025,1739,Home,2940.64,3820.47,2995.65,12157.01
3/2/2025,1363,Home,2274.64,2332.83,2664.69,9727.75
3/2/2025,1050,Home,2845.55,3342.86,1621.84,10624.9
3/2/2025,1333,Electronics,1387.19,2072.98,842.04,5976.13
3/2/2025,1647,Beauty,3278.08,2235.41,2411.32,10745.63
3/2/2025,1588,Beauty,903.32,2955.25,1519.69,7405.26
10/2/2025,1158,Home,4204.28,2233.18,606.63,10102.17
10/2/2025,1951,Beauty,1828.75,3235.61,811.25,8120.33
10/2/2025,1558,Home,2410.37,622.04,2891.97,8133.07
10/2/2025,1219,Electronics,511.63,2660.55,1900.08,5367.8
10/2/2025,1151,Beauty,4071.85,2692.44,776.75,10909.92
17/2/2025,1663,Home,3004.87,1245.89,1196.8,7436.25
17/2/2025,1438,Clothing,4330.28,2784.27,2871.45,13276.83
17/2/2025,1672,Home,1188.43,1170.37,3068.56,7202.49
17/2/2025,1522,Clothing,2909.11,960.25,1230.5,6738.42
17/2/2025,1920,Clothing,1655.98,2638.46,392.9,6063.95
24/2/2025,1936,Electronics,1735.43,2509.07,2430.28,7847.17
24/2/2025,1579,Clothing,571.19,1562.03,2901.73,5382.43
24/2/2025,1675,Home,2017.52,3650.44,2749.66,11476.21
24/2/2025,1998,Beauty,2305.35,1657.74,1333.75,6558.6
24/2/2025,1775,Clothing,2490.84,3377.89,2409.69,11117.23
3/3/2025,1057,Home,3167.65,2816.49,961.74,9266.06
3/3/2025,1661,Beauty,1356.46,493.65,1448.97,3439.64
3/3/2025,1302,Electronics,899.45,3858.16,2295.37,8804.76
3/3/2025,1278,Beauty,1267.15,1815.07,1594.87,5006.37
3/3/2025,1247,Beauty,4883.74,374.1,2632.69,10721.52
10/3/2025,1799,Electronics,1596.46,2784.77,506.67,7029.54
10/3/2025,1508,Clothing,3238.19,2175.3,725.7,8212.86
10/3/2025,1760,Electronics,1029.42,1141.22,1327.53,3852.53
10/3/2025,1665,Beauty,2793.76,3353.56,1123.02,9302.57
10/3/2025,1243,Clothing,4430.01,898.53,2794.71,12011.6
17/3/2025,1428,Home,2755.19,1769.56,1981.12,9757.08
17/3/2025,1525,Electronics,3534.37,2270.18,2580.69,11168.92
17/3/2025,1930,Clothing,1951.38,384.71,707.51,4051.15
17/3/2025,1288,Beauty,4807.95,2420.25,3278.01,13930.77
17/3/2025,1237,Electronics,3228.67,815.03,3164.26,11014.26
24/3/2025,1916,Clothing,1928.89,3825.35,2923.16,10974.31
24/3/2025,1011,Beauty,2777.48,1160.44,463.89,6407.23
24/3/2025,1446,Clothing,1049.16,2498.88,3290.28,8583.71
24/3/2025,1662,Home,501.5,365.44,2188.43,4169.75
24/3/2025,1689,Electronics,2239.07,2388.43,832.88,7423.48
31/3/2025,1904,Home,3287.74,1400.21,1620.11,7955.87
31/3/2025,1608,Beauty,835.83,3195.98,1075.78,6973.72
31/3/2025,1379,Beauty,3349.35,1146.91,967.73,7215.07
31/3/2025,1702,Clothing,1281.89,2726.18,2998.45,9503.55
31/3/2025,1981,Clothing,3443.34,1610.67,1934.17,9916.67
7/4/2025,1404,Clothing,1295.99,1732.19,3380.61,7968.97
7/4/2025,1887,Home,4056.52,2453.95,2447.22,12016.87
7/4/2025,1996,Electronics,3285.27,1496.95,2953.2,9820.56
7/4/2025,1394,Home,691.85,1674.21,1077.31,4763.44
7/4/2025,1226,Electronics,2747,1284.27,692.48,6987.2
14/4/2025,1553,Electronics,3263.37,3480.39,1494.9,10594.29
14/4/2025,1241,Clothing,1099.15,677.95,2373.68,5279.57
14/4/2025,1758,Clothing,1638.76,3190.33,2922.94,9923.22
14/4/2025,1172,Home,2238.57,1965.46,948.98,6989.83
14/4/2025,1007,Electronics,975.62,3956.62,1662.8,9090.47
21/4/2025,1016,Electronics,671.1,3060.97,2374.82,7774.19
21/4/2025,1227,Home,2788.61,1734.1,1724.42,8553.64
21/4/2025,1772,Clothing,4868.37,3786.57,2786.68,15030.78
21/4/2025,1622,Beauty,4237.41,554.24,1600.18,8107.53
21/4/2025,1616,Electronics,2815.96,3625.93,2782.2,11845.6
28/4/2025,1777,Electronics,4819.22,2069.54,3156.19,13266.43
28/4/2025,1911,Beauty,1063.55,2055.04,576.06,5373.96
28/4/2025,1272,Beauty,3274.25,879.56,344.33,6454.16
28/4/2025,1625,Clothing,3193.39,651.54,1440.78,6283.2
28/4/2025,1434,Clothing,1732.06,3734.77,2430.74,10857.31
5/5/2025,1779,Electronics,557.22,3971.94,2415.8,9271.41
5/5/2025,1637,Beauty,2948.07,2653.46,929.51,8806.92
5/5/2025,1661,Home,3866.86,2363.91,795.96,9755.03
5/5/2025,1769,Electronics,1711.99,2296.36,2602.23,8493.7
5/5/2025,1043,Electronics,3583.84,639.92,3384.93,9671.52
12/5/2025,1514,Home,2360.37,3105.28,2653.3,10741.51
12/5/2025,1228,Home,976.69,860.81,509.4,3696.34
12/5/2025,1875,Electronics,1223.82,2364.76,2878.78,8466.97
12/5/2025,1472,Clothing,4466.53,1328.24,840.05,9604
12/5/2025,1065,Electronics,3840.57,3315.94,2229.43,13155.33
19/5/2025,1011,Electronics,2188.67,2293.17,1929.73,8695.27
19/5/2025,1019,Home,549.64,3073.32,2703.63,8182.34
19/5/2025,1622,Electronics,3903.19,3415.29,1752.05,11861.49
19/5/2025,1153,Electronics,3395.14,3187.48,916.79,10339.51
19/5/2025,1272,Home,3483.69,306.96,788.63,6362.84
26/5/2025,1740,Beauty,4434.18,2187.41,2587.24,12009.2
26/5/2025,1914,Home,1514.47,2900.89,374.25,7748.17
26/5/2025,1210,Clothing,603.2,1491.8,2300.42,5435.21
26/5/2025,1018,Beauty,3629.37,3059.92,3440.5,13827.19
26/5/2025,1480,Clothing,2180.58,752.79,2627.17,7155.22
2/6/2025,1117,Electronics,2171.4,531.41,2844.51,7747.89
2/6/2025,1046,Home,1280.66,2658.95,1837.35,8023.55
2/6/2025,1137,Clothing,3586.6,3967.17,2665.82,13888.1
2/6/2025,1629,Electronics,1243.82,2890.93,2133.17,7495.4
2/6/2025,1831,Electronics,3639.52,3739.58,2782.52,13368.05
9/6/2025,1355,Clothing,2664.08,1910.17,3293.27,9780.81
9/6/2025,1611,Clothing,3340.27,484.42,3182.91,9509.6
9/6/2025,1568,Home,757.76,1133.33,633.45,3443.09
9/6/2025,1905,Beauty,4651.8,2046.62,244.23,10374.55
9/6/2025,1581,Beauty,2170.43,1886.88,2987.81,8947.63
16/6/2025,1532,Beauty,1203.72,3456.43,353.45,7249.61
16/6/2025,1398,Clothing,1918.35,3155.87,485.2,8579.05
16/6/2025,1665,Beauty,3796.02,2328.79,622.36,9339.09
16/6/2025,1020,Clothing,4044.71,2610.37,1831.79,12122.05
16/6/2025,1852,Beauty,1152.34,2983.08,830.93,5968.79
23/6/2025,1628,Beauty,4853.1,977.16,272.72,8340.49
23/6/2025,1572,Clothing,2380.95,628.71,2456.39,7523.63
23/6/2025,1176,Electronics,2317.9,2307.42,2812.61,9869.2
23/6/2025,1021,Beauty,3258.05,1136.74,2994.83,9410.96
23/6/2025,1434,Electronics,1472.41,660.98,3044.56,6612.5
30/6/2025,1610,Beauty,3871.78,1375.17,2797,10800.56
30/6/2025,1276,Beauty,4072.95,2139.85,1321.27,10069.27
30/6/2025,1781,Home,2951.81,3106.17,2688.56,11204.99
30/6/2025,1880,Home,4759.87,2372.67,2677.41,13666.92
30/6/2025,1226,Clothing,2986.31,648.36,1334.09,6542.51
7/7/2025,1426,Home,2173.7,1971.28,1511.36,6878.78
7/7/2025,1296,Beauty,2484.82,1023.75,816.19,5908.76
7/7/2025,1910,Beauty,1743.99,1455.69,2320.83,6928.65
7/7/2025,1847,Home,4536.31,2224.7,1677.09,11094.68
7/7/2025,1485,Electronics,2133.95,3619.59,757.22,8921.43
14/7/2025,1612,Beauty,3191.51,423.52,1400.19,7408.97
14/7/2025,1648,Beauty,4093.67,3880.7,3027.68,15102.47
14/7/2025,1669,Electronics,3828.79,1482.77,3461.77,11296.05
14/7/2025,1570,Home,1953.55,3989.19,3433.28,12215.7
14/7/2025,1226,Electronics,1084.22,2841.39,507,5666.54
21/7/2025,1645,Home,2015.78,1397.01,3230.09,8087.29
21/7/2025,1727,Clothing,3918.12,3955.84,3174.28,14164.94
21/7/2025,1276,Beauty,4186.65,3555.34,1933.94,13764.23
21/7/2025,1942,Clothing,4216.9,672.36,991.58,8193.43
21/7/2025,1661,Electronics,3869.72,2709.25,2249.06,11564.27
28/7/2025,1271,Clothing,4286.51,2043.76,3431.05,12752.06
28/7/2025,1934,Clothing,1962.97,2839.41,429.82,7313.65
28/7/2025,1515,Clothing,4350.82,1140.56,2962.24,11999.29
28/7/2025,1226,Electronics,2806.95,1429.65,901.73,6478.06
28/7/2025,1866,Beauty,1867.76,2716.7,3296.41,10464.36
4/8/2025,1250,Home,3492.43,1263.06,3193.74,10957.7
4/8/2025,1839,Clothing,3021.98,710.66,1675.28,8020.7
4/8/2025,1595,Electronics,2211.8,3914.16,565.38,9024.37
4/8/2025,1082,Clothing,689.11,3037.65,3229.65,9212.92
4/8/2025,1979,Electronics,4598.47,3089.66,2856.19,14123.95
11/8/2025,1942,Beauty,4832.65,2988.68,1205.68,12313.09
11/8/2025,1935,Electronics,1006.61,2903.64,1984.3,8069.84
11/8/2025,1738,Electronics,1588.52,345.89,1746.93,5328.72
11/8/2025,1067,Beauty,1849.64,3049.81,358.87,6361.7
11/8/2025,1880,Beauty,4335.19,2770.88,2157.63,12641.86
18/8/2025,1561,Electronics,656.02,2456.39,3484.94,8274.85
18/8/2025,1200,Electronics,2846.51,535.47,2943.53,8034.44
18/8/2025,1479,Electronics,646.91,3513.64,413.47,6153.45
18/8/2025,1526,Electronics,613.14,2869.47,471.28,5177.65
18/8/2025,1532,Home,2033.67,909.85,762.92,5586.57
25/8/2025,1771,Clothing,2503.61,3218.57,2310.7,10306.65
25/8/2025,1983,Clothing,4040.38,3891.83,1066.83,12594.51
25/8/2025,1721,Electronics,2628.04,3940.21,1087.56,10822.83
25/8/2025,1466,Beauty,678.95,2535.07,3194.35,8039.49
25/8/2025,1431,Electronics,2613.39,1966.72,1760.81,8308.49
1/9/2025,1048,Clothing,2634.3,2486.21,1972.12,10689.19
1/9/2025,1756,Electronics,722.74,3383.27,2984.95,8909.36
1/9/2025,1232,Electronics,1483.29,1768.15,346.48,4042.18
1/9/2025,1182,Electronics,4175.07,1352.88,2900.08,12020.56
1/9/2025,1906,Electronics,633.51,2799.05,348.02,5049.84
8/9/2025,1956,Electronics,4682.42,911.25,1486.58,10281.74
8/9/2025,1601,Home,1661.78,2796.05,1179.2,7716.71
8/9/2025,1323,Clothing,4793.15,2969.16,3047.51,14518.15
8/9/2025,1750,Beauty,4101.6,2751.86,504.54,10623.16
8/9/2025,1769,Home,2973.13,3631.19,1533.99,11588.27
15/9/2025,1972,Clothing,3407.43,2691.87,3130.53,11961.01
15/9/2025,1453,Home,2759.69,813.82,581.93,5745.19
15/9/2025,1349,Beauty,3834.42,2260.97,2935.76,12055.13
15/9/2025,1974,Clothing,4190.06,2871.06,1957.23,11948.17
15/9/2025,1920,Home,4595.61,3289.68,912.97,12214.12
22/9/2025,1428,Home,4175.25,549.32,1478.4,8603.08
22/9/2025,1363,Clothing,2941.4,3488.45,2147.94,12066.42
22/9/2025,1074,Beauty,2229.08,1965.46,2111.58,8253.1
22/9/2025,1379,Electronics,555.39,1669.19,290.49,3467.68
22/9/2025,1832,Electronics,2866.16,2193.22,746.57,7844.2
29/9/2025,1913,Beauty,4759.9,1557.14,1906.89,11517.11
29/9/2025,1575,Electronics,530.2,1727.74,2812.68,5983.82
29/9/2025,1485,Electronics,2889.88,3414.38,2114.73,11415.37
29/9/2025,1665,Home,973.67,919.23,3407.68,6187.35
29/9/2025,1562,Clothing,1439.89,3724.86,1566.7,8931.03
6/10/2025,1548,Clothing,1909.43,844.13,1809.91,6156.46
6/10/2025,1785,Beauty,2186.62,2053.82,1665.62,7923.21
6/10/2025,1908,Clothing,2954.22,626.35,2202.7,7879.67
6/10/2025,1898,Clothing,1257.52,1213.56,2967.21,7223.18
6/10/2025,1532,Beauty,4220.67,2923.35,784.66,12089.6
13/10/2025,1958,Beauty,4238.1,2315.86,745.3,10304.8
13/10/2025,1098,Beauty,4998.71,856.93,2895.3,12401.96
13/10/2025,1786,Electronics,1852.2,1505.55,2815.32,7786.86
13/10/2025,1063,Home,3788.68,847.54,3013.44,10662.83
13/10/2025,1139,Clothing,4592.55,2971.93,1960.18,13240.26
20/10/2025,1557,Clothing,1087.94,3224.89,611.91,6225.82
20/10/2025,1303,Beauty,4748.19,850.55,1726.83,10736.78
20/10/2025,1416,Clothing,2675.33,3495.13,2142.81,10404.01
20/10/2025,1760,Home,3928.09,1730.2,1887.21,9984.68
20/10/2025,1750,Beauty,3097.76,3502.64,3436.44,13920.78
27/10/2025,1839,Beauty,4129.65,2062.99,3487.21,12892.57
27/10/2025,1322,Home,514.26,2668.73,1184.84,5271.41
27/10/2025,1719,Clothing,638.56,3894.67,928.46,6595.22
27/10/2025,1125,Electronics,2645.8,2844.57,2584.93,10382.12
27/10/2025,1695,Clothing,2114.22,3276.42,1119.91,8978.09
3/11/2025,1326,Clothing,4808.52,332.79,3492.81,12313.34
3/11/2025,1000,Home,564.42,3030.12,2952.68,8585.54
3/11/2025,1675,Home,1142.83,3087.68,2737.45,9607.63
3/11/2025,1941,Clothing,3261.22,627.84,1809.17,8233.18
3/11/2025,1187,Electronics,2333.94,1806.29,417.83,5954.27
10/11/2025,1122,Clothing,4764.6,567.57,3351.88,11753.02
10/11/2025,1187,Clothing,1848.05,584.39,1852.06,6083.88
10/11/2025,1507,Clothing,828.06,1790.63,1174.46,5448.05
10/11/2025,1732,Home,1764.52,3272.89,3266.45,11236.01
10/11/2025,1436,Electronics,2643.97,1136.31,2313.57,8122.53
17/11/2025,1009,Beauty,3215.72,1623.91,2337.8,8230.27
17/11/2025,1013,Clothing,3311.53,565.57,2452.64,8737.63
17/11/2025,1204,Home,3712.79,3343.38,2853.06,13869.36
17/11/2025,1973,Clothing,3989.06,1304.21,1839.09,11265.8
17/11/2025,1990,Electronics,1102.23,2629.36,379.3,5115.46
24/11/2025,1293,Electronics,2800.08,473.35,1111.36,6709.62
24/11/2025,1156,Beauty,782.1,3405.55,212.61,6268.05
24/11/2025,1922,Clothing,958.52,1632.87,1092.3,4437.71
24/11/2025,1623,Beauty,1910.07,3219.05,3143.18,11247.13
24/11/2025,1217,Electronics,4838.35,630.37,2467.16,10968.91
1/12/2025,1928,Beauty,2244.42,2641.03,2522.73,10309.94
1/12/2025,1605,Electronics,3924.67,3817.86,2518.38,14490.34
1/12/2025,1303,Home,974.14,3192.75,2325.69,8332.89
1/12/2025,1657,Beauty,2751.81,1901.1,1710.41,8426.54
1/12/2025,1396,Beauty,2159.18,1668.31,372.77,5628.27
8/12/2025,1458,Clothing,4326.57,1084.38,2369.27,11468.91
8/12/2025,1727,Beauty,4460.71,1098.24,2436.78,10266.03
8/12/2025,1939,Electronics,3432.38,3032.81,1241.57,10990.05
8/12/2025,1053,Clothing,3569.87,2032.17,2111.28,11031.2
8/12/2025,1524,Electronics,2947.13,669.51,2470.55,8514.32
15/12/2025,1779,Clothing,4599.81,3401.58,2540.96,13941.38
15/12/2025,1377,Beauty,1700.04,2981.97,3096.24,10739.78
15/12/2025,1603,Clothing,2289.41,3120.54,1701.23,9099.36
15/12/2025,1174,Clothing,4357.08,2428.81,452.07,9875.94
15/12/2025,1054,Clothing,2154.8,972.44,1637.18,5857.85
22/12/2025,1719,Home,1504.69,2674.88,2083.14,7594.01
22/12/2025,1775,Electronics,3483.53,2329.01,3101.54,12234.79
22/12/2025,1058,Beauty,3227.66,1377.23,883.41,8663.78
22/12/2025,1123,Home,4547.55,2927.5,3293.35,14269.13
22/12/2025,1630,Electronics,3522.39,2454.17,3467.69,13838.92"""

# Show dataset information
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📊 **{demo_csv_data.count(chr(10))}** rows of data")
with col2:
    st.info("📅 **2 Years** of campaign data (2024-2025)")
with col3:
    st.info("🏷️ **4 Categories**: Home, Electronics, Clothing, Beauty")

# Download button
st.download_button(
    label="⬇️ Download Demo Dataset (CSV)",
    data=demo_csv_data,
    file_name="adventa_demo_data.csv",
    mime="text/csv",
    use_container_width=True,
    type="secondary"
)

# Show preview
with st.expander("👀 Preview Demo Data", expanded=False):
    # Load the data into a dataframe for preview
    demo_df = pd.read_csv(pd.compat.StringIO(demo_csv_data))
    st.dataframe(demo_df.head(10), use_container_width=True)
    
    # Show summary statistics
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Total Revenue", f"${demo_df['total_revenue'].sum():,.2f}")
    with col_sum2:
        total_ad_spend = demo_df[['fb_spend','instagram_spend','tiktok_spend']].sum().sum()
        st.metric("Total Ad Spend", f"${total_ad_spend:,.2f}")
    with col_sum3:
        roi = ((demo_df['total_revenue'].sum() - total_ad_spend) / total_ad_spend * 100)
        st.metric("Average ROI", f"{roi:.1f}%")
    
    st.caption("💡 **Tip:** Download this file and upload it using the button below to see Adventa in action!")

st.markdown("---")
st.markdown("### 📤 Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=["csv"],
    help="Upload CSV with columns: date, category, fb_spend, instagram_spend, tiktok_spend, total_revenue"
)

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

# ========== PREDICT SECTION ==========
st.markdown('<div id="predict-section"></div>', unsafe_allow_html=True)
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
                
                # Investment Allocation Recommendation
                st.markdown("---")
                st.markdown("### 📊 Optimal Investment Allocation")
                
                model_coef = st.session_state["trained_model"].coef_
                feature_names = st.session_state["feature_cols"]
                
                channel_coefs = {}
                for name, coef in zip(feature_names, model_coef):
                    if 'fb_spend' in name or 'fb_adstock' in name:
                        channel_coefs['Facebook'] = channel_coefs.get('Facebook', 0) + abs(coef)
                    elif 'instagram_spend' in name or 'insta_adstock' in name:
                        channel_coefs['Instagram'] = channel_coefs.get('Instagram', 0) + abs(coef)
                    elif 'tiktok_spend' in name or 'tiktok_adstock' in name:
                        channel_coefs['TikTok'] = channel_coefs.get('TikTok', 0) + abs(coef)
                
                if channel_coefs:
                    total_impact = sum(channel_coefs.values())
                    if total_impact > 0:
                        fb_pct = (channel_coefs.get('Facebook', 0) / total_impact) * 100
                        insta_pct = (channel_coefs.get('Instagram', 0) / total_impact) * 100
                        tiktok_pct = (channel_coefs.get('TikTok', 0) / total_impact) * 100
                        
                        col_rec1, col_rec2, col_rec3 = st.columns(3)
                        with col_rec1:
                            st.metric("🎯 Facebook", f"{fb_pct:.1f}%")
                        with col_rec2:
                            st.metric("🎯 Instagram", f"{insta_pct:.1f}%")
                        with col_rec3:
                            st.metric("🎯 TikTok", f"{tiktok_pct:.1f}%")
                        
                        st.caption(f"Based on your total spend of **${total_ad_spend:,.2f}**, the optimal allocation would be:")
                        col_dol1, col_dol2, col_dol3 = st.columns(3)
                        with col_dol1:
                            st.info(f"💰 **Facebook:** ${(fb_pct/100) * total_ad_spend:,.2f}")
                        with col_dol2:
                            st.info(f"💰 **Instagram:** ${(insta_pct/100) * total_ad_spend:,.2f}")
                        with col_dol3:
                            st.info(f"💰 **TikTok:** ${(tiktok_pct/100) * total_ad_spend:,.2f}")
                        
                        st.markdown("**Model Interpretation:**")
                        st.markdown("- Lasso Regression automatically selects important features")
                        st.markdown("- Adstock captures delayed/recurring effects of ad spend")
                        st.markdown("- Category dummies account for campaign type differences")

# ========== ANALYZE SECTION ==========
st.markdown('<div id="analytics-section"></div>', unsafe_allow_html=True)
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
    <p>🚀 Adventa - Advertisement Campaign Spend Optimizer</p>
    <p style='font-size: 12px;'>Powered by Lasso Regression & Adstock Transformation</p>
</div>
""", unsafe_allow_html=True)
