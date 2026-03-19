import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.graph_objects as go
import kagglehub

st.set_page_config(page_title="Ferrari Stock Price Predictor", layout="wide", page_icon="🏎️")

st.title("🏎️ Ferrari Stock Price Predictor")
st.markdown("Linear Regression model trained on Ferrari historical stock data (2015–2026)")

# ─────────────────────────────────────────────
# Feature engineering (mirrors the notebook)
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)

    return df.dropna().reset_index(drop=True)

FEATURE_COLS = ['MA10', 'MA20', 'EMA10', 'EMA20', 'RSI', 'Lag1', 'Lag2', 'Lag3']

# ─────────────────────────────────────────────
# Load or train model
# ─────────────────────────────────────────────
@st.cache_resource
def get_model(df_raw: pd.DataFrame):
    df = engineer_features(df_raw)
    X = df[FEATURE_COLS]
    y = df['Close']
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model_path = "lr_pipeline.pkl"
    if os.path.exists(model_path):
        pipeline = pickle.load(open(model_path, "rb"))
    else:
        pipeline = Pipeline([
            ('scaler', MinMaxScaler()),
            ('model', LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
        pickle.dump(pipeline, open(model_path, "wb"))

    return pipeline, df, X_train, X_test, y_train, y_test, split

# ─────────────────────────────────────────────
# Sidebar – data source
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data Source")
    data_source = st.radio("Choose data source", ["Upload CSV", "Download from Kaggle"])

    df_raw = None

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload Ferrari History Stock Data.csv", type="csv")
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df_raw):,} rows")
    else:
        if st.button("Download Dataset"):
            with st.spinner("Downloading from Kaggle…"):
                try:
                    import kagglehub
                    path = kagglehub.dataset_download("alehcleal/ferrari-stock-data-2015-2026")
                    csv_path = os.path.join(path, "Ferrari History Stock Data.csv")
                    df_raw = pd.read_csv(csv_path)
                    st.session_state["df_raw"] = df_raw
                    st.success(f"Loaded {len(df_raw):,} rows")
                except Exception as e:
                    st.error(f"Download failed: {e}")

        if "df_raw" in st.session_state:
            df_raw = st.session_state["df_raw"]

# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────
if df_raw is None:
    st.info("👈 Please upload the CSV or download the dataset from the sidebar to get started.")
    st.stop()

pipeline, df, X_train, X_test, y_train, y_test, split = get_model(df_raw)

y_pred_test = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

# ─── Metrics ───────────────────────────────
st.subheader("📊 Model Performance")
c1, c2, c3 = st.columns(3)
c1.metric("R² Score", f"{r2:.4f}")
c2.metric("MSE", f"{mse:.3f}")
c3.metric("RMSE", f"{rmse:.3f}")

# ─── Actual vs Predicted ───────────────────
st.subheader("📈 Actual vs Predicted (Test Set)")
test_dates = df['Date'].iloc[split:].reset_index(drop=True)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=test_dates, y=y_test.reset_index(drop=True),
                           mode='lines', name='Actual', line=dict(color='#1f77b4')))
fig1.add_trace(go.Scatter(x=test_dates, y=y_pred_test,
                           mode='lines', name='Predicted',
                           line=dict(color='#ff7f0e', dash='dash')))
fig1.update_layout(xaxis_title="Date", yaxis_title="Close Price (USD)",
                   legend=dict(x=0, y=1), height=420, margin=dict(t=10))
st.plotly_chart(fig1, width='stretch')

# ─── Full price history ────────────────────
st.subheader("📉 Full Price History")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['Date'].iloc[:split], y=df['Close'].iloc[:split],
                           mode='lines', name='Train', line=dict(color='steelblue')))
fig2.add_trace(go.Scatter(x=df['Date'].iloc[split:], y=df['Close'].iloc[split:],
                           mode='lines', name='Test', line=dict(color='orange')))
split_date = str(df['Date'].iloc[split].date())
fig2.add_trace(go.Scatter(
    x=[split_date, split_date],
    y=[df['Close'].min(), df['Close'].max()],
    mode='lines', name='Train/Test Split',
    line=dict(color='red', dash='dash', width=1.5)
))
fig2.update_layout(xaxis_title="Date", yaxis_title="Close Price (USD)", height=380, margin=dict(t=10))
st.plotly_chart(fig2, width='stretch')

# ─── Single-day prediction ─────────────────
st.subheader("🔮 Predict Next Close Price")
latest = df.iloc[-1]

with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        ma10  = st.number_input("MA10",  value=float(latest['MA10']),  format="%.4f")
        ma20  = st.number_input("MA20",  value=float(latest['MA20']),  format="%.4f")
        ema10 = st.number_input("EMA10", value=float(latest['EMA10']), format="%.4f")
        ema20 = st.number_input("EMA20", value=float(latest['EMA20']), format="%.4f")
    with col2:
        rsi  = st.number_input("RSI",  value=float(latest['RSI']),  min_value=0.0, max_value=100.0, format="%.4f")
        lag1 = st.number_input("Lag1 (yesterday's Close)",      value=float(latest['Lag1']), format="%.4f")
        lag2 = st.number_input("Lag2 (2 days ago Close)",       value=float(latest['Lag2']), format="%.4f")
        lag3 = st.number_input("Lag3 (3 days ago Close)",       value=float(latest['Lag3']), format="%.4f")

    submitted = st.form_submit_button("Predict", width='stretch')

if submitted:
    input_df = pd.DataFrame([[ma10, ma20, ema10, ema20, rsi, lag1, lag2, lag3]],
                             columns=FEATURE_COLS)
    prediction = pipeline.predict(input_df)[0]
    st.success(f"### Predicted Close Price: **${prediction:.2f} USD**")