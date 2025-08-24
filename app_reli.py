import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import os

# config
st.set_page_config(
    page_title="RELI Stock Price Predictor",
    layout="wide",
)

# title and intro
st.title('RELI Stock Price Predictor')
st.markdown("""
This application forecasts the stock price of Reliance (RELI) for August 2025. 
It uses historical data(past 10 years) and a LightGBM machine learning model for the prediction.
""")
st.markdown("---")

# sidebar
st.sidebar.header("About the Project")
st.sidebar.info("""
This app is an example of time series forecasting. 
The model is trained on historical data(past 10 years) and uses moving averages (5 and 10 days) as features.
""")

#Data Loading and Caching
@st.cache_data # basically creates a cache for data so it does not have to re-read and process the csv file

# loading data and preprocessing
def load_data(file_path):
    if not os.path.exists(file_path): # to check if file path exists on computer
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it is in the same directory as the app.")
        st.stop()
        
    df = pd.read_csv(file_path)
    df = df.rename(columns=lambda x: x.strip())  # remove extra spaces
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')  # converts date column from string format to datetime
    df = df.dropna(subset=['Date'])  # remove invalid dates
    df['Close'] = df['Price'].replace(',', '', regex=True).astype(float)  # remove commas
    df = df.sort_values('Date').reset_index(drop=True)
    df['MA_5'] = df['Close'].rolling(window=5).mean()  # calculate 5 day moving average
    df['MA_10'] = df['Close'].rolling(window=10).mean() # calculate 10 day moving average
    df.dropna(inplace=True)  # remove rows with NaN created after moving average calculation
    return df

# directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'RELI Historical Data.csv'
file_path = os.path.join(script_dir, file_name)

# Load data
df = load_data(file_path)

# Model Training
@st.cache_resource  # cache for trained model as models take up considerable amount of compuational resources to train
def train_model(data_frame):
    """Trains the LightGBM model and returns predictions."""
    split_date = data_frame['Date'].max() - pd.DateOffset(months=6)
    train_df = data_frame[data_frame['Date'] <= split_date]
    test_df = data_frame[data_frame['Date'] > split_date]

    features = ['MA_5', 'MA_10']
    target = 'Close'

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # LightGBM Model
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_test)
    
    # Linear Regression Model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    
    return lgb_model, y_test, lgb_preds, lr_preds

# Call the training function
lgb_model, y_test, lgb_preds, lr_preds = train_model(df)

# Performance Metrics 
st.header("Model Performance")
st.write("Comparing LightGBM and Linear Regression on the test set.")

col1, col2 = st.columns(2) # 2 cols with same width

with col1:
    st.subheader("LightGBM")
    st.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, lgb_preds):.2f}")
    st.metric("R² Score", f"{r2_score(y_test, lgb_preds):.2f}")

with col2:
    st.subheader("Linear Regression")
    st.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, lr_preds):.2f}")
    st.metric("R² Score", f"{r2_score(y_test, lr_preds):.2f}")

# Actual vs. Predicted Chart (Plotly)
st.markdown("---")
st.header("Actual vs. Predicted Prices on Test Data")
fig_preds = go.Figure()
fig_preds.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name='Actual', line=dict(color='white')))
fig_preds.add_trace(go.Scatter(x=y_test.index, y=lgb_preds, mode='lines', name='LightGBM Predicted', line=dict(dash='dash')))
fig_preds.add_trace(go.Scatter(x=y_test.index, y=lr_preds, mode='lines', name='LR Predicted', line=dict(dash='dot')))
fig_preds.update_layout(
    title="Comparison of Model Predictions",
    xaxis_title="Date Index (Last 6 Months)",
    yaxis_title="Price (INR)",
    hovermode='x unified',
)
st.plotly_chart(fig_preds, use_container_width=True)


# Forecast for August 2025
st.markdown("---")
st.header("Forecast for August 2025")

@st.cache_data # cache output of generate_forecast function
def generate_forecast(_model, data):
    forecast_days = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=31)
    forecast_df = pd.DataFrame({'Date': forecast_days})
    last_data = data[['Close']].copy()
    
    for i in range(31):
        ma_5 = last_data['Close'].iloc[-5:].mean()
        ma_10 = last_data['Close'].iloc[-10:].mean()
        features_today = pd.DataFrame({'MA_5': [ma_5], 'MA_10': [ma_10]})
        next_pred = _model.predict(features_today)[0]
        last_data = pd.concat([last_data, pd.DataFrame({'Close': [next_pred]})], ignore_index=True)
    
    forecast_df['Predicted Close'] = last_data['Close'].iloc[-31:].values
    return forecast_df

forecast_df = generate_forecast(lgb_model, df)

# Display forecast chart
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(
    x=forecast_df['Date'],
    y=forecast_df['Predicted Close'],
    mode='lines+markers',
    name='Predicted Close',
    line=dict(color='#FFD700', width=2),
    marker=dict(size=6)
))
fig_forecast.update_layout(
    title="Forecasted Prices for August 2025",
    xaxis_title="Date",
    yaxis_title="Predicted Price (INR)",
    hovermode='x unified'
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Display forecast table in an expander
with st.expander("Show Raw Forecast Data"):
    st.dataframe(forecast_df.set_index('Date'))

st.sidebar.markdown("---")