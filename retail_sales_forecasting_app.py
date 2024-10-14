
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Retail Sales Forecasting App")

# File uploader
uploaded_file = st.file_uploader("Upload your sales data CSV", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    
    # Data cleaning and preprocessing
    st.write("### Data Preview")
    st.dataframe(data.head())
    
    # Rename columns
    data.columns = ['date', 'Sales', 'Stock', 'Price']
    data['date'] = pd.to_datetime(data['date'])
    
    # Set date as index
    data.set_index('date', inplace=True)
    
    # Visualization: Sales over time
    st.write("### Sales Over Time")
    st.line_chart(data['Sales'])
    
    # Visualization: Seasonal Decomposition
    st.write("### Seasonal Decomposition")
    decomposition = seasonal_decompose(data['Sales'], model='multiplicative', period=12)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1, title='Observed')
    decomposition.trend.plot(ax=ax2, title='Trend')
    decomposition.seasonal.plot(ax=ax3, title='Seasonal')
    decomposition.resid.plot(ax=ax4, title='Residual')
    plt.tight_layout()
    st.pyplot(fig)

    # ACF and PACF plots
    st.write("### Autocorrelation and Partial Autocorrelation Plots")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(data['Sales'], ax=ax1)
    plot_pacf(data['Sales'], ax=ax2)
    st.pyplot(fig)
    
    # Forecasting with ARIMA
    st.write("### Forecasting using ARIMA")
    p, d, q = 1, 1, 1  # Hyperparameters for ARIMA model
    arima_model = ARIMA(data['Sales'], order=(p, d, q))
    arima_result = arima_model.fit()
    forecast = arima_result.forecast(steps=12)
    
    st.write("### Forecast for Next 12 Months")
    st.line_chart(forecast)
    
    # Model performance metrics
    st.write("### Model Summary")
    st.text(arima_result.summary())
