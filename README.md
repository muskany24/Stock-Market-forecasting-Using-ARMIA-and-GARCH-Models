# Stock-Market-forecasting-Using-ARMIA-and-GARCH-Models
Stock Forecasting Using ARIMA and GARCH Models

This project analyzes and forecasts the closing prices and volatility of major Indian bank stocks using time series modeling techniques. It applies ARIMA models to capture linear trends and structure in stock prices and selectively uses GARCH models to capture volatility clustering, where statistically significant.

**Key steps include:**

Data cleaning and log return computation

ADF test and differencing to achieve stationarity

ACF/PACF-based parameter selection for ARIMA and GARCH

Residual diagnostics and Ljung-Box test

Rolling forecasts and conditional volatility analysis

Visualization of actual vs forecasted prices for 90-day horizons

GARCH models were applied to INDUSINDBK and BAJAJFINSV, where volatility behavior was prominent and model parameters were statistically significant.
