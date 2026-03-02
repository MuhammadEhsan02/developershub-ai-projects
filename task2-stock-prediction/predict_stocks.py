# 🌸 Task 2: Predict Future Stock Prices

## 📌 Objective
# The goal of this task is to predict the next day’s closing price of a stock using historical data.
# A Linear Regression model is applied to learn patterns from previous stock prices.


# Import required libraries
# sys library is used to read command line arguments
# This allows user to select stock symbol from terminal (e.g., TSLA, AAPL)
import sys

# yfinance library is used to download historical stock data from Yahoo Finance
import yfinance as yf

# pandas library is used for data handling and analysis
import pandas as pd

# matplotlib is used for creating graphs and visualization
import matplotlib.pyplot as plt

# train_test_split is used to split data into training and testing sets
from sklearn.model_selection import train_test_split

# LinearRegression is a machine learning model used for prediction
from sklearn.linear_model import LinearRegression

# mean_absolute_error is used to evaluate prediction accuracy
from sklearn.metrics import mean_absolute_error


# --------------------------
# CLI Stock Symbol Selection
# --------------------------
# Use command line argument or default to Apple

# Check if user provided stock symbol via command line
if len(sys.argv) > 1:
    # Take stock symbol from user and convert to uppercase
    stock_symbol = sys.argv[1].upper()
else:
    # Default stock symbol is Apple if none provided
    stock_symbol = "AAPL"

# Print selected stock symbol
print(f"Selected Stock: {stock_symbol}")
# Purpose: User terminal se stock select kar sakta hai(Example:python script.py TSLA)

# --------------------------
# Load Historical Stock Data
# --------------------------
# Download stock data from Yahoo Finance
# Date range: 2020 to 2024
df = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")

# Display first 5 rows
print(df.head())


# -----------------------------
# # Exploratory Data Analysis (EDA)
# -----------------------------
# Plot Closing Price Trend

# Create figure for closing price plot
plt.figure(figsize=(10,4))

# Plot closing price over time
plt.plot(df["Close"], label="Closing Price")

# Add title and labels
plt.title(f"{stock_symbol} Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")

# Show legend
plt.legend()

# Save graph
plt.savefig(f"outputs/{stock_symbol}_close_trend.png")

# Display graph
plt.show()


# Closing price shows an overall trend with fluctuations.
# Visual inspection helps understand stock behavior over time.


# -----------------------------
# Plot Volume Trend
# -----------------------------

# Create figure for volume plot
plt.figure(figsize=(10,4))

# Plot trading volume
plt.plot(df["Volume"], label="Trading Volume", color="orange")

# Add title and labels
plt.title(f"{stock_symbol} Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")

# Show legend
plt.legend()

# Save graph
plt.savefig(f"outputs/{stock_symbol}_volume_trend.png")

# Display graph
plt.show()


# Volume spikes often correspond to large price movements.
# Volume is an important feature for predicting price changes.


# -----------------------------------------
# Prepare Target Variable (Next Day Close)
# -----------------------------------------
# Predict next day's closing price

# Create target column for next day's closing price
df["Target_Close"] = df["Close"].shift(-1)

# Remove last row (contains NaN)
df.dropna(inplace=True)
# Purpose:

# Model will predict next day's closing price

# Example:

# Close Today	Target_Close (Next Day)
# 150	152
# 152	148

# --------------------------
# Feature Selection
# --------------------------
# Select input features
X = df[["Open", "High", "Low", "Volume"]]

# Select target variable
y = df["Target_Close"]

# shift(-1) creates the "next day Close" target. Dataset is now ready for ML model.


# --------------------------
# Train-Test Split
# --------------------------
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False  # Important for time series data
)
# Why shuffle=False?
# Time series data me order important hota hai:
# Correct:
# 2020 → 2021 → 2022 → 2023

# Wrong:
# 2022 → 2020 → 2023 → 2021

# Time series data should not be shuffled to preserve temporal order.


# --------------------------
# Train Linear Regression Model
# --------------------------
# Create Linear Regression model
model = LinearRegression()

# Train model using training data
model.fit(X_train, y_train)
# Purpose:
# Model learns relationship between:
# Inputs:
# Open
# High
# Low
# Volume
# Output:
# Next day Close price
# Linear Regression models relationship between features and next-day close price.


# --------------------------
# Prediction & Evaluation
# --------------------------
# Predict closing price using test data
predictions = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)

# Print error
print("Mean Absolute Error:", mae)

# MAE indicates average error of prediction. Smaller MAE = better accuracy.


# --------------------------
# Plot Visualize Actual vs Predicted
# --------------------------
# Create figure
plt.figure(figsize=(10,5))

# Plot actual price
plt.plot(y_test.values, label="Actual Closing Price")

# Plot predicted price
plt.plot(predictions, label="Predicted Closing Price")

# Add title and labels
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")

# Show legend
plt.legend()

# Save plot
plt.savefig(f"outputs/{stock_symbol}_stock_prediction.png")

# Display plot
plt.show()


# Model captures general trend; minor deviations during high volatility.
# Linear Regression provides baseline prediction.

