# Task 2: Predict Future Stock Prices (Short-Term)

## Task Objective
The goal of this project is to build a machine learning model capable of predicting the next day's closing stock price using historical market data.

## Dataset Used
* **Name:** Historical Stock Market Data (e.g., AAPL, GOOGL, TSLA)
* **Source:** Retrieved dynamically via the `yfinance` Python library.

## Models and Techniques Applied
* **Features Used:** Open, High, Low, and Volume.
* **Models:** Regression Modeling (Linear Regression / Random Forest).
* **Techniques:** Time-series data handling and API data fetching.

## Key Results and Findings
The regression model was successfully trained on historical data. The evaluation includes visualizations plotting the actual closing prices against the model's predicted closing prices, demonstrating the model's capability to capture short-term pricing trends in volatile markets.
