# stock_price_prediction_system
Stock Price Prediction System that include PSX market companies share price prediction
This repository contains a Stock Price Prediction System web application using Streamlit. 
The system allows users to visualize and predict stock market prices for various companies. 
It uses historical stock price data and a trained deep learning model to make predictions.

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [How it Works](#how-it-works)
- [Suggestion and Metrics](#suggestion-and-metrics)
- [Company List](#company-list)

## Features

- Select a company from a predefined list to predict its stock prices.
- Choose a date range to view and predict historical stock prices.
- Visualize historical closing stock prices.
- Predict future stock prices using a deep learning model.
- Display investment suggestions based on recent and long-term trends.
- Calculate Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics for the predictions.

## Usage

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/yourusername/stock-price-prediction.git
   ```

2. Change to the project directory:

   ```
   cd stock-price-prediction
   ```

3. Install the required dependencies (see the Dependencies section).

4. Run the Streamlit app:

   ```
   streamlit run stock_price_prediction.py
   ```

5. Access the web app in your browser and follow the instructions to select a company, choose a date range, and view predictions.

## Dependencies

The following Python libraries are required to run this project:

- Streamlit
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Keras
- PIL (Python Imaging Library)
- yfinance

You can install these dependencies using pip:

```
pip install streamlit numpy pandas matplotlib scikit-learn keras pillow yfinance
```

## Setup

1. Create a Python environment (recommended) and install the required dependencies.

2. Clone this repository and navigate to the project directory.

3. Run the Streamlit app as described in the Usage section.

## How it Works

1. The web app allows you to select a company from a predefined list.

2. You can choose a start and end date to specify the date range for historical stock price data.

3. The app fetches historical stock price data using the yfinance library.

4. It visualizes the closing stock price history for the selected company.

5. It uses a deep learning model to predict future stock prices.

## Suggestion and Metrics

The app provides investment suggestions based on recent and long-term trends in stock prices:

- ✅ Invest: Positive recent trends and a positive long-term trend.
- ⚠️ Consider Investing: Positive long-term trend, but caution required for recent trends.
- ❌ Do Not Invest: Negative recent trends or negative long-term trend.

It also calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics to evaluate the quality of the predictions.

## Company List

The following companies are available for prediction:

- ATLAS HONDA
- HONDA ATLAS CARS
- PAK SUZUKI
- ASKARI BANK
- MEEZAN BANK
- BANK AL-FALAH
- PSO
- SHELL PAKISTAN
- SUI GAS
- PIA AIRLINE

You can add more companies and their symbols to the `stock_data` dictionary in the code as needed.
