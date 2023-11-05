import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from PIL import Image
import yfinance as yf
import math
import datetime

# Load the image at the beginning
# image = Image.open("C:/Users/M Saad/Desktop/Python_ws/IMG_2007.png")

# Streamlit app 
st.write('''
    <div style="text-align: center;">
        <h1>Stock Price Prediction System Web Application</h1>
        <p>Visualize and Predict The Stock Market Price</p>
    </div>
    ''', unsafe_allow_html=True)

# Edit the size of the image
# st.image(image, use_column_width=True, width=200)

st.sidebar.header('User Section')   
def main():
    st.sidebar.header('')

def valDate(date_string):
    try:
        dateObject = datetime.datetime.strptime(date_string, '%d-%m-%Y')
        return True
    except ValueError:
        return False

# Dictionary of company names and their symbols
stock_data = {
    "ATLAS HONDA": "ATLH.KA",
    "HONDA ATLAS CARS": "HCAR.KA",
    "PAK SUZUKI": "PSMC.KA",
    "ASKARI BANK": "AKBL.KA",
    "MEEZAN BANK": "MEBL.KA",
    "BANK AL-FALAH": "BAFL.KA",
    "PSO": "PSO.KA",
    "SHELL PAKISTAN": "SHEL.KA",
    "SUI GAS": "SNGP.KA",
    "PIA AIRLINE": "PIAA.KA"
    # Add more companies and symbols as needed
}

# Dropdown to select company
selected_company = st.sidebar.selectbox("Select Company", list(stock_data.keys()), 0)

# Get stock symbol corresponding to selected company
stock_symbol = stock_data[selected_company]
start_date_slot = st.sidebar.empty()
end_date_slot = st.sidebar.empty()

# Set date range restrictions
min_date = datetime.datetime(2012, 1, 1)
max_date = datetime.datetime(2023, 9, 30)
start_date_input = st.sidebar.date_input("Select Start Date", min_value=min_date, max_value=max_date, value=min_date)
end_date_input = st.sidebar.date_input("Select End Date", min_value=min_date, max_value=max_date, value=max_date)
confirm_button = st.sidebar.button("Get Prediction")

if confirm_button:
    if not valDate(start_date_input.strftime('%d-%m-%Y')):
        start_date_slot.markdown(unsafe_allow_html=True)
        st.stop()
    if not valDate(end_date_input.strftime('%d-%m-%Y')):
        end_date_slot.markdown(unsafe_allow_html=True)
        st.stop()

    # Convert input dates to datetime objects
    start_date = start_date_input
    end_date = end_date_input

    # Validate date range
    if start_date > end_date:
        start_date_slot.markdown(unsafe_allow_html=True)
        st.stop()

    # Download stock data using yfinance
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Display stock data
    st.subheader('Stock Data')
    st.write(df)

    # Plot closing price history
    plt.figure(figsize=(24, 12))
    plt.plot(df['Close'])
    plt.fill_between(df.index, 0, df['Close'], color='skyblue', alpha=0.3) # Add fill_between
    plt.title('Close Price History')
    plt.xlabel('Years', fontsize=20)
    plt.ylabel('Close Stock Market Price ', fontsize=20)
    st.subheader('Closing Price History')
    st.pyplot(plt)

    # Create a new DataFrame with only the 'Close' column
    data = df.filter(['Close'])

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Get the training data length
    training_data_len = math.ceil(len(scaled_data) * 0.8)

    # Create the training data set
    train_data = scaled_data[0:training_data_len, :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #Load My Model
    model = load_model('C:\\Users\\M. Asif\\Desktop\\Project Code\\asif_keras_model.h5')

    # Test the model
    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Create DataFrame for predictions
    valid = pd.DataFrame()
    valid['Close'] = data[training_data_len:]['Close']
    valid['Predictions'] = predictions
    

    # Plot training and validation
    plt.figure(figsize=(24, 12))
    # Plot actual close price
    plt.plot(data['Close'], label='Actual Close Price')
    # Plot predictions
    plt.plot(valid[['Close', 'Predictions']])
    # Add fill_between for actual close price
    plt.fill_between(data.index, data['Close'], color='red', alpha=0.3, label='Testing Data')
    # Add fill_between for predictions
    plt.fill_between(valid.index, valid['Predictions'], color='yellow', alpha=0.4, label='Predictions')
    plt.title('Model Training & Validation')
    plt.xlabel('Years', fontsize = 20)
    plt.ylabel('Close Stock Market Price', fontsize = 20)
    plt.legend(loc='upper left')
    st.subheader('Model Training & Validation')
    st.pyplot(plt)


    # Plot predictions
    plt.figure(figsize=(24, 12))
    # Plot predictions
    plt.plot(valid[['Predictions']], color='orange')
    # Add fill_between for predictions
    plt.fill_between(valid.index, valid['Predictions'], color='orange', alpha=0.3, label='Predictions')
    plt.title('Predictions')
    plt.xlabel('Years', fontsize = 20)
    plt.ylabel('Close Stock Prediction Price', fontsize = 20)
    plt.legend(loc='upper left')
    st.subheader('Predictions')
    st.pyplot(plt)

    
    # Display predicted prices
    st.subheader('Actual vs Predicted Prices')
    st.write(valid)

     # Calculate metrics
    mae = np.mean(np.abs(valid['Predictions'] - valid['Close']))
    rmse = np.sqrt(np.mean((valid['Predictions'] - valid['Close'])**2))
    # Display metrics
    st.subheader('Metrics')
    st.markdown(f'<p style="font-size:20px;color:red;">Mean Absolute Error (MAE): <strong>{mae:.2f}</strong></p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:20px;color:red;">Root Mean Squared Error (RMSE): <strong>{rmse:.2f}</strong></p>', unsafe_allow_html=True)


    # Calculate a long-term moving average (e.g., 200-day)
    df['SMA200'] = df['Close'].rolling(window=200).mean()

    # Determine if user should invest based on recent closing price
    invest_recent = df['Close'].iloc[-1] > df['Close'].iloc[-3]

    # Determine if user should invest based on long-term trend
    invest_long_term = df['Close'].iloc[-1] > df['SMA200'].iloc[-1]

    # Display a graphical suggestion with reasons
    st.subheader('Investment Suggestion')

    if invest_recent and invest_long_term:
        st.markdown('<h3 style="color:green;">✅ Invest</h3>', unsafe_allow_html=True)
        st.write("Reasons to Invest:")
        st.write("- **The closing price has increased significantly over the past few days.**")
        st.write("- **Positive news about the company or industry.**")
        st.write("- **The current price is above the 200-day moving average, suggesting a positive long-term trend.**")
    elif invest_recent:
        st.markdown('<h3 style="color:green;">✅ Invest</h3>', unsafe_allow_html=True)
        st.write("Reasons to Invest:")
        st.write("- **The closing price has increased significantly over the past few days.**")
        st.write("- **Positive news about the company or industry.**")
        st.write("- **Be cautious about the long-term trend.**")
    elif invest_long_term:
        st.markdown('<h3 style="color:orange;">⚠️ Consider Investing</h3>', unsafe_allow_html=True)
        st.write("Reasons to Consider Investing:")
        st.write("- **The current price is above the 200-day moving average, suggesting a positive long-term trend.**")
        st.write("- **Be cautious about recent price fluctuations.**")
    else:
        st.markdown('<h3 style="color:red;">❌ Do Not Invest</h3>', unsafe_allow_html=True)
        st.write("Reasons to Be Cautious:")
        st.write("- **The closing price has decreased recently**.")
        st.write("- **Negative news about the company or industry**.")
        st.write("- **The current price is below the 200-day moving average, suggesting a potential negative trend.**")

            

if __name__ == '__main__':
    main()
