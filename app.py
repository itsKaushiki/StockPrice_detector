import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model_path = r'C:\Users\Kaushiki\OneDrive\Desktop\Jupyter\Stock Predictions Model.keras'
model = load_model(model_path)

# Streamlit app title
st.title('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data using yfinance
data = yf.download(stock, start, end)

# Display stock data in Streamlit
st.subheader('Stock Data')
st.write(data)

# Prepare data for model prediction
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# Plot Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(ma_50_days, 'r', label='MA50')
ax1.plot(data.Close, 'g', label='Price')
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(ma_50_days, 'r', label='MA50')
ax2.plot(ma_100_days, 'b', label='MA100')
ax2.plot(data.Close, 'g', label='Price')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.legend()
st.pyplot(fig2)

# Plot Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(ma_100_days, 'r', label='MA100')
ax3.plot(ma_200_days, 'b', label='MA200')
ax3.plot(data.Close, 'g', label='Price')
ax3.set_xlabel('Date')
ax3.set_ylabel('Price')
ax3.legend()
st.pyplot(fig3)

# Prepare data for model prediction
x = []
y_actual = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y_actual.append(data_test_scaled[i, 0])

x, y_actual = np.array(x), np.array(y_actual)

# Make predictions using the loaded model
predictions = model.predict(x)

# Rescale the predictions and actual values
scale = 1 / scaler.scale_
predicted_prices = predictions * scale
y_actual_prices = y_actual * scale

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(predicted_prices, 'r', label='Predicted Price')
ax4.plot(y_actual_prices, 'g', label='Original Price')
ax4.set_xlabel('Date')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)

# Add a button for data download
st.markdown('<h3 style="color:blue;">Download Data</h3>', unsafe_allow_html=True)
data_csv = data.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=data_csv,
    file_name='stock_data.csv',
    mime='text/csv'
)

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f0f0f0;
    color: black;
    text-align: center;
    padding: 10px 0;
}
</style>
<div class="footer">
    <p>Stock Market Predictor App - Powered by Streamlit</p>
</div>
""", unsafe_allow_html=True)
