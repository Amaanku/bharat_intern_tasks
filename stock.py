import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#this wiil be generate the todays date
current_date = datetime.datetime.now().date()  


# Define the stock symbol and date range
stock_symbol = input("enter the stock name and add .ns at the end: ")  
start_date = str(current_date - datetime.timedelta(days=365))
end_date = str(current_date)

print("The stock of "+stock_symbol+" will be loaded below from "+start_date+" to "+end_date)

# Fetch stock price data using yfinance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)

# Preprocess the data
data = stock_data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(0.8 * len(data_scaled))
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append((seq, target))
    return sequences

seq_length = 10  # You can adjust this sequence length
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

# Create the LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Convert sequences to numpy arrays
X_train = np.array([seq for seq, target in train_sequences])
y_train = np.array([target for seq, target in train_sequences])
X_test = np.array([seq for seq, target in test_sequences])
y_test = np.array([target for seq, target in test_sequences])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

print(f"Train Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predict stock prices
def predict_prices(model, data, scaler, seq_length, num_predictions):
    last_sequence = data[-seq_length:]
    predictions = []

    for _ in range(num_predictions):
        prediction = model.predict(last_sequence.reshape(1, seq_length, 1))
        predictions.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction[0, 0])
    
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predicted_prices

num_predictions = len(test_data) - seq_length
predicted_prices = predict_prices(model, test_data, scaler, seq_length, num_predictions)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[train_size+seq_length:], stock_data['Close'][train_size+seq_length:], label='True Prices', color='blue')
plt.plot(stock_data.index[train_size+seq_length:], predicted_prices, label='Predicted Prices', color='red')
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
