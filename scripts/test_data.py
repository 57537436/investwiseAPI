import pandas as pd
import numpy as np
import tensorflow as tf
import mplfinance as mpf  # For candlestick chart plotting
import os

# Load the trained model
model_path = 'models/stock_lstm_model.keras'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"No saved model found at: '{model_path}'")

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load the dataset (same preprocessing as used in train_data.py)
data_path = r'C:\Users\Tlotliso\Desktop\PredictiveStockModel\data\stock_data.csv'
if not os.path.isfile(data_path):
    raise FileNotFoundError(f"No such file or directory: '{data_path}'")

data = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)
close_prices = data.xs('Close', level=1, axis=1)

# Prepare test data
def create_features_labels(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Using the stock 'AMZN' for prediction
X_test, y_test = create_features_labels(close_prices['AMZN'].values)

# Reshape for LSTM
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions using the loaded model
predictions = model.predict(X_test)

# Prepare data for candlestick plotting
predicted_dates = close_prices.index[-len(predictions):]
predicted_df = pd.DataFrame(predictions, index=predicted_dates, columns=['Predicted'])

# Add Open, High, Low, and Close prices for plotting (dummy for candlestick)
last_known_price = close_prices['AMZN'].iloc[-len(predictions) - 1]
predicted_df['Open'] = last_known_price
predicted_df['High'] = np.maximum(predicted_df['Predicted'], last_known_price)
predicted_df['Low'] = np.minimum(predicted_df['Predicted'], last_known_price)
predicted_df['Close'] = predicted_df['Predicted']

# Renaming for mplfinance
predicted_df = predicted_df[['Open', 'High', 'Low', 'Close']]

# Plot the candlestick chart of predicted prices
mpf.plot(predicted_df, type='candle', style='charles', title='Predicted Stock Prices', volume=False)
