import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt  # For financial visualization
import os  # For file checking

# Load the dataset
data_path = r'C:\Users\Tlotliso\Desktop\PredictiveStockModel\data\stock_data.csv'

# Check if the file exists
if not os.path.isfile(data_path):
    raise FileNotFoundError(f"No such file or directory: '{data_path}'")

# Load the dataset
data = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)

# Display the first few rows and the data types of the dataset
print("Data Overview:")
print(data.head())
print("\nData Types:")
print(data.dtypes)

# Select only the relevant numeric columns (Close prices for each stock)
close_prices = data.xs('Close', level=1, axis=1)

# Normalize the data before feature creation
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices['AMZN'].values.reshape(-1, 1))

# Generate features and labels
def create_features_labels(data, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

# Create features and labels for the first stock (e.g., AMZN)
X, y = create_features_labels(scaled_prices)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = keras.Sequential()

# Add LSTM layers with dropout and L2 regularization
model.add(layers.LSTM(50, return_sequences=True, activation='relu', 
                      input_shape=(X_train.shape[1], 1), 
                      kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.2))  # Dropout layer to prevent overfitting

model.add(layers.LSTM(50, return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.2))

# Output layer
model.add(layers.Dense(1))  # Output layer

# Compile the model with an optimized learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Define the path for saving the model with the correct format (.keras)
checkpoint_path = 'models/stock_lstm_model.keras'
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')

# Train the model with validation data
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Evaluate the model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'\nTest Loss (MSE): {test_loss:.4f}')

# Make predictions
predictions = model.predict(X_test)

# Reverse scaling for predictions and actual values
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate additional evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)

print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')

# Save the best model as .keras file
model.save('models/stock_lstm_model.keras')
print("Model saved as 'stock_lstm_model.keras'.")

# Prepare data for visualization with moving averages and Bollinger Bands
predicted_dates = close_prices.index[-len(predictions):]  # Dates corresponding to predictions
predicted_df = pd.DataFrame(predictions_rescaled, index=predicted_dates, columns=['Predicted'])

# Add real closing prices for comparison
predicted_df['Actual'] = y_test_rescaled

# Calculate moving averages
predicted_df['MA20'] = predicted_df['Predicted'].rolling(window=20).mean()
predicted_df['MA50'] = predicted_df['Predicted'].rolling(window=50).mean()

# Calculate Bollinger Bands
predicted_df['BB_upper'] = predicted_df['MA20'] + (2 * predicted_df['Predicted'].rolling(window=20).std())
predicted_df['BB_lower'] = predicted_df['MA20'] - (2 * predicted_df['Predicted'].rolling(window=20).std())

# Plot the predictions along with moving averages and Bollinger Bands
plt.figure(figsize=(14, 7))
plt.plot(predicted_df.index, predicted_df['Predicted'], label='Predicted Price', color='blue')
plt.plot(predicted_df.index, predicted_df['Actual'], label='Actual Price', color='green')
plt.plot(predicted_df.index, predicted_df['MA20'], label='MA20', color='orange')
plt.plot(predicted_df.index, predicted_df['MA50'], label='MA50', color='purple')
plt.fill_between(predicted_df.index, predicted_df['BB_upper'], predicted_df['BB_lower'], color='gray', alpha=0.3, label='Bollinger Bands')
plt.title('Stock Price Predictions with Moving Averages and Bollinger Bands')
plt.legend()
plt.show()

# Optionally: Load the saved model for further use
loaded_model = tf.keras.models.load_model('models/stock_lstm_model.keras')
print("Loaded model for further use.")
