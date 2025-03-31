import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Simulate realistic data (as done earlier)
np.random.seed(42)
def simulate_stock_data(n, mu=0.001, sigma=0.02):
    return np.cumprod(1 + np.random.normal(mu, sigma, n)) * 100

def simulate_bond_data(n, mu=0.0005, sigma=0.01):
    return np.cumprod(1 + np.random.normal(mu, sigma, n)) * 100

def simulate_commodity_data(n, amplitude=10, frequency=12, noise_level=2):
    t = np.linspace(0, 4 * np.pi, n)  # Adjusted for monthly intervals
    return amplitude * np.sin(frequency * t / (2 * np.pi)) + amplitude + np.random.normal(0, noise_level, n)

# Generate simulated data for stocks, bonds, and commodities
n_points = 200  # Number of historical data points
dates = pd.date_range(start='2010-01-01', periods=n_points, freq='M')  # Monthly date range
stocks = pd.DataFrame({
    'MSFT': simulate_stock_data(n_points),
    'TSLA': simulate_stock_data(n_points, mu=0.002, sigma=0.03),
    'AAPL': simulate_stock_data(n_points, mu=0.0015, sigma=0.025)
}, index=dates)

bonds = pd.DataFrame({
    'Bond1': simulate_bond_data(n_points),
    'Bond2': simulate_bond_data(n_points, mu=0.0003, sigma=0.008),
    'Bond3': simulate_bond_data(n_points, mu=0.0007, sigma=0.015)
}, index=dates)

commodities = pd.DataFrame({
    'Crude_Oil': simulate_commodity_data(n_points),
    'Corn': simulate_commodity_data(n_points, amplitude=8, frequency=10, noise_level=1.5),
    'Wheat': simulate_commodity_data(n_points, amplitude=12, frequency=14, noise_level=2.5)
}, index=dates)

# Combine into a single portfolio
portfolio = pd.concat([stocks, bonds, commodities], axis=1)

# Define allocation for each asset
allocation = np.array([0.5, 0.3, 0.2, 0.2, 0.4, 0.4, 1, 0, 0])
portfolio_value = portfolio.dot(allocation)

# Normalize portfolio value for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_portfolio_value = scaler.fit_transform(portfolio_value.values.reshape(-1, 1))

# Prepare data for LSTM
def create_dataset(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 60
X_train, y_train = create_dataset(scaled_portfolio_value, time_steps)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Forecast the next 120 values (10 years of monthly data)
def forecast_future(model, data, time_steps, forecast_horizon):
    future = []
    last_sequence = data[-time_steps:]
    
    for _ in range(forecast_horizon):
        prediction = model.predict(last_sequence.reshape(1, time_steps, 1))
        future.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    return np.array(future)

forecast_horizon = 120  # Forecast for the next 10 years (120 months)
forecast_values = forecast_future(model, scaled_portfolio_value, time_steps, forecast_horizon)

# Reverse the scaling for forecasted values
forecast_values = scaler.inverse_transform(forecast_values.reshape(-1, 1))

# Prepare data for plotting
forecast_dates = pd.date_range(start=portfolio.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='M')
forecast_series = pd.Series(forecast_values.flatten(), index=forecast_dates)

# Plot the results with extended forecast
plt.figure(figsize=(7, 4))
plt.plot(portfolio.index, portfolio_value, label='Historical Portfolio Value', color='blue')
plt.plot(forecast_series.index, forecast_series, label='Extended Forecasted Portfolio Value', color='orange')
plt.title('Optimized Portfolio Value Forecast (10 Years)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()

plt.grid()
plt.show()
