import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Load your bond data
df = pd.read_csv('bond_data.csv')

# Inspect the columns to match the correct features and target
print(df.columns)

# Define your features (X) and target (y)
X = df[['Bond A Price', 'Bond B Price', 'Bond C Price', 'Interest Rate', 'Inflation Rate', 'GDP Growth', 'Risk Tolerance', 'Investment Horizon', 'Client Goal']]
y = df['Predicted Return A']  # Example target

# Encode categorical variables if needed
X = pd.get_dummies(X, columns=['Risk Tolerance', 'Investment Horizon', 'Client Goal'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple Keras model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance (using Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model inside the 'models' folder in .keras format
model.save('models/bond_allocation_model.keras')

print(f"Model training complete and saved to models/bond_allocation_model.keras")
