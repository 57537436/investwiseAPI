import numpy as np
import pandas as pd
import os

# Seed for reproducibility
np.random.seed(42)

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load training data from CSV
def load_training_data(csv_path):
    return pd.read_csv(csv_path)

# Load a single client’s data from CSV
def load_single_client_data(csv_path):
    return pd.read_csv(csv_path)

# Reward function: portfolio growth minus risk penalty
def reward_function(portfolio_return, volatility, risk_tolerance):
    risk_penalty = volatility * (5 - risk_tolerance)  # Higher risk tolerance, lower penalty
    return portfolio_return - risk_penalty

# Simulate portfolio returns based on commodity price changes and allocations
def simulate_portfolio_return(allocations, prices):
    # Prices should only be for the three commodities
    commodity_prices = prices[:3]  # First three entries represent the commodity prices
    return np.dot(allocations, commodity_prices)

# Q-Learning Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 100  # Number of episodes for training

# Define state and action spaces
state_space = ['Risk_Tolerance', 'Investment_Horizon', 'Commodity_1_Price', 'Commodity_1_Volatility',
               'Commodity_2_Price', 'Commodity_2_Volatility', 'Commodity_3_Price', 'Commodity_3_Volatility',
               'Inflation', 'Interest_Rates']

# Actions represent allocation percentages to commodities, e.g., [0.5, 0.3, 0.2] for the three commodities
action_space = [
    [0.5, 0.3, 0.2],
    [0.4, 0.4, 0.2],
    [0.3, 0.5, 0.2],
    [0.4, 0.3, 0.3],
    [0.3, 0.3, 0.4]
]

# Load the training data from the CSV file
training_data_path = 'training_data.csv'  # Adjust the path as needed
df_clients = load_training_data(training_data_path)

num_clients = len(df_clients['Client_ID'].unique())  # Get number of unique clients

# Initialize Q-Table for clients
q_table = np.zeros((num_clients, len(action_space)))

# Q-Learning algorithm for training with multiple clients
for episode in range(num_episodes):
    for client_id in range(num_clients):
        state = df_clients[df_clients['Client_ID'] == client_id][state_space].sample(1).values[0]
        
        # Choose action using epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action_idx = np.random.choice(len(action_space))  # Explore: random action
        else:
            action_idx = np.argmax(q_table[client_id])  # Exploit: best known action
        
        action = action_space[action_idx]
        
        # Get current prices and volatilities
        prices = state[2:9:2]  # Commodity prices
        volatilities = state[3:10:2]  # Commodity volatilities
        risk_tolerance = state[0]
        
        # Calculate portfolio return and reward
        portfolio_return = simulate_portfolio_return(action, prices)
        reward = reward_function(portfolio_return, np.mean(volatilities), risk_tolerance)
        
        # Update Q-Table using the Q-Learning update rule
        future_rewards = np.max(q_table[client_id])
        q_table[client_id, action_idx] += alpha * (reward + gamma * future_rewards - q_table[client_id, action_idx])

# Save the trained Q-Table to the 'models' folder
q_table_file_path = 'models/q_table.pkl'
np.save(q_table_file_path, q_table)
print(f"Q-Table saved to {q_table_file_path}")

# Load the single client's data from the CSV file
single_client_data_path = 'single_client.csv'  # Adjust the path as needed
single_client_data = load_single_client_data(single_client_data_path)

# Extract relevant state information for the single client
single_client_state = single_client_data[state_space].values[0]

# Use the trained model to allocate funds for this single client
single_client_id = 0  # Assuming we're testing for a client with ID 0
action_idx = np.argmax(q_table[single_client_id])
final_allocation = action_space[action_idx]

# Display the final allocation for the single client
print("Final Allocation for the client:")
print(f"Commodity 1: {final_allocation[0]:.2f}, Commodity 2: {final_allocation[1]:.2f}, Commodity 3: {final_allocation[2]:.2f}")
