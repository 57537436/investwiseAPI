import numpy as np
import pandas as pd
import pickle  # Importing pickle for saving and loading the model

# Load your client and market data
client_data = pd.read_csv("merged_client_market_data.csv")

# Define actions (shifting portfolio percentages)
actions = [
    (+0.05, -0.05, 0),    # Stock up, Bond down
    (-0.05, +0.05, 0),    # Stock down, Bond up
    (+0.05, 0, -0.05),    # Stock up, Commodity down
    (0, +0.05, -0.05),    # Bond up, Commodity down
    (+0.01, -0.01, 0),    # Smaller stock/bond shift
    (-0.01, +0.01, 0),    # Smaller stock/bond shift
    (0, +0.01, -0.01),    # Smaller bond/commodity shift
    (+0.02, -0.02, 0)     # Slightly larger stock/bond shift
]

# Initialize Q-table
Q_table = {}

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate

# Reward function (refined: risk-adjusted returns with penalties for risk and transaction costs)
def calculate_reward(client, new_stock_alloc, new_bond_alloc, new_commodity_alloc, market):
    # Expected return based on allocations and market conditions
    expected_return = (
        new_stock_alloc * market['Stock_Price_Index'] +
        new_bond_alloc * market['Bond_Price_Index'] +
        new_commodity_alloc * market['Commodity_Price_Index']
    )

    # Risk-adjusted return (Sharpe ratio approximation: return/volatility)
    portfolio_volatility = (
        new_stock_alloc * market['Stock_Volatility'] +
        new_bond_alloc * market['Bond_Yield'] +
        new_commodity_alloc * market['Commodity_Price_Volatility']  # Assuming you have volatility for commodities
    )
    
    risk_adjusted_return = expected_return / (portfolio_volatility + 1e-6)  # Avoid division by zero

    # Penalty for exceeding risk tolerance
    risk_penalty = 0
    if client['Risk_Tolerance'] == 'Low':
        risk_penalty = abs(new_stock_alloc - 0.3) * 0.1  # Higher penalty for higher stock allocation

    # Transaction cost penalty (larger shifts in allocations incur higher costs)
    transaction_cost_penalty = abs(new_stock_alloc - client['Current_Stock_Allocation']) * 0.02 + \
                               abs(new_bond_alloc - client['Current_Bond_Allocation']) * 0.02 + \
                               abs(new_commodity_alloc - client['Current_Commodity_Allocation']) * 0.02

    # Total reward = risk-adjusted return - penalties
    total_reward = risk_adjusted_return - risk_penalty - transaction_cost_penalty
    return total_reward

# Train the Q-Learning model
for episode in range(500):  # Training episodes
    total_reward = 0  # Track total reward for this episode

    for _, row in client_data.iterrows():
        # Expanded state
        state = (
            row['Stock_Price_Index'], 
            row['Bond_Price_Index'], 
            row['Commodity_Price_Index'], 
            row['Stock_Volatility'], 
            row['Bond_Yield'], 
            row['Inflation_Rate'], 
            row['Risk_Tolerance'], 
            row['Investment_Horizon']
        )

        # Initialize Q-table for new states
        if state not in Q_table:
            Q_table[state] = np.random.rand(len(actions)) * 0.1  # Use small random values

        # Choose action (explore or exploit)
        if np.random.uniform(0, 1) < epsilon:
            action_idx = np.random.choice(len(actions))  # Explore
        else:
            action_idx = np.argmax(Q_table[state])  # Exploit

        action = actions[action_idx]

        # Update portfolio allocations based on action
        new_stock_alloc = row['Current_Stock_Allocation'] + action[0]
        new_bond_alloc = row['Current_Bond_Allocation'] + action[1]
        new_commodity_alloc = row['Current_Commodity_Allocation'] + action[2]

        # Keep allocations valid (ensure they sum to 1)
        total_alloc = new_stock_alloc + new_bond_alloc + new_commodity_alloc
        if total_alloc > 0:  # Avoid division by zero
            new_stock_alloc /= total_alloc
            new_bond_alloc /= total_alloc
            new_commodity_alloc /= total_alloc

        # Calculate reward
        reward = calculate_reward(row, new_stock_alloc, new_bond_alloc, new_commodity_alloc, market=row)
        total_reward += reward  # Track total reward for this episode

        # Find next state (assuming sequential rows represent time steps)
        next_state_idx = _ + 1  # Move to the next row
        if next_state_idx < len(client_data):
            next_row = client_data.iloc[next_state_idx]
            next_state = (
                next_row['Stock_Price_Index'], 
                next_row['Bond_Price_Index'], 
                next_row['Commodity_Price_Index'], 
                next_row['Stock_Volatility'], 
                next_row['Bond_Yield'], 
                next_row['Inflation_Rate'], 
                next_row['Risk_Tolerance'], 
                next_row['Investment_Horizon']
            )
        else:
            next_state = state  # End of dataset, repeat current state

        # Update Q-value using the Bellman equation
        Q_table[state][action_idx] += alpha * (reward + gamma * np.max(Q_table.get(next_state, np.zeros(len(actions)))) - Q_table[state][action_idx])

    # Optional: Decay epsilon
    epsilon = max(0.01, epsilon * 0.99)  # Decay epsilon to reduce exploration over time

# Save the Q-table to a file after training
with open("models/Officialreinforcement_model.pkl", "wb") as f:
    pickle.dump(Q_table, f)
print("Model saved successfully!")

# Test with a new client
MyClient = pd.read_csv("market_client_data.csv")

new_market_data = {
    'Stock_Price_Index': MyClient['Stock_Price_Index'].iloc[0],
    'Bond_Price_Index': MyClient['Bond_Price_Index'].iloc[0],
    'Commodity_Price_Index': MyClient['Commodity_Price_Index'].iloc[0],
    'Stock_Volatility': MyClient['Stock_Volatility'].iloc[0],
    'Bond_Yield': MyClient['Bond_Yield'].iloc[0],
    'GDP_Growth': MyClient['GDP_Growth'].iloc[0],
    'Inflation_Rate': MyClient['Inflation_Rate'].iloc[0]
}

# Test on a new client with given market conditions
test_client = {
    'Current_Stock_Allocation': MyClient['Current_Stock_Allocation'].iloc[0],
    'Current_Bond_Allocation': MyClient['Current_Bond_Allocation'].iloc[0],
    'Current_Commodity_Allocation': MyClient['Current_Commodity_Allocation'].iloc[0],
    'Risk_Tolerance': MyClient['Risk_Tolerance'].iloc[0],
    'Investment_Horizon': MyClient['Investment_Horizon'].iloc[0]
}

# Find the best action based on the Q-table
state = (new_market_data['Stock_Price_Index'], new_market_data['Bond_Price_Index'], new_market_data['Commodity_Price_Index'])

if state in Q_table:
    best_action_idx = np.argmax(Q_table[state])
else:
    best_action_idx = np.random.choice(len(actions))  # Explore if state not in Q-table
best_action = actions[best_action_idx]

# Update allocations based on the best action
new_stock_alloc = test_client['Current_Stock_Allocation'] + best_action[0]
new_bond_alloc = test_client['Current_Bond_Allocation'] + best_action[1]
new_commodity_alloc = test_client['Current_Commodity_Allocation'] + best_action[2]

# Normalize allocations to ensure they are valid
total_alloc = new_stock_alloc + new_bond_alloc + new_commodity_alloc
if total_alloc > 1:
    new_stock_alloc /= total_alloc
    new_bond_alloc /= total_alloc
    new_commodity_alloc /= total_alloc

# Display the recommended allocations
print(f"Recommended Allocations: Stocks: {new_stock_alloc:.2f}, Bonds: {new_bond_alloc:.2f}, Commodities: {new_commodity_alloc:.2f}")
print(f"Best Action Taken: {best_action}")
