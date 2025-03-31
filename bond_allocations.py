import numpy as np
import pandas as pd
import pickle  # For saving/loading the Q-table

# Step 2: Load Training Data and Implement Q-Learning Model
data = pd.read_csv('bond_data.csv')

# Define action space (allocations to 3 bonds, sum = 100%)
actions = [(a, b, 1 - a - b) for a in np.arange(0, 1.1, 0.1) 
           for b in np.arange(0, 1.1, 0.1) if a + b <= 1]

# Discretize state into a tuple (for Q-table)
def discretize_state(row):
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    horizon_mapping = {'Short': 0, 'Medium': 1, 'Long': 2}
    goal_mapping = {'Income': 0, 'Preservation': 1, 'Growth': 2}

    state = (
        risk_mapping[row['Risk Tolerance']],
        horizon_mapping[row['Investment Horizon']],
        goal_mapping[row['Client Goal']],
        int(row['Interest Rate'] * 10),  # Discretizing interest rates
        int(row['Inflation Rate'] * 10),  # Discretizing inflation
        int(row['GDP Growth'] * 10)  # Discretizing GDP growth
    )
    return state

# Initialize Q-table
Q_table = {}
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate
num_days = 500

# Reward function
def calculate_reward(row, action):
    alloc_A, alloc_B, alloc_C = action
    bond_return = (alloc_A * row['Predicted Return A'] +
                   alloc_B * row['Predicted Return B'] +
                   alloc_C * row['Predicted Return C'])
    risk_penalty = -0.01 * (alloc_A * row['Volatility A'] +
                            alloc_B * row['Volatility B'] +
                            alloc_C * row['Volatility C'])
    return bond_return + risk_penalty

# Q-learning algorithm
for day in range(num_days - 1):
    state = discretize_state(data.iloc[day])
    if state not in Q_table:
        Q_table[state] = np.zeros(len(actions))
    
    if np.random.uniform(0, 1) < epsilon:
        # Explore
        action_idx = np.random.choice(len(actions))
    else:
        # Exploit
        action_idx = np.argmax(Q_table[state])
    
    action = actions[action_idx]
    reward = calculate_reward(data.iloc[day], action)
    
    # Get next state
    next_state = discretize_state(data.iloc[day + 1])
    if next_state not in Q_table:
        Q_table[next_state] = np.zeros(len(actions))
    
    # Bellman update
    Q_table[state][action_idx] = (1 - alpha) * Q_table[state][action_idx] + \
                                 alpha * (reward + gamma * np.max(Q_table[next_state]))

# Save the Q-table model
with open('bond_allocation_model.pkl', 'wb') as f:
    pickle.dump(Q_table, f)

# Step 3: Test Function with Client Data
def find_closest_state(client_state, Q_table):
    min_distance = float('inf')
    closest_state = None
    
    for state in Q_table.keys():
        # Calculate a simple distance measure between the states
        distance = np.linalg.norm(np.array(client_state) - np.array(state))
        if distance < min_distance:
            min_distance = distance
            closest_state = state
    
    return closest_state

# Modified test function to handle unseen states
def test_client_data(file_name):
    client_data = pd.read_csv(file_name)

    for i, row in client_data.iterrows():
        test_state = discretize_state(row)

        # If the exact state isn't in the Q-table, find the closest matching state
        if test_state in Q_table:
            best_action_idx = np.argmax(Q_table[test_state])
            best_action = actions[best_action_idx]
            best_action_percentage = np.round(np.array(best_action) * 100, 2)  # Convert to percentage
            print(f"Client {i + 1}:")
            print(f"  Bond A: {best_action_percentage[0]}%")
            print(f"  Bond B: {best_action_percentage[1]}%")
            print(f"  Bond C: {best_action_percentage[2]}%\n")
        else:
            closest_state = find_closest_state(test_state, Q_table)
            if closest_state is not None:
                best_action_idx = np.argmax(Q_table[closest_state])
                best_action = actions[best_action_idx]
                best_action_percentage = np.round(np.array(best_action) * 100, 2)
                print(f"Client {i + 1} (Closest Match Found):")
                print(f"  Bond A: {best_action_percentage[0]}%")
                print(f"  Bond B: {best_action_percentage[1]}%")
                print(f"  Bond C: {best_action_percentage[2]}%\n")
            else:
                print(f"Client {i + 1}: No suitable state or closest match found.\n")

# Test the model using the test client data
test_client_data('client_test_data.csv')

# Load the saved Q-table model
with open('bond_allocation_model.pkl', 'rb') as f:
    loaded_Q_table = pickle.load(f)
    print("Loaded Q-table from file.")
