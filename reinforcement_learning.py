import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def run_reinforcement_learning(train_file: str, test_file: str):
    # Load training dataset
    train_data = pd.read_csv(train_file)

    # Preprocessing: Normalize necessary columns for training data
    columns_to_scale = [
        'stock1_close', 'stock2_close', 'bond_yield', 'gold_price', 
        'oil_price', 'CPI', 'unemployment_rate', 'GDP_growth', 
        'interest_rate', 'consumer_confidence', 'VIX'
    ]

    # Check for missing columns in training data
    missing_columns = [col for col in columns_to_scale if col not in train_data.columns]
    if missing_columns:
        print(f"Error: The following columns are missing in the training dataset: {missing_columns}")
        return

    scaler = MinMaxScaler()
    train_data_scaled = train_data.copy()
    train_data_scaled[columns_to_scale] = scaler.fit_transform(train_data[columns_to_scale])

    # Define the state and action spaces for training
    states = train_data_scaled[['stock1_close', 'stock2_close', 'bond_yield', 'gold_price', 
                                 'oil_price', 'CPI', 'unemployment_rate', 'GDP_growth', 
                                 'interest_rate', 'VIX']].values

    # Action space: Change in stock, bond, and commodity allocations
    actions = [
        [+5, -5, 0], [-5, +5, 0], [0, 0, +5], [-5, 0, +5], 
        [0, -5, +5], [+5, 0, -5], [0, +5, -5], [-5, +5, -5], 
        [0, 0, 0]  # No change
    ]

    num_states = states.shape[0]
    num_actions = len(actions)

    # Initialize Q-table
    Q_table = np.zeros((num_states, num_actions))

    # Hyperparameters
    alpha = 0.1   # Learning rate
    gamma = 0.9   # Discount factor
    epsilon = 0.1 # Exploration rate

    # Get portfolio allocation and return data from training data
    stock_allocations = train_data_scaled['stock_allocation'].values
    bond_allocations = train_data_scaled['bond_allocation'].values
    commodity_allocations = train_data_scaled['commodity_allocation'].values
    portfolio_returns = train_data['portfolio_return'].values  # Unscaled for reward calculation

    # Function to calculate portfolio return
    def calculate_portfolio_return(stock_allocation, bond_allocation, commodity_allocation, time_step):
        stock_return = stock_allocations[time_step] * stock_allocation
        bond_return = bond_allocations[time_step] * bond_allocation
        commodity_return = commodity_allocations[time_step] * commodity_allocation
        return stock_return + bond_return + commodity_return

    # Q-learning loop for training
    num_episodes = 1000
    for episode in range(num_episodes):
        state = np.random.randint(0, num_states)  # Start from a random state
        total_allocations = np.array([stock_allocations[state], bond_allocations[state], commodity_allocations[state]])

        for step in range(num_states):
            # Epsilon-greedy policy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(num_actions)  # Explore
            else:
                action = np.argmax(Q_table[state, :])   # Exploit the best action
            
            # Update allocations
            allocation_change = actions[action]
            total_allocations += np.array(allocation_change) / 100.0  # Adjust allocations by ±5%
            total_allocations = np.clip(total_allocations, 0, 1)  # Ensure allocations remain between 0 and 1

            # Calculate reward (portfolio return)
            reward = calculate_portfolio_return(*total_allocations, state)
            
            # Move to the next state
            next_state = (state + 1) % num_states
            
            # Update Q-value using the Q-learning formula
            Q_table[state, action] = (1 - alpha) * Q_table[state, action] + alpha * (reward + gamma * np.max(Q_table[next_state, :]))

            # Update the current state
            state = next_state

    # Testing on Book2.csv
    test_data = pd.read_csv(test_file)

    # Preprocessing: Normalize necessary columns for testing data
    test_data_scaled = test_data.copy()
    test_data_scaled[columns_to_scale] = scaler.transform(test_data[columns_to_scale])

    # Define the state for testing
    test_states = test_data_scaled[['stock1_close', 'stock2_close', 'bond_yield', 'gold_price', 
                                     'oil_price', 'CPI', 'unemployment_rate', 'GDP_growth', 
                                     'interest_rate', 'VIX']].values

    # Output the optimal allocations for each time step in the testing data
    optimal_allocations = []
    for step in range(test_states.shape[0]):
        # Select the best action using the Q-table
        action = np.argmax(Q_table[step, :])
        allocation_change = actions[action]
        total_allocations = np.array([test_data_scaled['stock_allocation'][step], 
                                       test_data_scaled['bond_allocation'][step], 
                                       test_data_scaled['commodity_allocation'][step]])
        
        # Adjust allocations based on the selected action
        total_allocations += np.array(allocation_change) / 100.0  # Adjust by ±5%
        total_allocations = np.clip(total_allocations, 0, 1)  # Ensure allocations are within [0, 1]

        # Store the current allocation
        optimal_allocations.append(total_allocations.copy())

    optimal_allocations_df = pd.DataFrame(optimal_allocations, columns=['Stock_Allocation', 'Bond_Allocation', 'Commodity_Allocation'])
    print(optimal_allocations_df.head())  # Show the first 5 rows of optimal allocations

# Main function call
if __name__ == "__main__":
    train_file = 'Book1.csv'  # Replace with your actual 'Book1.csv' file path
    test_file = 'Book2.csv'    # Replace with your actual 'Book2.csv' file path
    run_reinforcement_learning(train_file, test_file)
