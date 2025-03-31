import numpy as np
import random
import pickle  # Import the pickle module

# Simulate Client Data
def simulate_client_data(num_clients=1000):
    client_data = []
    for _ in range(num_clients):
        literacy_level = random.choice([0, 1, 2])  # Beginner (0), Intermediate (1), Advanced (2)
        portfolio_risk = random.choice([0, 1, 2])  # Conservative (0), Balanced (1), Aggressive (2)
        portfolio_allocation_stocks = random.randint(0, 100)
        portfolio_allocation_bonds = 100 - portfolio_allocation_stocks
        investment_goal = random.choice([0, 1, 2])  # Short-term (0), Growth (1), Income (2)
        age_group = random.choice([0, 1, 2])  # 20-30 (0), 30-40 (1), 40-50 (2)
        investment_experience = random.randint(1, 10)  # 1 to 10 years
        quiz_score = random.randint(30, 100)  # Initial quiz score (between 30 and 100)
        
        client_data.append([
            literacy_level, portfolio_risk, portfolio_allocation_stocks,
            portfolio_allocation_bonds, investment_goal, age_group,
            investment_experience, quiz_score
        ])
    return np.array(client_data)

# Simulate Client Data
client_data = simulate_client_data()

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((state_size, action_size))  # Initialize Q-table
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit (choose action with max Q-value)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)  # Save Q-table to a file

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)  # Load Q-table from a file

# Set parameters
state_size = 3  # Assuming simplified state space (literacy level)
action_size = 5  # Number of available modules (actions)

# Create a Q-Learning agent
agent = QLearningAgent(state_size, action_size)

# Simulate rewards for each action (for simplicity, random reward generation)
def get_reward(literacy_level, action):
    # Assume rewards based on improved quiz scores and portfolio alignment
    reward_matrix = {
        0: [10, 15, 5, 20, 5],  # Beginner: Higher reward for basic lessons
        1: [5, 10, 20, 25, 5],  # Intermediate: Higher reward for advanced lessons
        2: [5, 5, 25, 30, 10]   # Advanced: Highest reward for advanced topics
    }
    return reward_matrix[literacy_level][action]

# Train the agent with simulated data
def train_agent(client_data, agent, episodes=1000):
    for episode in range(episodes):
        for client in client_data:
            state = int(client[0])  # Use literacy level as the state (simplified)
            action = agent.choose_action(state)  # Agent selects an action
            reward = get_reward(state, action)  # Simulate reward based on state and action
            next_state = min(state + 1, 2)  # Assume literacy improves (simplified)
            agent.update_q_value(state, action, reward, next_state)

# Train the Q-learning agent with the simulated client data
train_agent(client_data, agent)

# Save the trained Q-learning model
agent.save_model('q_learning_model.pkl')
print("Model saved.")

# Learning Progress Tracker
class LearningProgressTracker:
    def __init__(self):
        self.client_progress = {}  # Dictionary to store progress for each client

    def initialize_client(self, client_id):
        # Initialize a new client's learning progress
        self.client_progress[client_id] = {
            'quiz_scores': [],
            'course_completion': 0,
            'engagement_level': 0,  # Number of interactions
            'portfolio_adjustments': 0,
            'modules_completed': []  # List to track completed modules
        }

    def update_quiz_score(self, client_id, score):
        # Update quiz score for a client
        if client_id in self.client_progress:
            self.client_progress[client_id]['quiz_scores'].append(score)

    def mark_module_completed(self, client_id, module_name):
        # Mark a module as completed for a client
        if client_id in self.client_progress:
            self.client_progress[client_id]['course_completion'] += 1
            self.client_progress[client_id]['modules_completed'].append(module_name)

    def update_engagement(self, client_id):
        # Increment engagement level for a client
        if client_id in self.client_progress:
            self.client_progress[client_id]['engagement_level'] += 1

    def update_portfolio_adjustments(self, client_id):
        # Increment portfolio adjustments for a client
        if client_id in self.client_progress:
            self.client_progress[client_id]['portfolio_adjustments'] += 1

    def get_progress(self, client_id):
        # Retrieve a client's learning progress
        return self.client_progress.get(client_id, "Client not found")

# Example Usage
tracker = LearningProgressTracker()

# Initialize a new client with ID 1
tracker.initialize_client(client_id=1)

# Recommend a module to a client and track progress
def recommend_module(client, agent, tracker, client_id):
    state = int(client[0])  # Use literacy level as the state
    action = agent.choose_action(state)  # Agent selects an action (module)
    modules = ["basic_risk_management", "diversification_strategies", "portfolio_rebalancing", 
               "derivatives_intro", "retirement_planning"]
    recommended_module = modules[action]

    # Mark the module as completed in the tracker
    tracker.mark_module_completed(client_id, recommended_module)

    return recommended_module

# Example client
example_client = [0, 1, 60, 40, 1, 0, 2, 45]  # A beginner with a balanced portfolio
# Example client ID
client_id = 1

# Recommend a module
recommended_module = recommend_module(example_client, agent, tracker, client_id)
print(f"Recommended module for the client: {recommended_module}")

# Update progress after module completion
tracker.update_quiz_score(client_id, 75)  # After completing a quiz
tracker.update_engagement(client_id)  # After interaction with the platform
tracker.update_portfolio_adjustments(client_id)  # After making changes to their portfolio

# Retrieve and display progress for client 1
client_progress = tracker.get_progress(client_id)
print(f"Learning progress for client {client_id}: {client_progress}")

# Load the trained Q-learning model
agent.load_model('q_learning_model.pkl')
print("Model loaded.")
