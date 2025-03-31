# main.py

from reinforcement_learning import run_reinforcement_learning

# Specify the paths to your CSV files
data_file_path = 'Book1.csv'  # Ensure this file is in the AIP directory
new_env_file_path = 'Book2.csv'  # Ensure this file is in the AIP directory

# Call the function with the paths as arguments
run_reinforcement_learning(data_file_path, new_env_file_path)
