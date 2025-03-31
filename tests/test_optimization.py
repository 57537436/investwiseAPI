# tests/test_optimization.py
import pytest

def test_optimize_portfolio(client):
    # Prepare any necessary data files and new_env_file paths
    # For example, create dummy CSV files or any required inputs
    data_file = "tests/dummy_data.csv"  # Example path, create this file with dummy data
    new_env_file = "tests/dummy_env.csv"  # Example path, create this file with dummy data

    # Create the dummy files for testing
    with open(data_file, 'w') as f:
        f.write("Date,Open,High,Low,Close\n")
        f.write("2024-01-01,100,105,95,102\n")
        f.write("2024-01-02,102,107,98,104\n")

    with open(new_env_file, 'w') as f:
        f.write("Parameter,Value\n")
        f.write("Risk-Free Rate,0.02\n")
        f.write("Market Return,0.08\n")

    # Call the optimize portfolio endpoint
    response = client.post("/optimize_portfolio/", json={"data_file": data_file, "new_env_file": new_env_file})
    
    # Check response status and content
    assert response.status_code == 200
    assert "allocations" in response.json()
    assert "returns" in response.json()
