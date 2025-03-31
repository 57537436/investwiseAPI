import numpy as np
import matplotlib.pyplot as plt

def create_visualization(optimal_allocations):
    """
    Function to create a line plot for optimal allocations over time.
    
    Parameters:
    - optimal_allocations: list of numpy arrays or DataFrame with shape (n_time_steps, 3),
                          where each array contains the stock, bond, and commodity allocations.
    
    Returns: None
    """
    # Convert the input to a NumPy array if it's not already
    allocations = np.array(optimal_allocations)
    
    plt.figure(figsize=(12, 6))
    
    # Plot each allocation
    plt.plot(allocations[:, 0], label='Stock Allocation', color='blue', marker='o')
    plt.plot(allocations[:, 1], label='Bond Allocation', color='orange', marker='o')
    plt.plot(allocations[:, 2], label='Commodity Allocation', color='green', marker='o')
    
    plt.title('Optimal Allocations Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Allocation Percentage')
    plt.xticks(ticks=np.arange(len(allocations)), labels=np.arange(1, len(allocations) + 1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlim(left=0, right=len(allocations) - 1)  # Limit x-axis
    plt.ylim(0, 1)  # Ensure y-axis is between 0 and 1
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example of how to call the function (uncomment below to test)
# optimal_allocations_example = [
#     [0.4, 0.4, 0.2],
#     [0.5, 0.3, 0.2],
#     [0.3, 0.5, 0.2],
#     [0.6, 0.2, 0.2],
#     [0.2, 0.6, 0.2]
# ]
# create_visualization(optimal_allocations_example)
