import pandas as pd

def generate_report(optimal_allocations_df):
    """
    Generate a summary report for the optimal portfolio allocations.

    Parameters:
    - optimal_allocations_df: DataFrame containing the optimal allocations.

    Returns:
    - report: Dictionary containing the summary of the allocations.
    """

    # Ensure the input DataFrame has the correct columns
    if not {'Stock_Allocation', 'Bond_Allocation', 'Commodity_Allocation'}.issubset(optimal_allocations_df.columns):
        raise ValueError("DataFrame must contain 'Stock_Allocation', 'Bond_Allocation', and 'Commodity_Allocation' columns.")

    # Calculate summary statistics
    report = {
        "Total_Allocations": optimal_allocations_df.sum().to_dict(),
        "Average_Allocations": optimal_allocations_df.mean().to_dict(),
        "Optimal_Allocations": optimal_allocations_df.iloc[-1].to_dict()  # Get the last row for optimal allocations
    }

    return report
