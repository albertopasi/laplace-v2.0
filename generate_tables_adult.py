import os
import numpy as np
import pandas as pd


def generate_table():
    """
    Loads all .npy results from the results/Adult directory,
    processes them, and prints a Markdown table.
    """
    results_dir = './results/Adult/'

    # Define the order and display names for the table rows
    scenarios = {
        'MAP Baseline': 'map_baseline.npy',
        'Laplace LL Baseline': 'laplace_ll_baseline.npy',
        'Shift: Male-to-Female': 'shift_male_to_female.npy',
        'Shift: Female-to-Male': 'shift_female_to_male.npy',
        'Noise Intensity: 0.1': 'noise_0.1.npy',
        'Noise Intensity: 0.25': 'noise_0.25.npy',
        'Noise Intensity: 0.5': 'noise_0.5.npy',
        'Noise Intensity: 0.75': 'noise_0.75.npy',
        'Noise Intensity: 1.0': 'noise_1.0.npy'
    }

    processed_results = []

    print("Loading and processing results...")
    for name, filename in scenarios.items():
        file_path = os.path.join(results_dir, filename)

        if not os.path.exists(file_path):
            print(f"Warning: Could not find result file {filename}")
            continue

        # Load the numpy file
        raw_data = np.load(file_path, allow_pickle=True)

        # The data is saved as an array of one dictionary, e.g. [{'acc': 0.85, ...}]
        # So we extract the first element.
        metrics = raw_data[0]

        # Extract the metrics you care about for the table
        result_row = {
            'Experiment': name,
            'Accuracy': metrics.get('acc', float('nan')),
            'NLL': metrics.get('nll', float('nan')),
            'Confidence': metrics.get('conf', float('nan')),
            'Test Time (s)': metrics.get('test_time', float('nan'))
        }
        processed_results.append(result_row)

    if not processed_results:
        print("No result files found. Cannot generate table.")
        return

    # Create a pandas DataFrame for easy formatting
    df = pd.DataFrame(processed_results)

    # Format the numbers to 4 decimal places for clarity
    for col in ['Accuracy', 'NLL', 'Confidence', 'Test Time (s)']:
        df[col] = df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

    # Convert the DataFrame to a Markdown table and print
    markdown_table = df.to_markdown(index=False)

    print("\n--- Results Table ---")
    print(markdown_table)
    print("\n")


if __name__ == '__main__':
    generate_table()