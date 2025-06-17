import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Configuration ---
RESULTS_DIR = './results/Adult/'
SEEDS = [6, 12, 13, 523, 972394]

# Define the experiments and their human-readable names
# This dictionary maps a display name to a file prefix
EXPERIMENTS = {
    'Baselines': [
        ('MAP', 'map_baseline'),
        ('LA', 'laplace_ll_baseline'),
        ('LA*', 'laplace_star_baseline'),
        ('Subspace LA', 'subspace_baseline'),
        ('SWAG-Laplace', 'swag_laplace_baseline'),
    ],
    'Domain Shift': [
        ('Shift: Male-to-Female', 'shift_male_to_female'),
        ('Shift: Female-to-Male', 'shift_female_to_male'),
    ],
    'Noise Intensity': [
        ('Noise: 0.1', 'noise_0.1'),
        ('Noise: 0.25', 'noise_0.25'),
        ('Noise: 0.5', 'noise_0.5'),
        ('Noise: 0.75', 'noise_0.75'),
        ('Noise: 1.0', 'noise_1.0'),
    ]
}


def calculate_ece(metrics):
    """A placeholder for ECE calculation if it's not in the file."""
    # In the actual codebase, this would be properly calculated.
    # For now, we'll return a placeholder if not found.
    return metrics.get('ece', np.nan)


def load_and_aggregate_results():
    """
    Loads all .npy result files, groups them by experiment,
    and computes the mean and standard deviation of metrics across seeds.
    """
    aggregated_results = dict()

    for category, experiments in EXPERIMENTS.items():
        for display_name, file_prefix in experiments:
            experiment_metrics = defaultdict(list)

            for seed in SEEDS:
                filename = f"{file_prefix}_seed{seed}.npy"
                file_path = os.path.join(RESULTS_DIR, filename)

                if os.path.exists(file_path):
                    try:
                        raw_data = np.load(file_path, allow_pickle=True)
                        metrics = raw_data[0]
                        experiment_metrics['Accuracy'].append(metrics.get('acc', np.nan))
                        experiment_metrics['NLL'].append(metrics.get('nll', np.nan))
                        experiment_metrics['ECE'].append(calculate_ece(metrics))  # Add ECE
                    except Exception as e:
                        print(f"Warning: Could not load or process {filename}. Error: {e}")
                else:
                    print(f"Warning: Missing file for seed {seed}: {filename}")

            # Compute mean and std dev if we have any results
            if experiment_metrics:
                summary = dict()
                for metric, values in experiment_metrics.items():
                    valid_values = [v for v in values if not np.isnan(v)]
                    if valid_values:
                        summary[f'{metric}_mean'] = np.mean(valid_values)
                        summary[f'{metric}_std'] = np.std(valid_values)
                    else:
                        summary[f'{metric}_mean'] = np.nan
                        summary[f'{metric}_std'] = np.nan
                aggregated_results[(category, display_name)] = summary

    return aggregated_results


def format_table(results_df, title):
    """Formats and prints a DataFrame as a Markdown table."""
    print(f"\n--- {title} ---")
    if results_df.empty:
        print("No results to display.")
        return

    # Format to 'mean ± std'
    for metric in ['Accuracy', 'NLL', 'ECE']:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col in results_df.columns and std_col in results_df.columns:
            results_df[metric] = results_df.apply(
                lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}"
                if pd.notna(row[mean_col]) else "N/A",
                axis=1
            )

    # Select and rename columns for display
    display_cols = ['Experiment', 'Accuracy', 'NLL', 'ECE']
    results_df = results_df[display_cols]

    print(results_df.to_markdown(index=False))


def create_plots(results_df):
    """Creates and saves plots similar to Figure 4."""
    print("\n--- Creating Plots ---")
    sns.set_theme(style="whitegrid")

    # Filter for necessary data
    noise_df = results_df[results_df['Category'] == 'Noise Intensity'].copy()
    shift_df = results_df[results_df['Category'] == 'Domain Shift']
    baseline_df = results_df[results_df['Category'] == 'Baselines']

    # Add baseline results to the shift_df for plotting comparison
    plot_shift_df = pd.concat([
        baseline_df[baseline_df['Experiment'] == 'LA'],
        shift_df
    ]).rename(columns={'Experiment': 'Condition'})

    # Plot 1: Noise Intensity vs. NLL and Accuracy
    if not noise_df.empty:
        noise_df['Intensity'] = noise_df['Experiment'].str.extract(r'(\d+\.\d+)').astype(float)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        sns.lineplot(data=noise_df, x='Intensity', y='NLL_mean', marker='o', ax=ax1)
        ax1.set_title('Shift Intensity vs. NLL')
        ax1.set_ylabel('Negative Log-Likelihood (NLL)')
        ax1.set_xlabel('Gaussian Noise Intensity')

        sns.lineplot(data=noise_df, x='Intensity', y='Accuracy_mean', marker='o', ax=ax2)
        ax2.set_title('Shift Intensity vs. Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Gaussian Noise Intensity')

        fig.suptitle('Performance under Noise-based Distribution Shift', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('adult_noise_intensity_plot.png')
        print("Saved noise intensity plot to adult_noise_intensity_plot.png")

    # Plot 2: Domain Shift Comparison
    if not plot_shift_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

        sns.barplot(data=plot_shift_df, x='Condition', y='Accuracy_mean', ax=ax1)
        ax1.set_title('Accuracy under Gender-based Domain Shift')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', rotation=15)

        sns.barplot(data=plot_shift_df, x='Condition', y='NLL_mean', ax=ax2, palette='viridis')
        ax2.set_title('NLL under Gender-based Domain Shift')
        ax2.set_ylabel('Negative Log-Likelihood (NLL)')
        ax2.set_xlabel('')
        ax2.tick_params(axis='x', rotation=15)

        fig.suptitle('Performance under Gender-based Domain Shift', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('adult_domain_shift_plot.png')
        print("Saved domain shift plot to adult_domain_shift_plot.png")


if __name__ == '__main__':
    all_results = load_and_aggregate_results()

    # Convert to DataFrame for easier manipulation
    results_list = []
    for (category, name), metrics in all_results.items():
        row = {'Category': category, 'Experiment': name, **metrics}
        results_list.append(row)

    df = pd.DataFrame(results_list)

    # Generate and print tables
    format_table(df[df['Category'] == 'Baselines'].copy(), "Baseline Method Comparison")
    format_table(df[df['Category'] == 'Domain Shift'].copy(), "Domain Shift Results")
    format_table(df[df['Category'] == 'Noise Intensity'].copy(), "Noise Intensity Results")

    # Generate and save plots
    create_plots(df)

