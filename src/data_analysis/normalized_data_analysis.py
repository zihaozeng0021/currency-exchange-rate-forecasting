import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

sns.set_style('whitegrid')

# Ensure results directory exists
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def min_max_scale(series):
    """ Min-Max scale a pandas Series to [0, 1]. """
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val)


def z_score_scale(series):
    """ Z-score standardization (mean=0, std=1). """
    mean_val = series.mean()
    std_val = series.std()
    return (series - mean_val) / std_val


def log_transform(series):
    """ Log transform. Assumes all values are > 0. """
    return np.log(series)


def unit_length_scale(series):
    """ Normalize data to have unit length (L2 norm = 1). """
    norm = np.sqrt((series ** 2).sum())
    return series / norm if norm != 0 else series


def adf_test(series):
    """ Runs Augmented Dickey-Fuller test on a series; returns dict. """
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value, usedlag, nobs, critical_values, icbest = result
    return {
        'adf_stat': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values
    }


def print_distribution_stats(name, series):
    """ Print mean, std, skewness, kurtosis for the series. """
    print(f"\n--- {name} Distribution Stats ---")
    print(f"Mean:       {series.mean():.4f}")
    print(f"Std Dev:    {series.std():.4f}")
    print(f"Skewness:   {series.skew():.4f}")
    print(f"Kurtosis:   {series.kurtosis():.4f}")


def plot_distribution(name, series):
    """ Plot histogram + KDE for the series and save the plot. """
    plt.figure(figsize=(8, 4))
    sns.histplot(series.dropna(), kde=True, color='blue', edgecolor='black')
    plt.title(f"{name} - Distribution (Histogram + KDE)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()

    # Save the figure
    file_path = os.path.join(RESULTS_DIR, f"{name.replace(' ', '_').lower()}_distribution.png")
    plt.savefig(file_path)
    plt.close()


def analyze_transformations(file_path):
    # --- Load data ---
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df.sort_index(inplace=True)

    close_series = df['Close'].dropna()
    print(f"Data loaded: {close_series.shape[0]} rows.")

    # --- Original Series Stats & ADF ---
    print_distribution_stats("Original Series", close_series)
    orig_adf = adf_test(close_series)
    print("\n--- Original Series ADF Test ---")
    print(f"ADF Statistic: {orig_adf['adf_stat']:.6f}")
    print(f"P-Value:       {orig_adf['p_value']:.6f}")
    for key, val in orig_adf['critical_values'].items():
        print(f"   {key}: {val:.6f}")
    plot_distribution("Original Series", close_series)

    # Prepare a dictionary to store each transformed series
    transformations = {}

    # --- Min-Max Normalization ---
    mm_series = min_max_scale(close_series)
    transformations["Min-Max"] = mm_series
    print_distribution_stats("Min-Max Normalized", mm_series)
    plot_distribution("Min-Max Normalized", mm_series)

    # --- Z-score Standardization ---
    z_series = z_score_scale(close_series)
    transformations["Z-score"] = z_series
    print_distribution_stats("Z-score Standardized", z_series)
    plot_distribution("Z-score Standardized", z_series)

    # --- Log Transform ---
    if (close_series <= 0).any():
        print("\nLog Transformation not applicable (non-positive values found).")
        log_series = None
    else:
        log_series = log_transform(close_series)
        transformations["Log"] = log_series
        print_distribution_stats("Log-Transformed", log_series)
        plot_distribution("Log-Transformed", log_series)

    # --- Unit-Length Normalization ---
    unit_series = unit_length_scale(close_series)
    transformations["Unit-Length"] = unit_series
    print_distribution_stats("Unit-Length Normalized", unit_series)
    plot_distribution("Unit-Length Normalized", unit_series)

    # --- Overlay all transformations with the original series ---
    plt.figure(figsize=(12, 6))
    plt.plot(close_series.index, close_series, label='Original', lw=2)
    for t_name, t_series in transformations.items():
        if t_series is not None:
            plt.plot(t_series.index, t_series, label=t_name)
    plt.title("Overlay: Original vs. Transformed Series")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    # Save the overlay plot
    overlay_path = os.path.join(RESULTS_DIR, "transformed_series_overlay.png")
    plt.savefig(overlay_path)
    plt.close()


if __name__ == "__main__":
    FILE_PATH = "../../data/raw/USDEUR=X_max_1d.csv"
    analyze_transformations(FILE_PATH)
