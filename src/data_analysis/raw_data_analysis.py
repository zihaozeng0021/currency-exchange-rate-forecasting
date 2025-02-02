import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def analyze_exchange_rate(file_path):
    # Create results directory if it doesn't exist
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    #df.set_index('Date', inplace=True) # Disable this line because it duplicates doesn't work well with it

    # Inspect Data
    print("\n--- Basic Information ---")
    print("DataFrame Shape:", df.shape)
    print("\nDataFrame Head:")
    print(df.head())
    print("\nDataFrame Tail:")
    print(df.tail())

    print("\n--- Summary Statistics ---")
    print(df.describe())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\n--- Missing Values ---")
    print(missing_values)

    # Check for duplicates
    duplicates = df.duplicated().sum()
    print("\n--- Duplicates ---")
    print(f"Number of duplicate rows: {duplicates}")


    # Check for outliers
    print("\n--- Outliers ---")
    Q1 = df['Close'].quantile(0.25)
    Q3 = df['Close'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['Close'] < (Q1 - 1.5 * IQR)) | (df['Close'] > (Q3 + 1.5 * IQR))]
    print(outliers)

    # Plot the time series with rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    rolling_mean = df['Close'].rolling(window=30).mean()
    rolling_std = df['Close'].rolling(window=30).std()
    plt.plot(df.index, rolling_mean, label='30-Day Rolling Mean', color='red')
    plt.plot(df.index, rolling_std, label='30-Day Rolling Std', color='green')
    plt.title("Exchange Rate Over Time with Rolling Statistics")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "time_series_with_rolling_stats.png"))
    plt.show()

    # distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Close'], kde=True, color='skyblue')
    plt.title("Distribution of Exchange Rate (Close)")
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "distribution_plot.png"))
    plt.show()

    # Box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['Close'], color='lightgreen')
    plt.title("Box Plot of Exchange Rate (Close)")
    plt.xlabel("Close Price")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "box_plot.png"))
    plt.show()

    # ADF Stationarity Check
    close_series = df['Close'].dropna()
    adf_result = adfuller(close_series)
    adf_stat, p_value, usedlag, nobs, critical_values, icbest = adf_result
    print("\n--- Augmented Dickey-Fuller Test ---")
    print(f"ADF Statistic: {adf_stat:.6f}")
    print(f"P-Value: {p_value:.6f}")
    print("Critical Values:")
    for key, val in critical_values.items():
        print(f"   {key}: {val:.6f}")

    # ACF and PACF Plots
    plt.figure(figsize=(10, 6))
    plot_acf(close_series, lags=50)
    plt.title("ACF of Close Price")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "acf_plot.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    plot_pacf(close_series, lags=50, method='ywm')
    plt.title("PACF of Close Price")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pacf_plot.png"))
    plt.show()

    # Seasonal Decomposition
    try:
        decomposition = seasonal_decompose(close_series, model='additive', period=30)
        decomposition.plot()
        plt.suptitle("Seasonal Decomposition of Close Price", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "seasonal_decomposition.png"))
        plt.show()
    except Exception as e:
        print("Seasonal Decomposition could not be performed:", e)

    # Lag Plot
    plt.figure(figsize=(8, 6))
    pd.plotting.lag_plot(close_series, lag=1)
    plt.title("Lag Plot of Close Price")
    plt.xlabel("Close Price (t)")
    plt.ylabel("Close Price (t+1)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lag_plot.png"))
    plt.show()

if __name__ == "__main__":
    FILE_PATH = "../../data/raw/USDEUR=X_max_1d.csv"
    analyze_exchange_rate(FILE_PATH)
