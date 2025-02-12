import numpy as np
import pandas as pd
import os
import random
import warnings
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# ------------------------------
# Configuration Functions
# ------------------------------
def configure_tf():
    """Check for GPU availability and set environment variables."""
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU is available. GPU detected: {gpu_devices}")
    else:
        print("No GPU found. Running on CPU.")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def set_global_config(seed=42):
    """Set random seeds and suppress warnings for reproducibility."""
    warnings.filterwarnings('ignore')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ------------------------------
# Data Loading and Scaling
# ------------------------------
def load_data(data_path):
    """
    Load CSV data, clean infinities/NaNs, and return the 'Close' price as a numpy array.
    """
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df['Close'].values

def scale_data(train_raw, test_raw):
    """Scale train and test data independently using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1))
    test_scaled = scaler.transform(test_raw.reshape(-1, 1))
    return train_scaled, test_scaled

# ------------------------------
# Sliding Windows Generation
# ------------------------------
def generate_sliding_windows():
    """
    Generate sliding window configurations based on data proportions.
    Each dictionary specifies the training and testing splits.
    """
    return [
        {'type': 'window_1', 'train': (0.0, 0.3), 'test': (0.3, 0.4)},
        {'type': 'window_2', 'train': (0.3, 0.6), 'test': (0.6, 0.7)},
        {'type': 'window_3', 'train': (0.6, 0.9), 'test': (0.9, 1.0)}
    ]

# ------------------------------
# Dataset Creation for Regression
# ------------------------------
def create_dataset_regression(dataset: np.ndarray, look_back: int = 1, forecast_horizon: int = 1) -> tuple:
    """
    Create sequences (X) and corresponding targets (y) for LSTM training.

    Parameters:
      dataset: Scaled numpy array of the feature values.
      look_back: Number of past time steps to include.
      forecast_horizon: Number of future time steps to predict.

    Returns:
      X: Array of input sequences of shape (samples, look_back)
      y: Array of targets of shape (samples, forecast_horizon)
    """
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X_seq = dataset[i: i + look_back, 0]
        y_seq = dataset[i + look_back: i + look_back + forecast_horizon, 0]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)

# ------------------------------
# Model Building for Regression
# ------------------------------
def build_regression_model(look_back: int, units: int, forecast_horizon: int, learning_rate: float) -> Sequential:
    """
    Build a simple LSTM model for regression.

    Parameters:
      look_back: Number of past time steps used as input.
      units: Number of units in the LSTM layer.
      forecast_horizon: Number of future steps to predict.
      learning_rate: Learning rate for the optimizer.

    Returns:
      Compiled LSTM model.
    """
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(units, return_sequences=False),
        Dense(forecast_horizon)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model

# ------------------------------
# Training and Evaluation
# ------------------------------
def train_and_evaluate_regression(train_data: np.ndarray, test_data: np.ndarray,
                                  forecast_horizon: int, window_type: str, hyperparams: dict) -> dict:
    """
    Train and evaluate the regression model for a specific sliding window.

    Parameters:
      train_data: Scaled training data.
      test_data: Scaled testing data.
      forecast_horizon: Forecast horizon.
      window_type: Label for the current sliding window.
      hyperparams: Dictionary of hyperparameters.

    Returns:
      Dictionary of evaluation metrics and hyperparameters for the window.
    """
    look_back = hyperparams['look_back']
    units = hyperparams['units']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epochs = hyperparams['epochs']

    # Create regression datasets
    X_train, y_train = create_dataset_regression(train_data, look_back, forecast_horizon)
    X_test, y_test = create_dataset_regression(test_data, look_back, forecast_horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[Regression - {window_type}] Insufficient data for the given parameters.")
        return None

    # Reshape for LSTM: [samples, look_back, features]
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    # Build and train the model
    model = build_regression_model(look_back, units, forecast_horizon, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    epochs_run = len(history.history['loss'])

    # Predict and compute evaluation metrics
    y_pred_test = model.predict(X_test, verbose=0)
    y_test_flat = y_test.flatten()
    y_pred_test_flat = y_pred_test.flatten()

    mse = mean_squared_error(y_test_flat, y_pred_test_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_test_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, y_pred_test_flat)

    # Calculate prediction intervals using training residuals
    y_train_pred = model.predict(X_train, verbose=0)
    residuals = y_train.flatten() - y_train_pred.flatten()
    sigma = np.std(residuals)
    z = 1.96  # 95% confidence interval
    lower_bound = y_pred_test_flat - z * sigma
    upper_bound = y_pred_test_flat + z * sigma
    coverage = np.mean((y_test_flat >= lower_bound) & (y_test_flat <= upper_bound)) * 100
    interval_width = np.mean(upper_bound - lower_bound)

    result = {
        'type': window_type,
        'forecast_horizon': forecast_horizon,
        'look_back': look_back,
        'units': units,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs_run': epochs_run,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'coverage_probability (%)': coverage,
        'interval_width': interval_width
    }

    print(
        f"[Regression - {window_type}] Final metrics (horizon={forecast_horizon}): "
        f"MSE={mse:.9f}, MAE={mae:.9f}, RMSE={rmse:.9f}, R2={r2:.9f}, "
        f"Coverage={coverage:.2f}%, IntervalWidth={interval_width:.5f}, "
        f"EpochsRun={epochs_run}"
    )
    return result

# ------------------------------
# Process Each Sliding Window for Regression
# ------------------------------
def process_window_regression(window_config: dict, data: np.ndarray, forecast_horizon: int,
                              hyperparams: dict) -> dict:
    """
    For a given sliding window configuration, this function:
      - Extracts the training and testing data from the full dataset,
      - Applies independent scaling,
      - Runs the training and evaluation for regression.

    Returns:
      A dictionary of regression metrics for the window.
    """
    n_samples = len(data)
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    test_start = int(window_config['test'][0] * n_samples)
    test_end = int(window_config['test'][1] * n_samples)

    train_raw = data[train_start:train_end]
    test_raw = data[test_start:test_end]

    if len(train_raw) < hyperparams['look_back'] or len(test_raw) < hyperparams['look_back']:
        print(f"Skipping {window_config['type']} - insufficient data")
        return None

    train_scaled, test_scaled = scale_data(train_raw, test_raw)
    return train_and_evaluate_regression(train_scaled, test_scaled, forecast_horizon,
                                         window_config['type'], hyperparams)

# ------------------------------
# Save Results
# ------------------------------
def save_results_regression(results: list, csv_path: str):
    """Save regression results to CSV, including an extra row with average metrics."""
    if results:
        results_df = pd.DataFrame(results)
        numeric_cols = results_df.select_dtypes(include=np.number).columns
        avg_row = results_df[numeric_cols].mean().to_dict()
        avg_row['type'] = 'average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print("\nFinal regression results (including average metrics):")
        cols_to_show = ['type', 'mse', 'mae', 'rmse', 'r2_score', 'coverage_probability (%)', 'interval_width']
        print(results_df[cols_to_show].to_string(index=False))
    else:
        print("[Results] No regression results to save.")

# ------------------------------
# Main Entry Point
# ------------------------------
def main():
    # Define file paths and hyperparameters
    DATA_PATH = './../../../../data/raw/USDEUR=X_max_1d.csv'
    REGRESSION_CSV_PATH = './results/regression_results1.csv'
    FORECAST_HORIZONS_REG = [1]

    hyperparams = {
        'look_back': 60,
        'units': 128,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 10
    }

    # Configure TensorFlow and set seeds
    configure_tf()
    set_global_config(seed=42)

    # Load the data and generate sliding window configurations
    data = load_data(DATA_PATH)
    windows = generate_sliding_windows()

    results = []
    start_time = time.time()
    for window in windows:
        print(f"\n=== Processing {window['type']} ===")
        print(f"Training range: {window['train'][0] * 100:.1f}% - {window['train'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        for horizon in FORECAST_HORIZONS_REG:
            result = process_window_regression(window, data, horizon, hyperparams)
            if result is not None:
                results.append(result)

    # Save all regression results
    save_results_regression(results, REGRESSION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
