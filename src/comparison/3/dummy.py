import numpy as np
import pandas as pd
import os
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define file paths and parameters
    DATA_PATH = 'data/ERU_USD_10min.csv'
    REGRESSION_CSV_PATH = 'results/dummy_model_10min.csv'
    FORECAST_HORIZONS_REG = [1]

    # Dummy hyperparameter: using a fixed look_back window.
    dummy_hp = {'look_back': 30}

    data = load_data(DATA_PATH)
    train_raw, test_raw = split_data(data, train_ratio=0.8)
    train_raw = train_raw.reshape(-1, 1)
    test_raw = test_raw.reshape(-1, 1)
    train_scaled, test_scaled, scaler = apply_standard_scaling(train_raw, test_raw)

    results = []
    start_time = time.time()

    for horizon in FORECAST_HORIZONS_REG:
        print(f"\n=== Processing FORECAST_HORIZON {horizon} ===")
        result = train_and_evaluate_regression(train_scaled, test_scaled, horizon, "80-20 split", dummy_hp, scaler)
        if result is not None:
            results.append(result)

    save_results_regression(results, REGRESSION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================
def load_data(data_path):
    df = pd.read_csv(data_path, index_col='DateTime', parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df['Close'].values


def split_data(data, train_ratio=0.8):
    n_samples = len(data)
    split_index = int(train_ratio * n_samples)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


def apply_standard_scaling(train_data, test_data):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler


def inverse_standard_scaling(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)


# ==============================================================================
# Dataset Creation for Regression
# ==============================================================================
def create_dataset_regression(dataset, look_back=1, forecast_horizon=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X_seq = dataset[i: i + look_back, 0]
        y_seq = dataset[i + look_back: i + look_back + forecast_horizon, 0]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)


# ==============================================================================
# Dummy Model: Forecast using the last observed value
# ==============================================================================
def train_and_evaluate_regression(train_data: np.ndarray, test_data: np.ndarray,
                                  forecast_horizon, split_type, hyperparams, scaler):
    look_back = hyperparams['look_back']

    # Create regression datasets in scaled space
    X_train, y_train = create_dataset_regression(train_data, look_back, forecast_horizon)
    X_test, y_test = create_dataset_regression(test_data, look_back, forecast_horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        print("[Regression] Insufficient data for the given parameters.")
        return None

    # Dummy forecast: use the last value in each input sequence as the prediction.
    y_pred_test = X_test[:, -1].reshape(-1, forecast_horizon)
    y_train_pred = X_train[:, -1].reshape(-1, forecast_horizon)

    # Inverse transform predictions and targets back to original scale.
    y_pred_test_orig = inverse_standard_scaling(y_pred_test, scaler)
    y_test_orig = inverse_standard_scaling(y_test, scaler)
    y_train_pred_orig = inverse_standard_scaling(y_train_pred, scaler)
    y_train_orig = inverse_standard_scaling(y_train, scaler)

    # Compute evaluation metrics on original scale.
    y_test_flat = y_test_orig.flatten()
    y_pred_test_flat = y_pred_test_orig.flatten()
    mse_val = mean_squared_error(y_test_flat, y_pred_test_flat)
    mae_val = mean_absolute_error(y_test_flat, y_pred_test_flat)
    rmse_val = np.sqrt(mse_val)
    r2 = r2_score(y_test_flat, y_pred_test_flat)

    # Calculate prediction intervals using training residuals (on original scale).
    residuals = y_train_orig.flatten() - y_train_pred_orig.flatten()
    sigma = np.std(residuals)
    z = 1.96  # 95% confidence interval
    lower_bound = y_pred_test_flat - z * sigma
    upper_bound = y_pred_test_flat + z * sigma
    coverage = np.mean((y_test_flat >= lower_bound) & (y_test_flat <= upper_bound)) * 100
    interval_width = np.mean(upper_bound - lower_bound)

    result = {
        'forecast_horizon': forecast_horizon,
        'look_back': look_back,
        'mse': mse_val,
        'mae': mae_val,
        'rmse': rmse_val,
        'r2_score': r2,
        'coverage_probability (%)': coverage,
        'interval_width': interval_width
    }

    print(
        f"[Regression] Final metrics (horizon={forecast_horizon}): "
        f"MSE={mse_val:.9f}, MAE={mae_val:.9f}, RMSE={rmse_val:.9f}, R2={r2:.9f}, "
        f"Coverage={coverage:.2f}%, IntervalWidth={interval_width:.5f}"
    )
    return result


# ==============================================================================
# Save Results (Regression)
# ==============================================================================
def save_results_regression(results, csv_path):
    if results:
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print("\nFinal regression results:")
        cols_to_show = ['mse', 'mae', 'rmse', 'r2_score', 'coverage_probability (%)', 'interval_width']
        print(results_df[cols_to_show].to_string(index=False))
    else:
        print("[Results] No regression results to save.")


if __name__ == "__main__":
    main()
