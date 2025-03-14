import numpy as np
import pandas as pd
import os
import random
import warnings
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

import optuna


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define multiple currency pairs with their corresponding CSV file paths
    currency_pairs = {
        'AUDUSD': 'data/AUDUSD.csv',
        'EURUSD': 'data/EURUSD.csv',
        'GBPUSD': 'data/GBPUSD.csv',
        'JPYUSD': 'data/JPYUSD.csv',
        'NZDUSD': 'data/NZDUSD.csv'
    }
    AGGREGATED_REGRESSION_CSV_PATH = 'results/multivariate_LSTM.csv'
    FORECAST_HORIZONS_REG = [1]

    search_space = {
        'epochs': (10, 50),
        'look_back': (30, 120),
        'units': (50, 200),
        'batch_size': (16, 128),
        'learning_rate': (0.0001, 0.01)
    }

    # Configure TensorFlow and set seeds
    configure_tf()
    set_global_config(seed=42)

    all_results = []
    start_time = time.time()

    # Loop over each currency pair
    for pair_name, data_path in currency_pairs.items():
        print(f"\n=== Processing currency pair: {pair_name} ===")
        data = load_data(data_path)  # Merges currency data with FTSE data based on Date
        data = smooth_data(data, window=30)
        train_raw, test_raw = split_data(data, train_ratio=0.8)
        # Note: No need to reshape as data is multivariate
        train_scaled, test_scaled, scaler = apply_standard_scaling(train_raw, test_raw)

        # Determine number of features (should be 2: Currency_Close and FTSE_Close)
        num_features = train_scaled.shape[1]

        time_limit_for_search = 3600 if pair_name in ('EURUSD', 'GBPUSD') else 600

        # Loop over forecast horizons
        for horizon in FORECAST_HORIZONS_REG:
            print(f"\n=== Processing FORECAST_HORIZON {horizon} for {pair_name} ===")
            best_hp = bayesian_search_hyperparameters(train_scaled, test_scaled, scaler, horizon,
                                                      search_space, num_features, time_limit=time_limit_for_search)
            result = train_and_evaluate_regression(train_scaled, test_scaled, horizon, "80-20 split",
                                                   best_hp, scaler, num_features)
            if result is not None:
                result['best_hyperparameters'] = best_hp
                result['currency_pair'] = pair_name  # Record the currency pair in the results
                all_results.append(result)

    # Save aggregated results for all pairs in one file with the currency pair as the first column.
    save_results_regression(all_results, AGGREGATED_REGRESSION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


# ==============================================================================
# Global Configuration and Hyperparameters
# ==============================================================================
def configure_tf():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU is available. GPU detected: {gpu_devices}")
    else:
        print("No GPU found. Running on CPU.")


def set_global_config(seed=42):
    warnings.filterwarnings('ignore')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    optuna.samplers.TPESampler(seed=seed)


# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================
def load_data(data_path):
    df_currency = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df_ftse = pd.read_csv('data/ftse.csv', index_col='Date', parse_dates=True)
    df_currency = df_currency.rename(columns={"Close": "Currency_Close"})
    df_ftse = df_ftse.rename(columns={"Close": "FTSE_Close"})
    df_merged = df_currency.join(df_ftse[['FTSE_Close']], how='inner')
    df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna()
    return df_merged[['Currency_Close', 'FTSE_Close']].values


def smooth_data(data, window=30):
    if data.ndim == 1:
        data_series = pd.Series(data)
        smoothed_series = data_series.rolling(window=window, min_periods=1).mean()
        return smoothed_series.values
    else:
        df = pd.DataFrame(data)
        smoothed_df = df.rolling(window=window, min_periods=1).mean()
        return smoothed_df.values


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


def inverse_standard_scaling(scaled_data, scaler, col_index=0):
    """
    Inverse transform for a single column from a StandardScaler fitted on multivariate data.
    scaled_data: np.ndarray of shape (n_samples, 1) or (n_samples,)
    Applies the inverse transformation using parameters from the specified column (default is 0).
    """
    return scaled_data * scaler.scale_[col_index] + scaler.mean_[col_index]


# ==============================================================================
# Dataset Creation for Regression (Multivariate)
# ==============================================================================
def create_dataset_regression(dataset, look_back=1, forecast_horizon=1):
    """
    Create dataset for multivariate regression.
    X consists of all features over the look_back window.
    y is the target series (first column: Currency_Close).
    """
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X_seq = dataset[i: i + look_back, :]  # All features
        y_seq = dataset[i + look_back: i + look_back + forecast_horizon, 0]  # Only Currency_Close as target
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)


# ==============================================================================
# Model Building for Regression (Multivariate)
# ==============================================================================
def build_regression_model(look_back, num_features, units, forecast_horizon, learning_rate):
    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(units, return_sequences=False),
        Dense(forecast_horizon, activation='linear')
    ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


# ==============================================================================
# Training and Evaluation (Regression)
# ==============================================================================
def train_and_evaluate_regression(train_data: np.ndarray, test_data: np.ndarray,
                                  forecast_horizon, split_type, hyperparams, scaler, num_features):
    look_back = hyperparams['look_back']
    units = hyperparams['units']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epochs = hyperparams['epochs']

    # Create regression datasets in scaled space
    X_train, y_train = create_dataset_regression(train_data, look_back, forecast_horizon)
    X_test, y_test = create_dataset_regression(test_data, look_back, forecast_horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[Regression] Insufficient data for the given parameters.")
        return None

    # X_train and X_test are already in shape (samples, look_back, num_features)

    # Build and train the model
    model = build_regression_model(look_back, num_features, units, forecast_horizon, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    epochs_run = len(history.history['loss'])

    # Predict in scaled space
    y_pred_test = model.predict(X_test, verbose=0)

    # Inverse transform predictions and true targets back to original scale for the target column
    y_pred_test_orig = inverse_standard_scaling(y_pred_test, scaler, col_index=0)
    y_test_orig = inverse_standard_scaling(y_test, scaler, col_index=0)

    # Compute evaluation metrics on original scale
    y_test_flat = y_test_orig.flatten()
    y_pred_test_flat = y_pred_test_orig.flatten()
    mse_val = mean_squared_error(y_test_flat, y_pred_test_flat)
    mae_val = mean_absolute_error(y_test_flat, y_pred_test_flat)
    rmse_val = np.sqrt(mse_val)
    r2 = r2_score(y_test_flat, y_pred_test_flat)

    # Calculate prediction intervals using training residuals (on original scale)
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_orig = inverse_standard_scaling(y_train, scaler, col_index=0)
    y_train_pred_orig = inverse_standard_scaling(y_train_pred, scaler, col_index=0)
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
        'units': units,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs_run': epochs_run,
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
        f"Coverage={coverage:.2f}%, IntervalWidth={interval_width:.5f}, "
        f"EpochsRun={epochs_run}"
    )
    return result


# ==============================================================================
# Bayesian Optimization for Hyperparameters using Optuna
# ==============================================================================
def bayesian_search_hyperparameters(train_scaled, test_scaled, scaler, forecast_horizon, search_space,
                                    num_features, time_limit=600):
    def objective(trial):
        # Sample hyperparameters using the intervals from search_space
        epochs = trial.suggest_int('epochs', search_space['epochs'][0], search_space['epochs'][1])
        look_back = trial.suggest_int('look_back', search_space['look_back'][0], search_space['look_back'][1])
        units = trial.suggest_int('units', search_space['units'][0], search_space['units'][1])
        batch_size = trial.suggest_int('batch_size', search_space['batch_size'][0], search_space['batch_size'][1])
        learning_rate = trial.suggest_float('learning_rate', search_space['learning_rate'][0],
                                            search_space['learning_rate'][1])

        candidate = {
            'epochs': epochs,
            'look_back': look_back,
            'units': units,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        print(f"Evaluating candidate: {candidate}")

        X_train, y_train = create_dataset_regression(train_scaled, look_back, forecast_horizon)
        X_test, y_test = create_dataset_regression(test_scaled, look_back, forecast_horizon)
        if len(X_train) == 0 or len(X_test) == 0:
            # Return a large error if there is insufficient data
            return 1e6

        # Build model with the current candidate hyperparameters
        model = build_regression_model(look_back, num_features, units, forecast_horizon, learning_rate)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        # Inverse transform for the target column only
        y_test_pred_orig = inverse_standard_scaling(y_test_pred, scaler, col_index=0)
        y_test_orig = inverse_standard_scaling(y_test, scaler, col_index=0)
        mse_val = mean_squared_error(y_test_orig.flatten(), y_test_pred_orig.flatten())
        return mse_val

    study = optuna.create_study(direction="minimize")
    # Optimize until the time limit is reached.
    study.optimize(objective, timeout=time_limit)
    best_hp = study.best_trial.params
    return best_hp


# ==============================================================================
# Save Results (Regression)
# ==============================================================================
def save_results_regression(results, csv_path):
    if results:
        results_df = pd.DataFrame(results)
        # Reorder columns so that 'currency_pair' is the first column
        cols = results_df.columns.tolist()
        if 'currency_pair' in cols:
            cols.remove('currency_pair')
            cols = ['currency_pair'] + cols
            results_df = results_df[cols]
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print(f"\nFinal regression results saved to {csv_path}:")
        print(results_df.to_string(index=False))
    else:
        print("[Results] No regression results to save.")


if __name__ == "__main__":
    main()
