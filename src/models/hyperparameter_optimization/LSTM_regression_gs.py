import numpy as np
import pandas as pd
import os
import random
import warnings
import time
import itertools

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define file paths and hyperparameters
    DATA_PATH = '../../../data/raw/USDEUR=X_max_1d.csv'
    REGRESSION_CSV_PATH = 'results/LSTM_regression_gs.csv'
    FORECAST_HORIZONS_REG = [1]

    # Define search space for grid search
    search_space = {
        'epochs': [10, 20, 30, 40, 50],
        'look_back': [30, 60, 90, 120],
        'units': [50, 100, 150, 200],
        'batch_size': [16, 32, 48, 64, 80, 96, 112, 128],
        'learning_rate': [0.0001, 0.005, 0.01]
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
        print(f"Validation range: {window['validate'][0] * 100:.1f}% - {window['validate'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        for horizon in FORECAST_HORIZONS_REG:
            result = process_window_regression(window, data, horizon, search_space)
            if result is not None:
                results.append(result)

    # Save all regression results
    save_results_regression(results, REGRESSION_CSV_PATH)
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


# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================
def load_data(data_path):
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df['Close'].values


# ==============================================================================
# Z-Score Scaling for train and test data
# ==============================================================================
def apply_standard_scaling(train_data, test_data):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler


# ==============================================================================
# Inverse Z-Score Scaling
# ==============================================================================
def inverse_standard_scaling(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)


# ==============================================================================
# Sliding Windows
# ==============================================================================
def generate_sliding_windows():
    return [
        {'type': 'window_1', 'train': (0.0, 0.12), 'validate': (0.12, 0.16), 'test': (0.16, 0.2)},
        {'type': 'window_2', 'train': (0.16, 0.28), 'validate': (0.28, 0.32), 'test': (0.32, 0.36)},
        {'type': 'window_3', 'train': (0.32, 0.44), 'validate': (0.44, 0.48), 'test': (0.48, 0.52)},
        {'type': 'window_4', 'train': (0.48, 0.60), 'validate': (0.60, 0.64), 'test': (0.64, 0.68)},
        {'type': 'window_5', 'train': (0.64, 0.76), 'validate': (0.76, 0.80), 'test': (0.8, 0.84)},
        {'type': 'window_6', 'train': (0.8, 0.92), 'validate': (0.92, 0.96), 'test': (0.96, 1.0)}
    ]


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
# Model Building for Regression
# ==============================================================================
def build_regression_model(look_back, units, forecast_horizon, learning_rate):
    model = Sequential([
        Input(shape=(look_back, 1)),
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
                                  forecast_horizon, window_type, hyperparams, scaler):
    look_back = hyperparams['look_back']
    units = hyperparams['units']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epochs = hyperparams['epochs']

    # Create regression datasets in scaled space
    X_train, y_train = create_dataset_regression(train_data, look_back, forecast_horizon)
    X_test, y_test = create_dataset_regression(test_data, look_back, forecast_horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[Regression - {window_type}] Insufficient data for the given parameters.")
        return None

    # Reshape data for LSTM input: (samples, look_back, features)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    # Build and train the model
    model = build_regression_model(look_back, units, forecast_horizon, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    epochs_run = len(history.history['loss'])

    # Predict in scaled space
    y_pred_test = model.predict(X_test, verbose=0)

    # Inverse transform predictions and true targets back to original scale
    y_pred_test_orig = inverse_standard_scaling(y_pred_test, scaler)
    y_test_orig = inverse_standard_scaling(y_test, scaler)

    # Compute evaluation metrics on original scale
    y_test_flat = y_test_orig.flatten()
    y_pred_test_flat = y_pred_test_orig.flatten()
    mse_val = mean_squared_error(y_test_flat, y_pred_test_flat)
    mae_val = mean_absolute_error(y_test_flat, y_pred_test_flat)
    rmse_val = np.sqrt(mse_val)
    r2 = r2_score(y_test_flat, y_pred_test_flat)

    # Calculate prediction intervals using training residuals (on original scale)
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_orig = inverse_standard_scaling(y_train, scaler)
    y_train_pred_orig = inverse_standard_scaling(y_train_pred, scaler)
    residuals = y_train_orig.flatten() - y_train_pred_orig.flatten()
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
        'mse': mse_val,
        'mae': mae_val,
        'rmse': rmse_val,
        'r2_score': r2,
        'coverage_probability (%)': coverage,
        'interval_width': interval_width
    }

    print(
        f"[Regression - {window_type}] Final metrics (horizon={forecast_horizon}): "
        f"MSE={mse_val:.9f}, MAE={mae_val:.9f}, RMSE={rmse_val:.9f}, R2={r2:.9f}, "
        f"Coverage={coverage:.2f}%, IntervalWidth={interval_width:.5f}, "
        f"EpochsRun={epochs_run}"
    )
    return result


# ==============================================================================
# Grid Search for Hyperparameters using the Validation Set
# ==============================================================================
def grid_search_hyperparameters(train_scaled, validate_scaled, scaler, forecast_horizon, search_space, time_limit=600):
    keys = list(search_space.keys())
    all_combinations = list(itertools.product(*(search_space[k] for k in keys)))
    random.shuffle(all_combinations)

    best_mse = float('inf')
    best_hp = None
    start_time = time.time()

    for combo in all_combinations:
        if time.time() - start_time > time_limit:
            print("Time limit reached during grid search.")
            break

        candidate = dict(zip(keys, combo))
        print(f"Evaluating candidate: {candidate}")

        # Create training and validation datasets using candidate's look_back
        X_train, y_train = create_dataset_regression(train_scaled, candidate['look_back'], forecast_horizon)
        X_val, y_val = create_dataset_regression(validate_scaled, candidate['look_back'], forecast_horizon)
        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Skipping candidate {candidate} due to insufficient data.")
            continue

        X_train = X_train.reshape((X_train.shape[0], candidate['look_back'], 1))
        X_val = X_val.reshape((X_val.shape[0], candidate['look_back'], 1))

        model = build_regression_model(candidate['look_back'], candidate['units'], forecast_horizon,
                                       candidate['learning_rate'])
        model.fit(X_train, y_train, epochs=candidate['epochs'], batch_size=candidate['batch_size'], verbose=0)

        y_val_pred = model.predict(X_val, verbose=0)
        # Inverse transform using the scaler fitted on the training data
        y_val_pred_orig = inverse_standard_scaling(y_val_pred, scaler)
        y_val_orig = inverse_standard_scaling(y_val, scaler)
        mse_val = mean_squared_error(y_val_orig.flatten(), y_val_pred_orig.flatten())
        print(f"Candidate {candidate} achieved validation MSE: {mse_val:.9f}")

        if mse_val < best_mse:
            best_mse = mse_val
            best_hp = candidate

    if best_hp is None:
        print("No valid hyperparameter combination found, using default hyperparameters.")
        best_hp = {
            'epochs': 10,
            'look_back': 60,
            'units': 128,
            'batch_size': 64,
            'learning_rate': 1e-3
        }
    print(f"Best hyperparameters: {best_hp} with validation MSE: {best_mse:.9f}")
    return best_hp


# ==============================================================================
# Processing Each Sliding Window (Regression)
# ==============================================================================
def process_window_regression(window_config, data, forecast_horizon, search_space):
    n_samples = len(data)
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    validate_start = int(window_config['validate'][0] * n_samples)
    validate_end = int(window_config['validate'][1] * n_samples)
    test_start = int(window_config['test'][0] * n_samples)
    test_end = int(window_config['test'][1] * n_samples)

    train_raw = data[train_start:train_end]
    validate_raw = data[validate_start:validate_end]
    test_raw = data[test_start:test_end]

    if len(train_raw) < 10 or len(validate_raw) < 10 or len(test_raw) < 10:
        print(f"Skipping {window_config['type']} - insufficient data")
        return None

    # Reshape data for LSTM input: (samples, 1)
    train_raw = train_raw.reshape(-1, 1)
    validate_raw = validate_raw.reshape(-1, 1)
    test_raw = test_raw.reshape(-1, 1)

    # Scale using training data only (for grid search)
    scaler_train = StandardScaler()
    train_scaled = scaler_train.fit_transform(train_raw)
    validate_scaled = scaler_train.transform(validate_raw)

    # Perform grid search on the validation set
    best_hp = grid_search_hyperparameters(train_scaled, validate_scaled, scaler_train, forecast_horizon, search_space,
                                          time_limit=600)

    # Combine training and validation for final model training
    train_val_raw = np.concatenate([train_raw, validate_raw], axis=0)
    scaler_final = StandardScaler()
    train_val_scaled = scaler_final.fit_transform(train_val_raw)
    test_scaled = scaler_final.transform(test_raw)

    # Train and evaluate the final model using the best hyperparameters on train+validate and test on the test set.
    result = train_and_evaluate_regression(train_val_scaled, test_scaled, forecast_horizon, window_config['type'],
                                           best_hp, scaler_final)
    if result is not None:
        result['best_hyperparameters'] = best_hp
    return result


# ==============================================================================
# Save Results (Regression)
# ==============================================================================
def save_results_regression(results, csv_path):
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


if __name__ == "__main__":
    main()
