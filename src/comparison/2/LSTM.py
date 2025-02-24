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
    # Define file paths and hyperparameters
    DATA_PATH = 'data/EURUSD.csv'
    REGRESSION_CSV_PATH = 'results/LSTM.csv'
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

    data = load_data(DATA_PATH)
    train_raw, test_raw = split_data(data, train_ratio=0.8)
    train_raw = train_raw.reshape(-1, 1)
    test_raw = test_raw.reshape(-1, 1)
    train_scaled, test_scaled, scaler = apply_standard_scaling(train_raw, test_raw)

    results = []
    start_time = time.time()

    # Loop over forecast horizons
    for horizon in FORECAST_HORIZONS_REG:
        print(f"\n=== Processing FORECAST_HORIZON {horizon} ===")
        best_hp = bayesian_search_hyperparameters(train_scaled, test_scaled, scaler, horizon,
                                                  search_space, time_limit=600)
        result = train_and_evaluate_regression(train_scaled, test_scaled, horizon, "80-20 split",
                                               best_hp, scaler)
        if result is not None:
            result['best_hyperparameters'] = best_hp
            results.append(result)

    # Save all regression results (without 'type' or an average row)
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
    optuna.samplers.TPESampler(seed=seed)


# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================
def load_data(data_path):
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df['Close'].values


def split_data(data, train_ratio=0.8):
    n_samples = len(data)
    split_index = int(train_ratio * n_samples)
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


# ==============================================================================
# Z-Score Scaling for train and test data
# ==============================================================================
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
                                  forecast_horizon, split_type, hyperparams, scaler):
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
                                    time_limit=600):
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

        X_train = X_train.reshape((X_train.shape[0], look_back, 1))
        X_test = X_test.reshape((X_test.shape[0], look_back, 1))

        model = build_regression_model(look_back, units, forecast_horizon, learning_rate)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        y_test_pred_orig = inverse_standard_scaling(y_test_pred, scaler)
        y_test_orig = inverse_standard_scaling(y_test, scaler)
        mse_val = mean_squared_error(y_test_orig.flatten(), y_test_pred_orig.flatten())
        return mse_val

    study = optuna.create_study(direction="minimize")
    # Optimize until the time limit (600 seconds) is reached.
    study.optimize(objective, timeout=time_limit)
    best_hp = study.best_trial.params
    return best_hp


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
