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

import optuna

# Set global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
sampler = optuna.samplers.TPESampler(seed=SEED)

warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Check GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU: {gpu_devices}" if gpu_devices else "No GPU found; running on CPU.")

# File paths and constants
DATA_PATH = '../../../../../data/raw/USDEUR=X_max_1d.csv'
FORECAST_HORIZONS_REG = [1]
N_TRIALS = 200
REGRESSION_CSV_PATH = './ablation1_baseline.csv'

# Global list to store final regression results
regressor_results = []


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load data from CSV, handle infinities, and drop missing rows.
    """
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def split_data(data: np.ndarray, train_ratio: float = 0.8) -> tuple:
    """
    Split data into train and test arrays based on a train_ratio.
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data


def scale_data(train_data_raw: np.ndarray,
               test_data_raw: np.ndarray) -> tuple:
    """
    Scale data using MinMaxScaler from 0 to 1.
    """
    scaler = MinMaxScaler((0, 1))
    train_scaled = scaler.fit_transform(train_data_raw.reshape(-1, 1))
    test_scaled = scaler.transform(test_data_raw.reshape(-1, 1))
    return scaler, train_scaled, test_scaled


def create_dataset_regression(dataset: np.ndarray,
                              look_back: int = 1,
                              horizon: int = 1) -> tuple:
    """
    Create X, y sequences for regression based on look_back and horizon.
    """
    X, y = [], []
    for i in range(len(dataset) - look_back - horizon + 1):
        X_seq = dataset[i: i + look_back, 0]
        y_seq = dataset[i + look_back: i + look_back + horizon, 0]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)


def build_regression_model(look_back: int,
                           units: int,
                           horizon: int,
                           lr: float) -> Sequential:
    """
    Build and compile an LSTM-based regression model with RMSprop optimizer.
    """
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(units),
        Dense(horizon, activation='linear')
    ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def objective(trial: optuna.trial.Trial,
              X_train_full: np.ndarray,
              horizon: int,
              X_val: np.ndarray) -> float:
    """
    Optuna objective function to search for the best hyperparameters.
    """
    # Hyperparameter suggestions
    look_back = trial.suggest_int('look_back', 30, 120)
    units = trial.suggest_int('units', 50, 200)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 5, 50)

    # Create training/validation sets
    X_train, y_train = create_dataset_regression(X_train_full, look_back, horizon)
    X_val_, y_val_ = create_dataset_regression(X_val, look_back, horizon)

    # If no data, return infinity (invalid case)
    if len(X_train) == 0 or len(X_val_) == 0:
        return float('inf')

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_val_ = X_val_.reshape((X_val_.shape[0], look_back, 1))

    # Build and train model
    model = build_regression_model(look_back, units, horizon, lr)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_, y_val_),
        verbose=0
    )

    # Return final validation loss
    return history.history['val_loss'][-1]


def optimize_and_train_regression(train_scaled: np.ndarray,
                                  test_scaled: np.ndarray,
                                  scaler: MinMaxScaler,
                                  horizon: int) -> None:
    """
    Optimize hyperparameters for a given horizon using Optuna,
    then train and evaluate the final regression model.
    """
    # Create train/validation split for hyperparameter tuning
    val_ratio = 0.2
    val_size = int(len(train_scaled) * val_ratio)
    train_tune = train_scaled[:-val_size]
    val_tune = train_scaled[-val_size:]

    def optuna_objective(trial):
        return objective(trial, train_tune, horizon, val_tune)

    # Optuna study for hyperparameter optimization
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(optuna_objective, n_trials=N_TRIALS, timeout=600)

    best_params = study.best_params
    print("\n[Optuna] Best hyperparameters:", best_params)

    # Final model training with best params
    look_back = best_params['look_back']
    units = best_params['units']
    lr = best_params['learning_rate']
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']

    # Create datasets for final training/testing
    X_train, y_train = create_dataset_regression(train_scaled, look_back, horizon)
    X_test, y_test = create_dataset_regression(test_scaled, look_back, horizon)

    # If no data available, skip
    if len(X_train) == 0 or len(X_test) == 0:
        print("[Regression] Not enough data for these params.")
        return

    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    # Build and fit model
    model = build_regression_model(look_back, units, horizon, lr)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    final_epochs_run = len(history.history['loss'])

    # Predict on train/test sets
    y_train_pred_s = model.predict(X_train, verbose=0)
    y_test_pred_s = model.predict(X_test, verbose=0)

    # Reshape for inverse scaling
    y_train_s = y_train.flatten().reshape(-1, 1)
    y_test_s = y_test.flatten().reshape(-1, 1)
    y_train_pred_s = y_train_pred_s.flatten().reshape(-1, 1)
    y_test_pred_s = y_test_pred_s.flatten().reshape(-1, 1)

    # Inverse transform
    y_train_inv = scaler.inverse_transform(y_train_s).flatten()
    y_test_inv = scaler.inverse_transform(y_test_s).flatten()
    y_train_pred_inv = scaler.inverse_transform(y_train_pred_s).flatten()
    y_test_pred_inv = scaler.inverse_transform(y_test_pred_s).flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_test_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_test_pred_inv)

    # Prediction intervals (using train residuals standard deviation)
    residuals_train = y_train_inv - y_train_pred_inv
    sigma = np.std(residuals_train)
    z = 1.96  # 95% confidence
    lower_bound = y_test_pred_inv - z * sigma
    upper_bound = y_test_pred_inv + z * sigma
    coverage = np.mean((y_test_inv >= lower_bound) & (y_test_inv <= upper_bound)) * 100
    interval_width = np.mean(upper_bound - lower_bound)

    # Store results globally
    regressor_results.append({
        'forecast_horizon': horizon,
        'look_back': look_back,
        'units': units,
        'batch_size': batch_size,
        'learning_rate': lr,
        'epochs_run': final_epochs_run,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'coverage_probability (%)': coverage,
        'interval_width': interval_width
    })

    # Print final metrics
    print(
        f"[Regression] (h={horizon}) MSE={mse:.6f}, MAE={mae:.6f}, "
        f"RMSE={rmse:.6f}, R2={r2:.6f}, Coverage={coverage:.2f}%, "
        f"IntervalWidth={interval_width:.6f}, Epochs={final_epochs_run}"
    )


def main() -> None:
    """
    Main function to load data, split, scale, run optimization,
    and save final results to CSV.
    """
    # Load dataset
    df = load_data(DATA_PATH)
    data = df['Close'].values

    # Split data
    train_raw, test_raw = split_data(data, 0.8)

    # Scale data
    scaler, train_scaled, test_scaled = scale_data(train_raw, test_raw)

    print("\nCurrent results file path:", REGRESSION_CSV_PATH)
    print("=== Regression Optimization and Training ===")
    start_time = time.time()

    # Iterate over forecast horizons
    for horizon in FORECAST_HORIZONS_REG:
        optimize_and_train_regression(train_scaled, test_scaled, scaler, horizon)

    end_time = time.time()
    print(f"[Results] Done in {end_time - start_time:.2f}s.")

    # Save results if available
    if regressor_results:
        df_results = pd.DataFrame(regressor_results)
        os.makedirs(os.path.dirname(REGRESSION_CSV_PATH), exist_ok=True)
        df_results.to_csv(REGRESSION_CSV_PATH, index=False)
        print(f"[Results] Saved to {REGRESSION_CSV_PATH}")
    else:
        print("[Results] No results to save.")


# Run the main function
if __name__ == "__main__":
    main()
