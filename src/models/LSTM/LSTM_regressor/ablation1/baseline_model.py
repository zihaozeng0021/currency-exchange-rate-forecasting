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
N_TRIALS = 32
REGRESSION_CSV_PATH = './ablation1_baseline.csv'

# Global list to store final results
regressor_results = []


def generate_sliding_windows(data_length: int) -> list:
    """Generate sliding window configurations for time series validation"""
    return [
        {'type': 'window_1', 'train': (0.0, 0.3), 'test': (0.3, 0.4)},
        {'type': 'window_2', 'train': (0.3, 0.6), 'test': (0.6, 0.7)},
        {'type': 'window_3', 'train': (0.6, 0.9), 'test': (0.9, 1.0)}
    ]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess data from CSV"""
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def create_dataset_regression(dataset: np.ndarray,
                              look_back: int = 1,
                              horizon: int = 1) -> tuple:
    """Create time-series sequences for LSTM training"""
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
    """Construct LSTM model architecture"""
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
    """Optuna optimization objective function"""
    # Hyperparameter suggestions
    look_back = trial.suggest_int('look_back', 30, 120)
    units = trial.suggest_int('units', 50, 200)
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    epochs = trial.suggest_int('epochs', 5, 50)

    # Create training/validation sets
    X_train, y_train = create_dataset_regression(X_train_full, look_back, horizon)
    X_val_, y_val_ = create_dataset_regression(X_val, look_back, horizon)

    # Handle insufficient data cases
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

    return history.history['val_loss'][-1]


def process_window(window_config: dict,
                   data: np.ndarray,
                   horizon: int) -> None:
    """Process a single sliding window"""
    # Calculate data indices
    n_samples = len(data)
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    test_start = int(window_config['test'][0] * n_samples)
    test_end = int(window_config['test'][1] * n_samples)

    # Extract window data
    train_raw = data[train_start:train_end]
    test_raw = data[test_start:test_end]

    # Data validation check
    if len(train_raw) < 10 or len(test_raw) < 10:
        print(f"Skipping {window_config['type']} - insufficient data")
        return

    # Scale window data independently
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1))
    test_scaled = scaler.transform(test_raw.reshape(-1, 1))

    # Hyperparameter tuning setup
    val_ratio = 0.2
    val_size = max(1, int(len(train_scaled) * val_ratio))

    try:
        train_tune = train_scaled[:-val_size]
        val_tune = train_scaled[-val_size:]
    except Exception as e:
        print(f"Error processing {window_config['type']}: {str(e)}")
        return

    # Optuna optimization
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_tune, horizon, val_tune),
        n_trials=N_TRIALS,
        timeout=600
    )

    # Extract best parameters
    best_params = study.best_params
    print(f"\n[{window_config['type']}] Best params:", best_params)

    # Model training with best parameters
    look_back = best_params['look_back']
    units = best_params['units']
    lr = best_params['learning_rate']
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']

    # Create final datasets
    X_train, y_train = create_dataset_regression(train_scaled, look_back, horizon)
    X_test, y_test = create_dataset_regression(test_scaled, look_back, horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"{window_config['type']} - insufficient data after processing")
        return

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    # Build and train final model
    model = build_regression_model(look_back, units, horizon, lr)
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)
    final_epochs = len(history.history['loss'])

    # Generate predictions
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    # Inverse scaling transformations
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train_pred_inv = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    y_test_pred_inv = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_inv, y_test_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_test_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_test_pred_inv)

    # Calculate prediction intervals
    train_residuals = y_train_inv - y_train_pred_inv
    sigma = np.std(train_residuals)
    lower_bound = y_test_pred_inv - 1.96 * sigma
    upper_bound = y_test_pred_inv + 1.96 * sigma
    coverage = np.mean((y_test_inv >= lower_bound) & (y_test_inv <= upper_bound)) * 100
    interval_width = np.mean(upper_bound - lower_bound)

    # Store results
    regressor_results.append({
        'type': window_config['type'],
        'forecast_horizon': horizon,
        'look_back': look_back,
        'units': units,
        'batch_size': batch_size,
        'learning_rate': lr,
        'epochs_run': final_epochs,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'coverage_probability (%)': coverage,
        'interval_width': interval_width
    })

    # Print window results
    print(f"[{window_config['type']}] Metrics:")
    print(f"MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")
    print(f"R²: {r2:.4f}, Coverage: {coverage:.2f}%, Width: {interval_width:.6f}")


def main() -> None:
    """Main execution function"""
    # Load and prepare data
    df = load_data(DATA_PATH)
    data = df['Close'].values

    # Generate window configurations
    windows = generate_sliding_windows(len(data))

    print("\nStarting sliding window validation...")
    print(f"Results will be saved to: {REGRESSION_CSV_PATH}")
    start_time = time.time()

    # Process each window
    for window in windows:
        print(f"\n=== Processing {window['type']} ===")
        print(f"Training range: {window['train'][0] * 100}%-{window['train'][1] * 100}%")
        print(f"Testing range: {window['test'][0] * 100}%-{window['test'][1] * 100}%")

        for horizon in FORECAST_HORIZONS_REG:
            process_window(window, data, horizon)

    # Calculate average metrics
    if regressor_results:
        df_results = pd.DataFrame(regressor_results)
        numeric_cols = df_results.select_dtypes(include=np.number).columns
        avg_row = df_results[numeric_cols].mean().to_dict()
        avg_row['type'] = 'average'
        regressor_results.append(avg_row)

    # Save final results
    if regressor_results:
        df_final = pd.DataFrame(regressor_results)
        df_final = df_final[['type'] + [col for col in df_final if col != 'type']]
        os.makedirs(os.path.dirname(REGRESSION_CSV_PATH), exist_ok=True)
        df_final.to_csv(REGRESSION_CSV_PATH, index=False)

        print("\nFinal results:")
        print(df_final[['type', 'mse', 'mae', 'rmse', 'r2_score']].to_string(index=False))
    else:
        print("\nNo valid results generated")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()