import numpy as np
import pandas as pd
import os
import random
import warnings
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define file paths and forecast horizon(s)
    DATA_PATH = './../../../../data/raw/USDEUR=X_max_1d.csv'
    CLASSIFICATION_CSV_PATH = './results/classification_results33.csv'
    FORECAST_HORIZONS_CLF = [1]

    # Define hyperparameters for each window with new values.
    # Note: The 'threshold' parameter is no longer used for label creation in this version.
    hyperparams_list = [
        {'look_back': 30, 'units': 112, 'batch_size': 43, 'learning_rate': 0.006334967708569153, 'epochs': 38},
        {'look_back': 107, 'units': 144, 'batch_size': 71, 'learning_rate': 0.005441528224151469, 'epochs': 33},
        {'look_back': 59, 'units': 121, 'batch_size': 43, 'learning_rate': 0.004928085279110264, 'epochs': 31},
        {'look_back': 90, 'units': 175, 'batch_size': 38, 'learning_rate': 0.00519942556265058, 'epochs': 31},
        {'look_back': 117, 'units': 68, 'batch_size': 86, 'learning_rate': 0.005735583493692174, 'epochs': 48},
        {'look_back': 81, 'units': 128, 'batch_size': 88, 'learning_rate': 0.006631609216704043, 'epochs': 35}
    ]

    # Configure TensorFlow and set seeds
    configure_tf()
    set_global_config(seed=42)

    # Load data and generate sliding windows
    data = load_data(DATA_PATH)
    data = smooth_data(data)
    windows = generate_sliding_windows()

    results = []
    start_time = time.time()

    # Loop through each window and forecast horizon
    for i, window in enumerate(windows):
        print(f"\n=== Processing {window['type']} ===")
        print(f"Training range: {window['train'][0] * 100:.1f}% - {window['train'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        current_hyperparams = hyperparams_list[i]
        for horizon in FORECAST_HORIZONS_CLF:
            result = process_window_classification(window, data, horizon, current_hyperparams)
            if result is not None:
                results.append(result)

    # Save the results to CSV
    save_results(results, CLASSIFICATION_CSV_PATH)
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


def smooth_data(data):
    # Apply a simple first-order difference to smooth the data
    return np.diff(data, n=1)


# ==============================================================================
# MinMax Scaling for Train and Test Data
# ==============================================================================
def apply_minmax_scaling(train_data, test_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler


# ==============================================================================
# Sliding Windows Configuration
# ==============================================================================
def generate_sliding_windows():
    # Each window defines a different train-test split (expressed as fractions of the data)
    return [
        {'type': 'window_1', 'train': (0.0, 0.12), 'test': (0.16, 0.2)},
        {'type': 'window_2', 'train': (0.16, 0.28), 'test': (0.32, 0.36)},
        {'type': 'window_3', 'train': (0.32, 0.44), 'test': (0.48, 0.52)},
        {'type': 'window_4', 'train': (0.48, 0.60), 'test': (0.64, 0.68)},
        {'type': 'window_5', 'train': (0.64, 0.76), 'test': (0.8, 0.84)},
        {'type': 'window_6', 'train': (0.8, 0.92), 'test': (0.96, 1.0)}
    ]


# ==============================================================================
# Dataset Creation for Classification (Directional Labels)
# ==============================================================================
def create_dataset_classification_direction(dataset, look_back=1, forecast_horizon=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X_seq = dataset[i: i + look_back, 0]
        future_vals = dataset[i + look_back: i + look_back + forecast_horizon, 0]
        # Directional label: 1 if future value > last value of X_seq, else 0.
        labels = (future_vals > X_seq[-1]).astype(int)
        X.append(X_seq)
        y.append(labels)
    return np.array(X), np.array(y)


# ==============================================================================
# Model Building for Classification
# ==============================================================================
def build_classification_model(look_back, units, forecast_horizon, learning_rate):
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(units, return_sequences=False),
        Dense(forecast_horizon, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    # Using binary cross-entropy loss for direction prediction
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# ==============================================================================
# Evaluation Metrics Calculation (Classification)
# ==============================================================================
def evaluate_metrics(y_true, y_pred, forecast_horizon):
    accuracies, precisions, recalls, f1s, specifics = [], [], [], [], []
    for i in range(forecast_horizon):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]
        acc_i = accuracy_score(y_true_i, y_pred_i)
        prec_i = precision_score(y_true_i, y_pred_i, zero_division=0)
        rec_i = recall_score(y_true_i, y_pred_i, zero_division=0)
        f1_i = f1_score(y_true_i, y_pred_i, zero_division=0)
        cm = confusion_matrix(y_true_i, y_pred_i)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec_i = tn / (tn + fp) if (tn + fp) else 0.0
        else:
            spec_i = 1.0 if np.all(y_true_i == y_pred_i) else 0.0
        accuracies.append(acc_i)
        precisions.append(prec_i)
        recalls.append(rec_i)
        f1s.append(f1_i)
        specifics.append(spec_i)
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(specifics)


# ==============================================================================
# Moving Window Evaluation on the Test Set for Classification
# ==============================================================================
def moving_window_evaluation_classification(train_data, test_data, forecast_horizon, window_type, hyperparams, max_train_size=520):
    combined_data = train_data.copy()
    predictions = []
    ground_truth = []
    look_back = hyperparams['look_back']

    start_idx = 0
    while combined_data.shape[0] < max_train_size and start_idx < len(test_data):
        combined_data = np.vstack([combined_data, test_data[start_idx].reshape(1, 1)])
        start_idx += 1

    for i in range(start_idx, len(test_data) - forecast_horizon + 1):
        current_train_data = combined_data[-max_train_size:]

        # Skip if insufficient data for forming an input sequence
        if current_train_data.shape[0] < look_back:
            combined_data = np.vstack([combined_data, test_data[i].reshape(1, 1)])
            continue

        # Create training samples using the directional labeling
        X_train, y_train = create_dataset_classification_direction(current_train_data, look_back, forecast_horizon)
        if X_train.shape[0] == 0:
            combined_data = np.vstack([combined_data, test_data[i].reshape(1, 1)])
            if combined_data.shape[0] > max_train_size:
                combined_data = combined_data[1:]
            continue

        # Reshape input for LSTM: (samples, look_back, features)
        X_train = X_train.reshape((X_train.shape[0], look_back, 1))

        model = build_classification_model(look_back, hyperparams['units'], forecast_horizon, hyperparams['learning_rate'])
        model.fit(X_train, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], verbose=0)

        # Prepare the prediction input (last 'look_back' values from current training data)
        X_pred = current_train_data[-look_back:].reshape((1, look_back, 1))
        prob = model.predict(X_pred, verbose=0)
        pred = (prob > 0.5).astype(int)[0]
        predictions.append(pred)

        # Compute ground truth label using the directional rule:
        # Compare the last value of current training data to the next forecast_horizon value in test_data.
        last_val = current_train_data[-1, 0]
        next_vals = test_data[i: i + forecast_horizon, 0]
        true_label = (next_vals > last_val).astype(int)
        ground_truth.append(true_label)

        combined_data = np.vstack([combined_data, test_data[i].reshape(1, 1)])
        if combined_data.shape[0] > max_train_size:
            combined_data = combined_data[1:]

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    avg_acc, avg_prec, avg_rec, avg_f1, avg_spec = evaluate_metrics(ground_truth, predictions, forecast_horizon)

    result = {
        'type': window_type,
        'forecast_horizon': forecast_horizon,
        'look_back': look_back,
        'units': hyperparams['units'],
        'batch_size': hyperparams['batch_size'],
        'learning_rate': hyperparams['learning_rate'],
        'epochs': hyperparams['epochs'],
        'accuracy': avg_acc,
        'precision': avg_prec,
        'recall': avg_rec,
        'f1_score': avg_f1,
        'specificity': avg_spec,
        'n_predictions': len(predictions)
    }
    print(
        f"[Moving Window Evaluation - {window_type}] Final metrics (horizon={forecast_horizon}): "
        f"Accuracy={avg_acc:.5f}, Precision={avg_prec:.5f}, Recall={avg_rec:.5f}, "
        f"F1 Score={avg_f1:.5f}, Specificity={avg_spec:.5f}, Predictions={len(predictions)}"
    )
    return result



# ==============================================================================
# Processing Each Sliding Window (Classification)
# ==============================================================================
def process_window_classification(window_config, data, forecast_horizon, hyperparams):
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

    # Reshape raw data for LSTM input: (samples, 1)
    train_data = train_raw.reshape(-1, 1)
    test_data = test_raw.reshape(-1, 1)

    # Apply MinMax scaling to both training and testing data
    train_data_scaled, test_data_scaled, _ = apply_minmax_scaling(train_data, test_data)

    # Apply moving window evaluation on the test set with a fixed maximum training size of 520
    return moving_window_evaluation_classification(train_data_scaled, test_data_scaled, forecast_horizon,
                                                   window_config['type'], hyperparams, max_train_size=520)


# ==============================================================================
# Save Results (Classification)
# ==============================================================================
def save_results(results, csv_path):
    if results:
        results_df = pd.DataFrame(results)
        numeric_cols = results_df.select_dtypes(include=np.number).columns
        avg_row = results_df[numeric_cols].mean().to_dict()
        avg_row['type'] = 'average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print("\nFinal classification results:")
        print(results_df[['type', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity']].to_string(index=False))
    else:
        print("No valid classification results generated.")


if __name__ == "__main__":
    main()
