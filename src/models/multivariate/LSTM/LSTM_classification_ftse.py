import numpy as np
import pandas as pd
import os
import random
import warnings
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define file paths and hyperparameters
    CURRENCY_DATA_PATH = './../../../../data/raw/USDEUR=X_max_1d.csv'
    FTSE_DATA_PATH = './../../../../data/raw/ftse.csv'
    CLASSIFICATION_CSV_PATH = './results/classification_ftse.csv'
    FORECAST_HORIZONS_CLF = [1]

    hyperparams = {
        'look_back': 5,
        'units': 128,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'epochs': 10
    }

    # Configure TensorFlow and set seeds
    configure_tf()
    set_global_config(seed=42)

    # Load and merge currency and FTSE data based on the Date column
    data_df = load_and_merge_data(CURRENCY_DATA_PATH, FTSE_DATA_PATH)
    # Convert DataFrame to numpy array (columns: [currency, ftse])
    data = data_df.values
    data = smooth_data(data)

    windows = generate_sliding_windows()

    results = []
    start_time = time.time()
    for window in windows:
        print(f"\n=== Processing {window['type']} ===")
        print(f"Training range: {window['train'][0] * 100:.1f}% - {window['train'][1] * 100:.1f}%")
        print(f"Validation range: {window['validate'][0] * 100:.1f}% - {window['validate'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        for horizon in FORECAST_HORIZONS_CLF:
            # First, tune the threshold on the validation split.
            result_val = process_window_classification(window, data, horizon, hyperparams,
                                                       global_threshold=None, eval_split='validate')
            if result_val is None:
                print(f"Skipping {window['type']} due to insufficient validation data.")
                continue
            tuned_threshold = result_val['threshold']
            print(f"==> Tuned threshold on validation for {window['type']}: {tuned_threshold:.5f}")

            # Then, evaluate on the test split using the tuned threshold.
            result_test = process_window_classification(window, data, horizon, hyperparams,
                                                        global_threshold=tuned_threshold, eval_split='test')
            if result_test is not None:
                results.append(result_test)

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
# Data Loading and Preprocessing (Multivariate)
# ==============================================================================
def load_and_merge_data(currency_path, ftse_path):
    # Load currency data
    df_currency = pd.read_csv(currency_path, index_col='Date', parse_dates=True)
    df_currency = df_currency.replace([np.inf, -np.inf], np.nan).dropna()
    df_currency = df_currency.rename(columns={'Close': 'currency'})

    # Load FTSE data
    df_ftse = pd.read_csv(ftse_path, index_col='Date', parse_dates=True)
    df_ftse = df_ftse.replace([np.inf, -np.inf], np.nan).dropna()
    df_ftse = df_ftse.rename(columns={'Close': 'ftse'})

    # Merge datasets on Date (inner join to match currency dates)
    df_merged = df_currency.join(df_ftse, how='inner')
    df_merged = df_merged.sort_index()
    return df_merged


def smooth_data(data):
    # Apply first-order difference to each feature (axis=0)
    return np.diff(data, n=1, axis=0)


# ==============================================================================
# MinMax Scaling for train and test data
# ==============================================================================
def apply_minmax_scaling(train_data, test_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler


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
# Dataset Creation for Classification (Multivariate)
# ==============================================================================
def create_dataset_classification(dataset, look_back=1, forecast_horizon=1, threshold=0):
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        # Use all features (currency and FTSE) for input
        X_seq = dataset[i: i + look_back, :]
        # Compute labels based on currency data (first column)
        last_currency_val = dataset[i + look_back - 1, 0]
        future_vals = dataset[i + look_back: i + look_back + forecast_horizon, 0]
        pct_changes = (future_vals - last_currency_val) / last_currency_val
        labels = (pct_changes > threshold).astype(int)
        X.append(X_seq)
        y.append(labels)
    return np.array(X), np.array(y)


# ==============================================================================
# Model Building for Classification (Multivariate)
# ==============================================================================
def build_classification_model(look_back, num_features, units, forecast_horizon, learning_rate):
    model = Sequential([
        Input(shape=(look_back, num_features)),
        LSTM(units, return_sequences=False),
        Dense(forecast_horizon, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(loss='binary_focal_crossentropy', optimizer=optimizer)
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
# Precision-Recall Curve–Based Threshold Tuning Function
# ==============================================================================
def tune_threshold_pr(eval_probs, y_eval, forecast_horizon, grid_threshold=None):
    if grid_threshold is None:
        # Compute the Precision-Recall curve (flatten arrays, assuming forecast_horizon == 1)
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_eval.flatten(), eval_probs.flatten())
        best_acc = -1
        best_threshold = None
        best_preds = None
        for t in pr_thresholds:
            preds_temp = (eval_probs > t).astype(int)
            acc_temp, _, _, _, _ = evaluate_metrics(y_eval, preds_temp, forecast_horizon)
            if acc_temp > best_acc:
                best_acc = acc_temp
                best_threshold = t
                best_preds = preds_temp
        return best_threshold, best_preds
    else:
        preds = (eval_probs > grid_threshold).astype(int)
        return grid_threshold, preds


# ==============================================================================
# Training and Evaluation (Classification)
# ==============================================================================
def optimize_and_train_classification(train_data, eval_data, forecast_horizon, window_type, hyperparams,
                                      grid_threshold=None):
    look_back = hyperparams['look_back']
    units = hyperparams['units']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epochs = hyperparams['epochs']

    X_train, y_train = create_dataset_classification(train_data, look_back, forecast_horizon)
    X_eval, y_eval = create_dataset_classification(eval_data, look_back, forecast_horizon)

    if len(X_train) == 0 or len(X_eval) == 0:
        print(f"[Classification - {window_type}] Insufficient data for the given parameters.")
        return None

    # X_train and X_eval already have shape (samples, look_back, num_features)
    num_features = X_train.shape[-1]
    model = build_classification_model(look_back, num_features, units, forecast_horizon, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    epochs_run = len(history.history['loss'])

    # Get prediction probabilities
    eval_probs = model.predict(X_eval, verbose=0)

    # Use the Precision-Recall based threshold tuning function
    threshold_used, preds = tune_threshold_pr(eval_probs, y_eval, forecast_horizon, grid_threshold)

    avg_acc, avg_prec, avg_rec, avg_f1, avg_spec = evaluate_metrics(y_eval, preds, forecast_horizon)

    result = {
        'type': window_type,
        'forecast_horizon': forecast_horizon,
        'look_back': look_back,
        'units': units,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs_run': epochs_run,
        'threshold': threshold_used,
        'accuracy': avg_acc,
        'precision': avg_prec,
        'recall': avg_rec,
        'f1_score': avg_f1,
        'specificity': avg_spec
    }
    print(
        f"[Classification - {window_type}] Final metrics (horizon={forecast_horizon}): "
        f"Accuracy={avg_acc:.5f}, Precision={avg_prec:.5f}, Recall={avg_rec:.5f}, "
        f"F1={avg_f1:.5f}, Specificity={avg_spec:.5f}, EpochsRun={epochs_run}, "
        f"Threshold={threshold_used:.2f}"
    )
    return result


# ==============================================================================
# Processing Each Sliding Window (Classification)
# ==============================================================================
def process_window_classification(window_config, data, forecast_horizon, hyperparams, global_threshold=None,
                                  eval_split='test'):
    n_samples = len(data)
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    eval_start = int(window_config[eval_split][0] * n_samples)
    eval_end = int(window_config[eval_split][1] * n_samples)

    train_data = data[train_start:train_end]
    eval_data = data[eval_start:eval_end]

    if len(train_data) < hyperparams['look_back'] or len(eval_data) < hyperparams['look_back']:
        print(f"Skipping {window_config['type']} - insufficient data in {eval_split} split")
        return None

    # Apply MinMax scaling to both training and evaluation data
    train_data_scaled, eval_data_scaled, _ = apply_minmax_scaling(train_data, eval_data)

    return optimize_and_train_classification(
        train_data_scaled, eval_data_scaled, forecast_horizon,
        window_config['type'], hyperparams, grid_threshold=global_threshold
    )


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
