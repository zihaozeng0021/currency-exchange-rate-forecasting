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
from keras.layers import LSTM, Dense, Input, GRU

import optuna


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define file paths and hyperparameters
    CURRENCY_PATH = '../../../data/raw/USDEUR=X_max_1d.csv'
    FTSE_PATH = '../../../data/raw/ftse.csv'
    CLASSIFICATION_CSV_PATH = 'results/multivariate_GRU-LSTM_classification_bo.csv'
    FORECAST_HORIZONS_CLF = [1]

    search_space = {
        'epochs': (10, 50),
        'look_back': (1, 90),
        'units': (50, 200),
        'batch_size': (16, 128),
        'learning_rate': (0.0001, 0.01)
    }

    # Configure TensorFlow and set seeds
    configure_tf()
    set_global_config(seed=42)

    # Load the data and merge currency and FTSE data based on Date
    df = load_data(CURRENCY_PATH, FTSE_PATH)
    # Smooth the data using first-order differencing (applied on both features)
    data = smooth_data(df)
    windows = generate_sliding_windows()

    results = []
    start_time = time.time()
    for window in windows:
        print(f"\n=== Processing {window['type']} ===")
        print(f"Training range: {window['train'][0] * 100:.1f}% - {window['train'][1] * 100:.1f}%")
        print(f"Validation range: {window['validate'][0] * 100:.1f}% - {window['validate'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        for horizon in FORECAST_HORIZONS_CLF:
            result = process_window_classification(window, data, horizon, search_space, eval_split='test')
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
    optuna.samplers.TPESampler(seed=seed)


# ==============================================================================
# Data Loading and Preprocessing (Multivariate)
# ==============================================================================
def load_data(currency_path, ftse_path):
    # Load currency data and rename the 'Close' column to 'currency'
    df_currency = pd.read_csv(currency_path, index_col='Date', parse_dates=True)
    df_currency = df_currency[['Close']].rename(columns={'Close': 'currency'})
    # Load FTSE data and rename the 'Close' column to 'ftse'
    df_ftse = pd.read_csv(ftse_path, index_col='Date', parse_dates=True)
    df_ftse = df_ftse[['Close']].rename(columns={'Close': 'ftse'})
    # Merge the data based on Date (using an inner join on the currency data's dates)
    df_merged = df_currency.merge(df_ftse, left_index=True, right_index=True, how='inner')
    df_merged.dropna(inplace=True)
    return df_merged


def smooth_data(df):
    # Apply first-order differencing to each column to smooth the data
    df_diff = df.diff().dropna()
    return df_diff.values


# ==============================================================================
# MinMax Scaling for Train and Test Data (Multivariate)
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
        # Use all features for the input sequence
        X_seq = dataset[i: i + look_back, :]
        # For the target, use the currency column (assumed to be at index 0)
        last_val = X_seq[-1, 0]
        future_vals = dataset[i + look_back: i + look_back + forecast_horizon, 0]
        pct_changes = (future_vals - last_val) / last_val
        labels = (pct_changes > threshold).astype(int)
        X.append(X_seq)
        y.append(labels)
    return np.array(X), np.array(y)


# ==============================================================================
# Model Building for Classification (Multivariate)
# ==============================================================================
def build_classification_model(look_back, units, forecast_horizon, learning_rate, num_features):
    model = Sequential([
        Input(shape=(look_back, num_features)),
        GRU(units, return_sequences=True),
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
# Bayesian Optimization for Hyperparameters using Optuna (Classification)
# ==============================================================================
def bayesian_search_hyperparameters_classification(train_scaled, validate_scaled, forecast_horizon, search_space, time_limit=600):
    num_features = train_scaled.shape[1]

    def objective(trial):
        # Sample hyperparameters
        look_back = trial.suggest_int('look_back', search_space['look_back'][0], search_space['look_back'][1])
        units = trial.suggest_int('units', search_space['units'][0], search_space['units'][1])
        batch_size = trial.suggest_int('batch_size', search_space['batch_size'][0], search_space['batch_size'][1])
        learning_rate = trial.suggest_float('learning_rate', search_space['learning_rate'][0],
                                            search_space['learning_rate'][1])
        epochs = trial.suggest_int('epochs', search_space['epochs'][0], search_space['epochs'][1])

        # Create datasets
        X_train, y_train = create_dataset_classification(train_scaled, look_back, forecast_horizon)
        X_val, y_val = create_dataset_classification(validate_scaled, look_back, forecast_horizon)

        if len(X_train) == 0 or len(X_val) == 0:
            return 0.0  # Return low accuracy for invalid trials

        # Reshape data for LSTM
        X_train = X_train.reshape((X_train.shape[0], look_back, num_features))
        X_val = X_val.reshape((X_val.shape[0], look_back, num_features))

        # Build and train model
        model = build_classification_model(look_back, units, forecast_horizon, learning_rate, num_features)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predict on validation set
        val_probs = model.predict(X_val, verbose=0)
        val_preds = (val_probs > 0.5).astype(int)
        accuracy = accuracy_score(y_val.flatten(), val_preds.flatten())
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=time_limit)
    print(f"Best hyperparameters: {study.best_params} with validation accuracy: {study.best_value:.4f}")
    return study.best_params


def tune_threshold_pr(y_true, probs, forecast_horizon):
    precision, recall, thresholds = precision_recall_curve(y_true.flatten(), probs.flatten())
    best_f1 = -1
    best_threshold = 0.5  # default value
    for i, t in enumerate(thresholds):
        if precision[i] + recall[i] > 0:
            f1 = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1 = 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    preds = (probs > best_threshold).astype(int)
    return best_threshold, preds


# ==============================================================================
# Processing Each Sliding Window (Classification) (Multivariate)
# ==============================================================================
def process_window_classification(window_config, data, forecast_horizon, search_space, eval_split='test'):
    n_samples = len(data)
    # Extract indices for train, validate, and test splits
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    validate_start = int(window_config['validate'][0] * n_samples)
    validate_end = int(window_config['validate'][1] * n_samples)
    test_start = int(window_config['test'][0] * n_samples)
    test_end = int(window_config['test'][1] * n_samples)

    # Extract data splits
    train_raw = data[train_start:train_end]
    validate_raw = data[validate_start:validate_end]
    test_raw = data[test_start:test_end]

    # Choose evaluation data based on eval_split parameter
    if eval_split == 'validate':
        eval_raw = validate_raw
    elif eval_split == 'test':
        eval_raw = test_raw
    else:
        raise ValueError("eval_split must be either 'validate' or 'test'")

    if len(train_raw) < 10 or len(validate_raw) < 10 or len(eval_raw) < 10:
        print(f"Skipping {window_config['type']} - insufficient data")
        return None

    # No need to reshape as data is already multivariate
    # Scale training and validation data using the helper function
    train_scaled, validate_scaled, scaler = apply_minmax_scaling(train_raw, validate_raw)
    # Scale evaluation data using the same scaler
    eval_scaled = scaler.transform(eval_raw)

    # Perform Bayesian optimization for hyperparameters using train and validation sets
    best_hp = bayesian_search_hyperparameters_classification(
        train_scaled, validate_scaled, forecast_horizon, search_space
    )

    # Combine training and validation data for final model training
    train_val_raw = np.concatenate([train_raw, validate_raw], axis=0)
    train_val_scaled = scaler.transform(train_val_raw)

    # Create datasets using the best look_back parameter
    look_back = best_hp['look_back']
    num_features = train_scaled.shape[1]
    X_train_val, y_train_val = create_dataset_classification(train_val_scaled, look_back, forecast_horizon)
    X_eval, y_eval = create_dataset_classification(eval_scaled, look_back, forecast_horizon)

    if len(X_train_val) == 0 or len(X_eval) == 0:
        print(f"[Classification - {window_config['type']}] Insufficient data after combining.")
        return None

    # Reshape data for LSTM input
    X_train_val = X_train_val.reshape((X_train_val.shape[0], look_back, num_features))
    X_eval = X_eval.reshape((X_eval.shape[0], look_back, num_features))

    # Build and train final model on combined data
    model = build_classification_model(
        look_back=look_back,
        units=best_hp['units'],
        forecast_horizon=forecast_horizon,
        learning_rate=best_hp['learning_rate'],
        num_features=num_features
    )
    history = model.fit(
        X_train_val, y_train_val,
        epochs=best_hp['epochs'],
        batch_size=best_hp['batch_size'],
        verbose=0
    )
    epochs_run = len(history.history['loss'])

    # Use the validation set for threshold tuning using the PR curve
    X_val, y_val = create_dataset_classification(validate_scaled, look_back, forecast_horizon)
    if len(X_val) == 0:
        print(f"[Classification - {window_config['type']}] Insufficient validation data for threshold tuning.")
        return None
    X_val = X_val.reshape((X_val.shape[0], look_back, num_features))
    val_probs = model.predict(X_val, verbose=0)

    best_threshold, _ = tune_threshold_pr(y_val, val_probs, forecast_horizon)

    # Predict on evaluation data using the tuned threshold
    eval_probs = model.predict(X_eval, verbose=0)
    eval_preds = (eval_probs > best_threshold).astype(int)

    # Evaluate metrics on the chosen evaluation split
    avg_acc, avg_prec, avg_rec, avg_f1, avg_spec = evaluate_metrics(y_eval, eval_preds, forecast_horizon)

    result = {
        'type': window_config['type'],
        'forecast_horizon': forecast_horizon,
        'look_back': best_hp['look_back'],
        'units': best_hp['units'],
        'batch_size': best_hp['batch_size'],
        'learning_rate': best_hp['learning_rate'],
        'epochs_run': epochs_run,
        'threshold': best_threshold,
        'accuracy': avg_acc,
        'precision': avg_prec,
        'recall': avg_rec,
        'f1_score': avg_f1,
        'specificity': avg_spec
    }

    print(
        f"[Classification - {window_config['type']}] Final metrics (horizon={forecast_horizon}): "
        f"Accuracy={avg_acc:.5f}, Precision={avg_prec:.5f}, Recall={avg_rec:.5f}, "
        f"F1={avg_f1:.5f}, Specificity={avg_spec:.5f}, Threshold={best_threshold:.2f}"
    )
    return result


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
        cols_to_show = ['type', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'threshold']
        print("\nFinal classification results:")
        print(results_df[cols_to_show].to_string(index=False))
    else:
        print("No valid classification results generated.")


if __name__ == "__main__":
    main()
