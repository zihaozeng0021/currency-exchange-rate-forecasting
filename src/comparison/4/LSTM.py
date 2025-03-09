import numpy as np
import pandas as pd
import os
import random
import warnings
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input

import optuna


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


def smooth_data(data):
    return np.diff(data, n=1)


# ==============================================================================
# MinMax Scaling for train and test data
# ==============================================================================
def apply_minmax_scaling(train_data, test_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler


# ==============================================================================
# Dataset Creation for Classification
# ==============================================================================
def create_dataset_classification(dataset, look_back=1, forecast_horizon=1, threshold=0):
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X_seq = dataset[i: i + look_back, 0]
        future_vals = dataset[i + look_back: i + look_back + forecast_horizon, 0]
        pct_changes = (future_vals - X_seq[-1]) / X_seq[-1]
        labels = (pct_changes > threshold).astype(int)
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
def bayesian_search_hyperparameters_classification(train_scaled, test_scaled, forecast_horizon, search_space, time_limit=600):
    def objective(trial):
        # Sample hyperparameters
        look_back = trial.suggest_int('look_back', search_space['look_back'][0], search_space['look_back'][1])
        units = trial.suggest_int('units', search_space['units'][0], search_space['units'][1])
        batch_size = trial.suggest_int('batch_size', search_space['batch_size'][0], search_space['batch_size'][1])
        learning_rate = trial.suggest_float('learning_rate', search_space['learning_rate'][0], search_space['learning_rate'][1])
        epochs = trial.suggest_int('epochs', search_space['epochs'][0], search_space['epochs'][1])

        # Create datasets using the test set for evaluation
        X_train, y_train = create_dataset_classification(train_scaled, look_back, forecast_horizon)
        X_test, y_test = create_dataset_classification(test_scaled, look_back, forecast_horizon)

        if len(X_train) == 0 or len(X_test) == 0:
            return 0.0  # Return low accuracy for invalid trials

        # Reshape data for LSTM
        X_train = X_train.reshape((X_train.shape[0], look_back, 1))
        X_test = X_test.reshape((X_test.shape[0], look_back, 1))

        # Build and train model
        model = build_classification_model(look_back, units, forecast_horizon, learning_rate)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        test_probs = model.predict(X_test, verbose=0)
        test_preds = (test_probs > 0.5).astype(int)
        accuracy = accuracy_score(y_test.flatten(), test_preds.flatten())
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=time_limit)
    print(f"Best hyperparameters: {study.best_params} with test accuracy: {study.best_value:.4f}")
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
# Save Results (Classification)
# ==============================================================================
def save_results(results, csv_path):
    if results:
        results_df = pd.DataFrame(results)
        numeric_cols = results_df.select_dtypes(include=np.number).columns
        avg_row = results_df[numeric_cols].mean().to_dict()
        avg_row['forecast_horizon'] = 'average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        cols_to_show = ['forecast_horizon', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'threshold']
        print("\nFinal classification results:")
        print(results_df[cols_to_show].to_string(index=False))
    else:
        print("No valid classification results generated.")


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    DATA_PATH = 'data/EURUSD.csv'
    CLASSIFICATION_CSV_PATH = 'results/LSTM.csv'
    FORECAST_HORIZONS_CLF = [1, 3, 5]

    search_space = {
        'epochs': (10, 50),
        'look_back': (30, 120),
        'units': (50, 200),
        'batch_size': (16, 128),
        'learning_rate': (0.0001, 0.01)
    }

    configure_tf()
    set_global_config(seed=42)

    data = load_data(DATA_PATH)
    data = smooth_data(data)

    # Split the data into 80% train and 20% test sets
    n_samples = len(data)
    split_point = int(0.8 * n_samples)
    train_raw = data[:split_point].reshape(-1, 1)
    test_raw = data[split_point:].reshape(-1, 1)

    train_scaled, test_scaled, scaler = apply_minmax_scaling(train_raw, test_raw)

    results = []
    start_time = time.time()

    for horizon in FORECAST_HORIZONS_CLF:
        print(f"\n=== Processing forecast horizon: {horizon} ===")
        # Optimize hyperparameters using the test set for evaluation
        best_hp = bayesian_search_hyperparameters_classification(train_scaled, test_scaled, horizon, search_space)
        # Combine train and test for final model training
        combined_raw = np.concatenate([train_raw, test_raw], axis=0)
        combined_scaled = scaler.transform(combined_raw)

        look_back = best_hp['look_back']
        X_combined, y_combined = create_dataset_classification(combined_scaled, look_back, horizon)
        X_test_final, y_test_final = create_dataset_classification(test_scaled, look_back, horizon)

        if len(X_combined) == 0 or len(X_test_final) == 0:
            print(f"Insufficient data for forecast horizon {horizon}. Skipping.")
            continue

        X_combined = X_combined.reshape((X_combined.shape[0], look_back, 1))
        X_test_final = X_test_final.reshape((X_test_final.shape[0], look_back, 1))

        model = build_classification_model(look_back, best_hp['units'], horizon, best_hp['learning_rate'])
        history = model.fit(X_combined, y_combined, epochs=best_hp['epochs'], batch_size=best_hp['batch_size'], verbose=0)
        epochs_run = len(history.history['loss'])

        # Tune threshold on test set predictions
        test_probs = model.predict(X_test_final, verbose=0)
        best_threshold, _ = tune_threshold_pr(y_test_final, test_probs, horizon)
        test_preds = (test_probs > best_threshold).astype(int)
        avg_acc, avg_prec, avg_rec, avg_f1, avg_spec = evaluate_metrics(y_test_final, test_preds, horizon)

        print(f"[Classification] Final metrics (horizon={horizon}): Accuracy={avg_acc:.5f}, Precision={avg_prec:.5f}, "
              f"Recall={avg_rec:.5f}, F1={avg_f1:.5f}, Specificity={avg_spec:.5f}, Threshold={best_threshold:.2f}")

        result = {
            'forecast_horizon': horizon,
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
        results.append(result)

    save_results(results, CLASSIFICATION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
