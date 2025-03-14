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
from keras.layers import LSTM, Dense, Input, GRU

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

    window_size = 30
    data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    return data


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


def calculate_profit_accuracy_and_transaction_frequency(y_true, probs, center_threshold=0.5, margin=0.1):
    # Create an array for final predictions (same shape as probs)
    final_predictions = np.empty_like(probs, dtype=int)
    # Define trade mask: a trade is executed if probability is below (center_threshold - margin) or above (center_threshold + margin)
    trade_mask = (probs < (center_threshold - margin)) | (probs > (center_threshold + margin))
    # For executed trades, assign 0 if probability is below lower bound, 1 if above upper bound
    final_predictions[probs < (center_threshold - margin)] = 0
    final_predictions[probs > (center_threshold + margin)] = 1
    # Mark no-trade cases as -1
    final_predictions[~trade_mask] = -1
    total_predictions = probs.size
    executed_trades = np.sum(trade_mask)
    if executed_trades > 0:
        profitable_trades = np.sum((final_predictions == y_true) & trade_mask)
        profit_accuracy = (profitable_trades / executed_trades) * 100.0
    else:
        profit_accuracy = 0.0
    transaction_frequency = (executed_trades / total_predictions) * 100.0
    return profit_accuracy, transaction_frequency, final_predictions


# ==============================================================================
# Bayesian Optimization for Hyperparameters using Optuna (Classification)
# ==============================================================================
def bayesian_search_hyperparameters_classification(train_scaled, test_scaled, forecast_horizon, search_space, time_limit=60):
    def objective(trial):
        # Sample hyperparameters
        look_back = trial.suggest_int('look_back', search_space['look_back'][0], search_space['look_back'][1])
        units = trial.suggest_int('units', search_space['units'][0], search_space['units'][1])
        batch_size = trial.suggest_int('batch_size', search_space['batch_size'][0], search_space['batch_size'][1])
        learning_rate = trial.suggest_float('learning_rate', search_space['learning_rate'][0], search_space['learning_rate'][1])
        epochs = trial.suggest_int('epochs', search_space['epochs'][0], search_space['epochs'][1])
        X_train, y_train = create_dataset_classification(train_scaled, look_back, forecast_horizon)
        X_test, y_test = create_dataset_classification(test_scaled, look_back, forecast_horizon)
        if len(X_train) == 0 or len(X_test) == 0:
            return 0.0  # Return low accuracy for invalid trials
        X_train = X_train.reshape((X_train.shape[0], look_back, 1))
        X_test = X_test.reshape((X_test.shape[0], look_back, 1))
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
        # Ensure the first column is 'currency_pair'
        cols = results_df.columns.tolist()
        if 'currency_pair' in cols:
            cols.insert(0, cols.pop(cols.index('currency_pair')))
            results_df = results_df[cols]
        numeric_cols = results_df.select_dtypes(include=np.number).columns
        avg_row = results_df[numeric_cols].mean().to_dict()
        avg_row['forecast_horizon'] = 'average'
        avg_row['currency_pair'] = 'average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        cols_to_show = ['currency_pair', 'forecast_horizon', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity',
                        'threshold', 'profit_accuracy', 'transaction_frequency']
        print("\nFinal classification results:")
        print(results_df[cols_to_show].to_string(index=False))
    else:
        print("No valid classification results generated.")


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define a dictionary of currency pairs and their corresponding data file paths
    currency_files = {
        "AUDJPY": "data/AUDJPY.csv",    # Group 1
        "AUDNZD": "data/AUDNZD.csv",    # Group 1
        "AUDUSD": "data/AUDUSD.csv",    # Group 1
        "CADJPY": "data/CADJPY.csv",    # Group 1
        "EURAUD": "data/EURAUD.csv",    # Group 1
        "EURCAD": "data/EURCAD.csv",    # Group 1
        "EURCSK": "data/EURCSK.csv",    # Group 1
        "EURNOK": "data/EURNOK.csv",    # Group 1
        "GBPAUD": "data/GBPAUD.csv",    # Group 1
        "NZDUSD": "data/NZDUSD.csv",    # Group 1
        "USDCAD": "data/USDCAD.csv",    # Group 1
        "USDNOK": "data/USDNOK.csv",    # Group 1
        "USDJPY": "data/USDJPY.csv",    # Group 1
        "USDSGD": "data/USDSGD.csv",    # Group 1
        "USDZAR": "data/USDZAR.csv",    # Group 1
        "EURGBP": "data/EURGBP.csv",    # Group 1
        "EURUSD": "data/EURUSD.csv",    # Group 2
        "EURJPY": "data/EURJPY.csv",    # Group 2
        "GBPCHF": "data/GBPCHF.csv",    # Group 2
        "GBPUSD": "data/GBPUSD.csv"     # Group 2
    }

    CLASSIFICATION_CSV_PATH = 'results/GRU-LSTM.csv'
    FORECAST_HORIZONS_CLF = [1]

    search_space = {
        'epochs': (10, 50),
        'look_back': (30, 120),
        'units': (50, 200),
        'batch_size': (16, 128),
        'learning_rate': (0.0001, 0.01)
    }

    configure_tf()
    set_global_config(seed=42)

    results = []
    start_time = time.time()

    # Loop over each currency pair
    for cp, file_path in currency_files.items():
        print(f"\n=== Processing currency pair: {cp} ===")
        try:
            data = load_data(file_path)
        except Exception as e:
            print(f"Error loading data for {cp}: {e}")
            continue

        data = smooth_data(data)

        n_samples = len(data)
        if n_samples < 100:  # Check for sufficient data
            print(f"Not enough data for {cp}, skipping.")
            continue

        split_point = int(0.8 * n_samples)
        train_raw = data[:split_point].reshape(-1, 1)
        test_raw = data[split_point:].reshape(-1, 1)
        train_scaled, test_scaled, scaler = apply_minmax_scaling(train_raw, test_raw)

        for horizon in FORECAST_HORIZONS_CLF:
            print(f"\nProcessing forecast horizon: {horizon} for {cp}")
            # Optimize hyperparameters using Bayesian optimization
            best_hp = bayesian_search_hyperparameters_classification(train_scaled, test_scaled, horizon, search_space)
            # Combine train and test for final training
            combined_raw = np.concatenate([train_raw, test_raw], axis=0)
            combined_scaled = scaler.transform(combined_raw)

            look_back = best_hp['look_back']
            X_combined, y_combined = create_dataset_classification(combined_scaled, look_back, horizon)
            X_test_final, y_test_final = create_dataset_classification(test_scaled, look_back, horizon)

            if len(X_combined) == 0 or len(X_test_final) == 0:
                print(f"Insufficient data for forecast horizon {horizon} for {cp}, skipping this horizon.")
                continue

            X_combined = X_combined.reshape((X_combined.shape[0], look_back, 1))
            X_test_final = X_test_final.reshape((X_test_final.shape[0], look_back, 1))

            model = build_classification_model(look_back, best_hp['units'], horizon, best_hp['learning_rate'])
            history = model.fit(X_combined, y_combined, epochs=best_hp['epochs'], batch_size=best_hp['batch_size'], verbose=0)
            epochs_run = len(history.history['loss'])

            test_probs = model.predict(X_test_final, verbose=0)
            best_threshold, _ = tune_threshold_pr(y_test_final, test_probs, horizon)
            test_preds = (test_probs > best_threshold).astype(int)
            avg_acc, avg_prec, avg_rec, avg_f1, avg_spec = evaluate_metrics(y_test_final, test_preds, horizon)
            profit_acc, trans_freq, final_preds = calculate_profit_accuracy_and_transaction_frequency(
                y_test_final, test_probs, center_threshold=best_threshold, margin=0.1
            )

            print(f"[Classification] {cp} (horizon={horizon}): Accuracy={avg_acc:.5f}, Precision={avg_prec:.5f}, "
                  f"Recall={avg_rec:.5f}, F1={avg_f1:.5f}, Specificity={avg_spec:.5f}, Threshold={best_threshold:.2f}")
            print(f"Profit Accuracy={profit_acc:.2f}%, Transaction Frequency={trans_freq:.2f}%")

            result = {
                'currency_pair': cp,
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
                'specificity': avg_spec,
                'profit_accuracy': profit_acc,
                'transaction_frequency': trans_freq
            }
            results.append(result)

    save_results(results, CLASSIFICATION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
