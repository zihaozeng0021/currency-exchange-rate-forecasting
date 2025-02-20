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

from deap import base, creator, tools


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    # Define file paths and hyperparameters
    DATA_PATH = '../../../data/raw/USDEUR=X_max_1d.csv'
    CLASSIFICATION_CSV_PATH = 'results/LSTM_classification_ga.csv'
    FORECAST_HORIZONS_CLF = [1]

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

    # Load and preprocess data
    data = load_data(DATA_PATH)
    data = smooth_data(data)
    windows = generate_sliding_windows()

    results = []
    total_start_time = time.time()

    # Loop over each sliding window and forecast horizon
    for window in windows:
        print(f"\n=== Processing {window['type']} ===")
        print(f"Training range: {window['train'][0] * 100:.1f}% - {window['train'][1] * 100:.1f}%")
        print(f"Validation range: {window['validate'][0] * 100:.1f}% - {window['validate'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        for horizon in FORECAST_HORIZONS_CLF:
            print(f"\n--- Genetic Algorithm Search for forecast horizon {horizon} on validation split ---")
            best_hyperparams, best_threshold, best_val_result = genetic_search_classification(
                window, data, horizon, search_space, time_limit=600
            )
            if best_hyperparams is None:
                print(f"No valid configuration found for window {window['type']}. Skipping test evaluation.")
                continue

            print(
                f"==> Best hyperparameters for {window['type']}: {best_hyperparams} with threshold {best_threshold:.2f}")
            print("Evaluating on test split with best hyperparameters...")
            test_result = process_window_classification(
                window, data, horizon, best_hyperparams, global_threshold=best_threshold, eval_split='test'
            )
            if test_result is not None:
                results.append(test_result)

    # Save the results to CSV
    save_results(results, CLASSIFICATION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - total_start_time:.2f} seconds")


# ==============================================================================
# Genetic Algorithm for Hyperparameter Tuning (Classification)
# ==============================================================================
def genetic_search_classification(window_config, data, forecast_horizon, search_space, time_limit=600):
    # --- DEAP setup ---
    # Create fitness and individual classes (if not already created)
    try:
        creator.FitnessMax
    except AttributeError:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generators for each hyperparameter
    toolbox.register("epochs_attr", random.randint, search_space['epochs'][0], search_space['epochs'][1])
    toolbox.register("look_back_attr", random.randint, search_space['look_back'][0], search_space['look_back'][1])
    toolbox.register("units_attr", random.randint, search_space['units'][0], search_space['units'][1])
    toolbox.register("batch_size_attr", random.randint, search_space['batch_size'][0], search_space['batch_size'][1])
    toolbox.register("learning_rate_attr", random.uniform, search_space['learning_rate'][0],
                     search_space['learning_rate'][1])

    # Structure initializers: an individual is a list of the five hyperparameters.
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.epochs_attr, toolbox.look_back_attr, toolbox.units_attr,
                      toolbox.batch_size_attr, toolbox.learning_rate_attr), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Evaluation function: converts the individual into a hyperparameter dict and evaluates
    def eval_individual(individual):
        hyperparams = {
            'epochs': int(individual[0]),
            'look_back': int(individual[1]),
            'units': int(individual[2]),
            'batch_size': int(individual[3]),
            'learning_rate': float(individual[4])
        }
        result = process_window_classification(window_config, data, forecast_horizon, hyperparams,
                                               global_threshold=None, eval_split='validate')
        if result is None:
            return (0.0,)
        return (result['accuracy'],)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)

    # Custom mutation: with probability indpb mutate each gene by re-sampling
    def mutate_individual(individual, indpb):
        if random.random() < indpb:
            individual[0] = random.randint(search_space['epochs'][0], search_space['epochs'][1])
        if random.random() < indpb:
            individual[1] = random.randint(search_space['look_back'][0], search_space['look_back'][1])
        if random.random() < indpb:
            individual[2] = random.randint(search_space['units'][0], search_space['units'][1])
        if random.random() < indpb:
            individual[3] = random.randint(search_space['batch_size'][0], search_space['batch_size'][1])
        if random.random() < indpb:
            individual[4] = random.uniform(search_space['learning_rate'][0], search_space['learning_rate'][1])
        return (individual,)

    toolbox.register("mutate", mutate_individual, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- Genetic Algorithm Loop ---
    pop_size = 20
    pop = toolbox.population(n=pop_size)
    start_time = time.time()
    best_individual = None
    best_fitness = 0.0
    gen = 0

    while time.time() - start_time < time_limit:
        # Evaluate individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
            if ind.fitness.values[0] > best_fitness:
                best_fitness = ind.fitness.values[0]
                best_individual = ind
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness after crossover and mutation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)
            if ind.fitness.values[0] > best_fitness:
                best_fitness = ind.fitness.values[0]
                best_individual = ind
        pop[:] = offspring
        gen += 1
        print(f"Generation {gen} - best fitness so far: {best_fitness:.5f}")

    if best_individual is None:
        return None, None, None

    # Convert best individual to hyperparameters
    best_hyperparams = {
        'epochs': int(best_individual[0]),
        'look_back': int(best_individual[1]),
        'units': int(best_individual[2]),
        'batch_size': int(best_individual[3]),
        'learning_rate': float(best_individual[4])
    }
    # Re-evaluate on the validation split to get the tuned threshold
    best_result = process_window_classification(window_config, data, forecast_horizon,
                                                best_hyperparams, global_threshold=None, eval_split='validate')
    best_threshold = best_result['threshold'] if best_result is not None else None
    return best_hyperparams, best_threshold, best_result


def tune_threshold_pr(y_true, probs, forecast_horizon):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true.flatten(), probs.flatten())
    best_f1 = -1
    best_threshold = 0.5  # default value
    for i, t in enumerate(thresholds):
        if precision_vals[i] + recall_vals[i] > 0:
            f1 = 2 * precision_vals[i] * recall_vals[i] / (precision_vals[i] + recall_vals[i])
        else:
            f1 = 0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    preds = (probs > best_threshold).astype(int)
    return best_threshold, preds


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

    # Reshape data for LSTM input: (samples, look_back, features)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_eval = X_eval.reshape((X_eval.shape[0], look_back, 1))

    model = build_classification_model(look_back, units, forecast_horizon, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    epochs_run = len(history.history['loss'])

    # Get prediction probabilities
    eval_probs = model.predict(X_eval, verbose=0)

    # Precision-Recall Curve–Based Threshold Tuning on Evaluation Data
    if grid_threshold is None:
        best_threshold, preds = tune_threshold_pr(y_eval, eval_probs, forecast_horizon)
    else:
        best_threshold = grid_threshold
        preds = (eval_probs > grid_threshold).astype(int)

    avg_acc, avg_prec, avg_rec, avg_f1, avg_spec = evaluate_metrics(y_eval, preds, forecast_horizon)

    result = {
        'type': window_type,
        'forecast_horizon': forecast_horizon,
        'look_back': look_back,
        'units': units,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs_run': epochs_run,
        'threshold': best_threshold,
        'accuracy': avg_acc,
        'precision': avg_prec,
        'recall': avg_rec,
        'f1_score': avg_f1,
        'specificity': avg_spec
    }
    print(
        f"[Classification - {window_type}] Metrics (horizon={forecast_horizon}): "
        f"Accuracy={avg_acc:.5f}, Precision={avg_prec:.5f}, Recall={avg_rec:.5f}, "
        f"F1={avg_f1:.5f}, Specificity={avg_spec:.5f}, EpochsRun={epochs_run}, "
        f"Threshold={best_threshold:.2f}"
    )
    return result


# ==============================================================================
# Processing Each Sliding Window (Classification)
# ==============================================================================
def process_window_classification(window_config, data, forecast_horizon, hyperparams, global_threshold=None,
                                  eval_split='test'):
    n_samples = len(data)
    # Determine training indices
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    # Determine evaluation indices based on eval_split ('validate' or 'test')
    eval_start = int(window_config[eval_split][0] * n_samples)
    eval_end = int(window_config[eval_split][1] * n_samples)

    train_raw = data[train_start:train_end]
    eval_raw = data[eval_start:eval_end]

    if len(train_raw) < hyperparams['look_back'] or len(eval_raw) < hyperparams['look_back']:
        print(f"Skipping {window_config['type']} - insufficient data in {eval_split} split")
        return None

    # Reshape data for LSTM input: (samples, 1)
    train_data = train_raw.reshape(-1, 1)
    eval_data = eval_raw.reshape(-1, 1)

    # Apply MinMax scaling to both training and evaluation data
    train_data_scaled, eval_data_scaled, _ = apply_minmax_scaling(train_data, eval_data)

    return optimize_and_train_classification(
        train_data_scaled, eval_data_scaled, forecast_horizon,
        window_config['type'] + "_" + eval_split, hyperparams, grid_threshold=global_threshold
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
        cols_to_show = ['type', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'threshold']
        print("\nFinal classification results:")
        print(results_df[cols_to_show].to_string(index=False))
    else:
        print("No valid classification results generated.")


if __name__ == "__main__":
    main()
