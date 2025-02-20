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
from keras.optimizers import Nadam

# =================== Reinforcement Learning Components ===================

# Environment for Hyperparameter Tuning using Classification
class HyperparameterTuningEnvironment:
    def __init__(self, train_scaled, validate_scaled, scaler, forecast_horizon, search_space, n_candidates=50):
        self.train_scaled = train_scaled
        self.validate_scaled = validate_scaled
        self.scaler = scaler  # Not used for inverse scaling in classification
        self.forecast_horizon = forecast_horizon
        self.search_space = search_space
        self.n_candidates = n_candidates
        self.candidates = []
        for i in range(n_candidates):
            look_back = random.randint(search_space['look_back'][0], search_space['look_back'][1])
            units = random.randint(search_space['units'][0], search_space['units'][1])
            batch_size = random.randint(search_space['batch_size'][0], search_space['batch_size'][1])
            epochs = random.randint(search_space['epochs'][0], search_space['epochs'][1])
            learning_rate = random.uniform(search_space['learning_rate'][0], search_space['learning_rate'][1])
            self.candidates.append((look_back, units, batch_size, learning_rate, epochs))

    def reset(self):
        # Return default state, using first candidate's parameters (except epochs)
        state = list(self.candidates[0][:4])
        return state

    def step(self, action):
        candidate = self.candidates[action]
        look_back, units, batch_size, learning_rate, epochs = candidate

        X_train, y_train = create_dataset_classification(self.train_scaled, look_back, self.forecast_horizon,
                                                         threshold=0)
        X_val, y_val = create_dataset_classification(self.validate_scaled, look_back, self.forecast_horizon,
                                                     threshold=0)

        if len(X_train) == 0 or len(X_val) == 0:
            reward = -1000
        else:
            X_train = X_train.reshape((X_train.shape[0], look_back, 1))
            X_val = X_val.reshape((X_val.shape[0], look_back, 1))
            model = build_classification_model(look_back, units, self.forecast_horizon, learning_rate)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            y_val_pred = model.predict(X_val, verbose=0)
            best_threshold, preds = tune_threshold_pr(y_val, y_val_pred, self.forecast_horizon)
            try:
                reward = f1_score(y_val.flatten(), preds.flatten(), zero_division=0)
            except Exception as e:
                reward = -1000
        next_state = list(candidate[:4])
        done = True  # one-step episode
        return next_state, reward, done


# DQN Agent for Hyperparameter Tuning
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # e.g., 4
        self.action_size = action_size  # number of candidate hyperparameter configurations
        self.memory = []
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# RL Training Loop
def train_dqn_agent(env, time_limit=600, batch_size=32):
    state_size = 4  # [look_back, units, batch_size, learning_rate]
    action_size = len(env.candidates)
    agent = DQNAgent(state_size, action_size)
    start_time = time.time()
    episode = 0
    while time.time() - start_time < time_limit:
        state = env.reset()
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode += 1
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode {episode}: Reward = {reward:.5f}, Epsilon = {agent.epsilon:.5f}")
    candidate_states = np.array([list(c[:4]) for c in env.candidates])
    q_values = agent.model.predict(candidate_states, verbose=0)
    best_idx = np.argmax(np.max(q_values, axis=1))
    best_candidate = env.candidates[best_idx]
    return {
        'look_back': best_candidate[0],
        'units': best_candidate[1],
        'batch_size': best_candidate[2],
        'learning_rate': best_candidate[3],
        'epochs': best_candidate[4]
    }


def rl_search_hyperparameters(train_scaled, validate_scaled, scaler, forecast_horizon, search_space, time_limit=600):
    env = HyperparameterTuningEnvironment(train_scaled, validate_scaled, scaler, forecast_horizon, search_space,
                                          n_candidates=50)
    best_hp = train_dqn_agent(env, time_limit=time_limit, batch_size=32)
    print(f"Best hyperparameters from RL: {best_hp}")
    return best_hp


# =================== Main Entry Point ===================

def main():
    DATA_PATH = '../../../data/raw/USDEUR=X_max_1d.csv'
    CLASSIFICATION_CSV_PATH = 'results/LSTM_classification_rl.csv'
    FORECAST_HORIZONS_CLF = [1]

    search_space = {
        'epochs': (10, 50),
        'look_back': (30, 120),
        'units': (50, 200),
        'batch_size': (16, 128),
        'learning_rate': (0.0001, 0.01)
    }

    # Configure TensorFlow and seeds
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    warnings.filterwarnings('ignore')
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

    # Load and preprocess data
    df = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    data = df['Close'].values
    data = np.diff(data, n=1)  # smoothing

    windows = generate_sliding_windows()
    results = []
    total_start_time = time.time()

    for window in windows:
        print(f"\n=== Processing {window['type']} ===")
        n_samples = len(data)
        print(f"Training range: {window['train'][0] * 100:.1f}% - {window['train'][1] * 100:.1f}%")
        print(f"Validation range: {window['validate'][0] * 100:.1f}% - {window['validate'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        for horizon in FORECAST_HORIZONS_CLF:
            # Split data for RL tuning: training and validation splits
            train_start = int(window['train'][0] * n_samples)
            train_end = int(window['train'][1] * n_samples)
            validate_start = int(window['validate'][0] * n_samples)
            validate_end = int(window['validate'][1] * n_samples)
            test_start = int(window['test'][0] * n_samples)
            test_end = int(window['test'][1] * n_samples)

            train_raw = data[train_start:train_end]
            validate_raw = data[validate_start:validate_end]
            test_raw = data[test_start:test_end]

            train_data = train_raw.reshape(-1, 1)
            validate_data = validate_raw.reshape(-1, 1)
            test_data = test_raw.reshape(-1, 1)

            train_scaled, validate_scaled, scaler = apply_minmax_scaling(train_data, validate_data)

            print(f"\n--- RL Hyperparameter Tuning for forecast horizon {horizon} ---")
            best_hyperparams = rl_search_hyperparameters(train_scaled, validate_scaled, scaler, horizon, search_space,
                                                         time_limit=600)
            if best_hyperparams is None:
                print(f"No valid hyperparameters found for window {window['type']}.")
                continue

            print(f"Best hyperparameters for {window['type']}: {best_hyperparams}")
            print("Evaluating on test split with best hyperparameters...")
            result = process_window_classification(window, data, horizon, best_hyperparams, global_threshold=None,
                                                   eval_split='test')
            if result is not None:
                results.append(result)

    save_results(results, CLASSIFICATION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - total_start_time:.2f} seconds")



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


def build_classification_model(look_back, units, forecast_horizon, learning_rate):
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(units, return_sequences=False),
        Dense(forecast_horizon, activation='sigmoid')
    ])
    optimizer = Nadam(learning_rate=learning_rate)
    model.compile(loss='binary_focal_crossentropy', optimizer=optimizer)
    return model


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


def apply_minmax_scaling(train_data, test_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler


def optimize_and_train_classification(train_data, eval_data, forecast_horizon, window_type, hyperparams,
                                      grid_threshold=None):
    look_back = hyperparams['look_back']
    units = hyperparams['units']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epochs = hyperparams['epochs']

    X_train, y_train = create_dataset_classification(train_data, look_back, forecast_horizon, threshold=0)
    X_eval, y_eval = create_dataset_classification(eval_data, look_back, forecast_horizon, threshold=0)

    if len(X_train) == 0 or len(X_eval) == 0:
        print(f"[Classification - {window_type}] Insufficient data for parameters.")
        return None

    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_eval = X_eval.reshape((X_eval.shape[0], look_back, 1))

    model = build_classification_model(look_back, units, forecast_horizon, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    epochs_run = len(history.history['loss'])

    eval_probs = model.predict(X_eval, verbose=0)

    # Use PR curve–based threshold tuning on evaluation data
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
        f"[Classification - {window_type}] Metrics (horizon={forecast_horizon}): Accuracy={avg_acc:.5f}, F1={avg_f1:.5f}, Threshold={best_threshold:.2f}")
    return result

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

def process_window_classification(window_config, data, forecast_horizon, hyperparams, global_threshold=None,
                                  eval_split='test'):
    n_samples = len(data)
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    eval_start = int(window_config[eval_split][0] * n_samples)
    eval_end = int(window_config[eval_split][1] * n_samples)

    train_raw = data[train_start:train_end]
    eval_raw = data[eval_start:eval_end]

    if len(train_raw) < hyperparams['look_back'] or len(eval_raw) < hyperparams['look_back']:
        print(f"Skipping {window_config['type']} - insufficient data in {eval_split} split")
        return None

    train_data = train_raw.reshape(-1, 1)
    eval_data = eval_raw.reshape(-1, 1)
    train_data_scaled, eval_data_scaled, _ = apply_minmax_scaling(train_data, eval_data)

    return optimize_and_train_classification(train_data_scaled, eval_data_scaled, forecast_horizon,
                                             window_config['type'] + "_" + eval_split,
                                             hyperparams, grid_threshold=global_threshold)


def generate_sliding_windows():
    return [
        {'type': 'window_1', 'train': (0.0, 0.12), 'validate': (0.12, 0.16), 'test': (0.16, 0.2)},
        {'type': 'window_2', 'train': (0.16, 0.28), 'validate': (0.28, 0.32), 'test': (0.32, 0.36)},
        {'type': 'window_3', 'train': (0.32, 0.44), 'validate': (0.44, 0.48), 'test': (0.48, 0.52)},
        {'type': 'window_4', 'train': (0.48, 0.60), 'validate': (0.60, 0.64), 'test': (0.64, 0.68)},
        {'type': 'window_5', 'train': (0.64, 0.76), 'validate': (0.76, 0.80), 'test': (0.8, 0.84)},
        {'type': 'window_6', 'train': (0.8, 0.92), 'validate': (0.92, 0.96), 'test': (0.96, 1.0)}
    ]


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
