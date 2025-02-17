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


# ====================== RL Components: Environment and Agent ======================

# --- Environment for Hyperparameter Tuning ---
class HyperparameterTuningEnvironment:
    def __init__(self, train_scaled, validate_scaled, scaler, forecast_horizon, search_space, n_candidates=50):
        self.train_scaled = train_scaled
        self.validate_scaled = validate_scaled
        self.scaler = scaler
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
        # Return a default state (here, we use the state representation of the first candidate).
        state = list(self.candidates[0][:4])  # state = [look_back, units, batch_size, learning_rate]
        return state

    def step(self, action):
        """
        Given an action (an index into the candidate list), train a model using the chosen hyperparameters
        and return the next state (the candidate's state representation), a reward (negative MSE), and done.
        """
        candidate = self.candidates[action]
        look_back, units, batch_size, learning_rate, epochs = candidate

        # Create training and validation datasets using candidate's look_back
        X_train, y_train = create_dataset_regression(self.train_scaled, look_back, self.forecast_horizon)
        X_val, y_val = create_dataset_regression(self.validate_scaled, look_back, self.forecast_horizon)

        # If there is insufficient data, return a large negative reward.
        if len(X_train) == 0 or len(X_val) == 0:
            reward = -1000
        else:
            X_train = X_train.reshape((X_train.shape[0], look_back, 1))
            X_val = X_val.reshape((X_val.shape[0], look_back, 1))
            model = build_regression_model(look_back, units, self.forecast_horizon, learning_rate)
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            y_val_pred = model.predict(X_val, verbose=0)
            y_val_pred_orig = inverse_standard_scaling(y_val_pred, self.scaler)
            y_val_orig = inverse_standard_scaling(y_val, self.scaler)
            mse_val = mean_squared_error(y_val_orig.flatten(), y_val_pred_orig.flatten())
            reward = -mse_val  # reward is negative MSE (we want to minimize MSE)

        next_state = list(candidate[:4])
        done = True  # one-step episode
        return next_state, reward, done


# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # e.g., 4 (look_back, units, batch_size, learning_rate)
        self.action_size = action_size  # number of candidate hyperparameter combinations
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
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
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


# --- RL Training Loop ---
def train_dqn_agent(env, time_limit=600, batch_size=32):
    state_size = 4  # [look_back, units, batch_size, learning_rate]
    action_size = len(env.candidates)
    agent = DQNAgent(state_size, action_size)
    start_time = time.time()
    episode = 0
    # Run episodes until the time limit is reached
    while time.time() - start_time < time_limit:
        state = env.reset()
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode += 1
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode {episode}: Reward = {reward:.9f}, Epsilon = {agent.epsilon:.9f}")
    # After training, evaluate Q-values for all candidate states and select the best.
    candidate_states = np.array([list(c[:4]) for c in env.candidates])
    q_values = agent.model.predict(candidate_states, verbose=0)
    # For each candidate, take the maximum Q-value; then select the candidate with highest Q-value.
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



# ====================== Main Function ======================
def main():
    # Define file paths and hyperparameters
    DATA_PATH = '../../../data/raw/USDEUR=X_max_1d.csv'
    REGRESSION_CSV_PATH = 'results/LSTM_regression_rl.csv'
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

    # Load the data and generate sliding window configurations
    data = load_data(DATA_PATH)
    windows = generate_sliding_windows()

    results = []
    overall_start = time.time()
    for window in windows:
        print(f"\n=== Processing {window['type']} ===")
        print(f"Training range: {window['train'][0] * 100:.1f}% - {window['train'][1] * 100:.1f}%")
        print(f"Validation range: {window['validate'][0] * 100:.1f}% - {window['validate'][1] * 100:.1f}%")
        print(f"Testing range: {window['test'][0] * 100:.1f}% - {window['test'][1] * 100:.1f}%")
        for horizon in FORECAST_HORIZONS_REG:
            # Process each window using RL for hyperparameter tuning
            result = process_window_regression(window, data, horizon, search_space, method='rl', time_limit=600)
            if result is not None:
                results.append(result)
    save_results_regression(results, REGRESSION_CSV_PATH)
    print(f"\nTotal execution time: {time.time() - overall_start:.2f} seconds")


def configure_tf():
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"GPU is available. Detected: {gpu_devices}")
    else:
        print("No GPU found. Running on CPU.")


def set_global_config(seed=42):
    warnings.filterwarnings('ignore')
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_data(data_path):
    df = pd.read_csv(data_path, index_col='Date', parse_dates=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df['Close'].values


def apply_standard_scaling(train_data, test_data):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler


def inverse_standard_scaling(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)


def generate_sliding_windows():
    return [
        {'type': 'window_1', 'train': (0.0, 0.12), 'validate': (0.12, 0.16), 'test': (0.16, 0.2)},
        {'type': 'window_2', 'train': (0.16, 0.28), 'validate': (0.28, 0.32), 'test': (0.32, 0.36)},
        {'type': 'window_3', 'train': (0.32, 0.44), 'validate': (0.44, 0.48), 'test': (0.48, 0.52)},
        {'type': 'window_4', 'train': (0.48, 0.60), 'validate': (0.60, 0.64), 'test': (0.64, 0.68)},
        {'type': 'window_5', 'train': (0.64, 0.76), 'validate': (0.76, 0.80), 'test': (0.8, 0.84)},
        {'type': 'window_6', 'train': (0.8, 0.92), 'validate': (0.92, 0.96), 'test': (0.96, 1.0)}
    ]


def create_dataset_regression(dataset, look_back=1, forecast_horizon=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X_seq = dataset[i: i + look_back, 0]
        y_seq = dataset[i + look_back: i + look_back + forecast_horizon, 0]
        X.append(X_seq)
        y.append(y_seq)
    return np.array(X), np.array(y)


def build_regression_model(look_back, units, forecast_horizon, learning_rate):
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(units, return_sequences=False),
        Dense(forecast_horizon, activation='linear')
    ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model


def train_and_evaluate_regression(train_data, test_data, forecast_horizon, window_type, hyperparams, scaler):
    look_back = hyperparams['look_back']
    units = hyperparams['units']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    epochs = hyperparams['epochs']

    X_train, y_train = create_dataset_regression(train_data, look_back, forecast_horizon)
    X_test, y_test = create_dataset_regression(test_data, look_back, forecast_horizon)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"[Regression - {window_type}] Insufficient data for the given parameters.")
        return None

    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    model = build_regression_model(look_back, units, forecast_horizon, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    epochs_run = len(history.history['loss'])

    y_pred_test = model.predict(X_test, verbose=0)
    y_pred_test_orig = inverse_standard_scaling(y_pred_test, scaler)
    y_test_orig = inverse_standard_scaling(y_test, scaler)

    y_test_flat = y_test_orig.flatten()
    y_pred_test_flat = y_pred_test_orig.flatten()
    mse_val = mean_squared_error(y_test_flat, y_pred_test_flat)
    mae_val = mean_absolute_error(y_test_flat, y_pred_test_flat)
    rmse_val = np.sqrt(mse_val)
    r2 = r2_score(y_test_flat, y_pred_test_flat)

    # Compute prediction intervals using training residuals
    y_train_pred = model.predict(X_train, verbose=0)
    y_train_orig = inverse_standard_scaling(y_train, scaler)
    y_train_pred_orig = inverse_standard_scaling(y_train_pred, scaler)
    residuals = y_train_orig.flatten() - y_train_pred_orig.flatten()
    sigma = np.std(residuals)
    z = 1.96  # 95% confidence
    lower_bound = y_pred_test_flat - z * sigma
    upper_bound = y_pred_test_flat + z * sigma
    coverage = np.mean((y_test_flat >= lower_bound) & (y_test_flat <= upper_bound)) * 100
    interval_width = np.mean(upper_bound - lower_bound)

    result = {
        'type': window_type,
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
        f"[Regression - {window_type}] Final metrics (horizon={forecast_horizon}): "
        f"MSE={mse_val:.9f}, MAE={mae_val:.9f}, RMSE={rmse_val:.9f}, R2={r2:.9f}, "
        f"Coverage={coverage:.2f}%, IntervalWidth={interval_width:.5f}, EpochsRun={epochs_run}"
    )
    return result


def process_window_regression(window_config, data, forecast_horizon, search_space, method='rl', time_limit=600):
    n_samples = len(data)
    train_start = int(window_config['train'][0] * n_samples)
    train_end = int(window_config['train'][1] * n_samples)
    validate_start = int(window_config['validate'][0] * n_samples)
    validate_end = int(window_config['validate'][1] * n_samples)
    test_start = int(window_config['test'][0] * n_samples)
    test_end = int(window_config['test'][1] * n_samples)

    train_raw = data[train_start:train_end]
    validate_raw = data[validate_start:validate_end]
    test_raw = data[test_start:test_end]

    if len(train_raw) < 10 or len(validate_raw) < 10 or len(test_raw) < 10:
        print(f"Skipping {window_config['type']} - insufficient data")
        return None

    train_raw = train_raw.reshape(-1, 1)
    validate_raw = validate_raw.reshape(-1, 1)
    test_raw = test_raw.reshape(-1, 1)

    scaler_train = StandardScaler()
    train_scaled = scaler_train.fit_transform(train_raw)
    validate_scaled = scaler_train.transform(validate_raw)

    if method == 'rl':
        best_hp = rl_search_hyperparameters(train_scaled, validate_scaled, scaler_train, forecast_horizon, search_space,
                                            time_limit=time_limit)
    else:
        print("Unknown hyperparameter search method.")
        return None

    train_val_raw = np.concatenate([train_raw, validate_raw], axis=0)
    scaler_final = StandardScaler()
    train_val_scaled = scaler_final.fit_transform(train_val_raw)
    test_scaled = scaler_final.transform(test_raw)

    result = train_and_evaluate_regression(train_val_scaled, test_scaled, forecast_horizon, window_config['type'],
                                           best_hp, scaler_final)
    if result is not None:
        result['best_hyperparameters'] = best_hp
    return result


def save_results_regression(results, csv_path):
    if results:
        results_df = pd.DataFrame(results)
        numeric_cols = results_df.select_dtypes(include=np.number).columns
        avg_row = results_df[numeric_cols].mean().to_dict()
        avg_row['type'] = 'average'
        results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        print("\nFinal regression results (including average metrics):")
        cols_to_show = ['type', 'mse', 'mae', 'rmse', 'r2_score', 'coverage_probability (%)', 'interval_width']
        print(results_df[cols_to_show].to_string(index=False))
    else:
        print("[Results] No regression results to save.")


if __name__ == "__main__":
    main()
