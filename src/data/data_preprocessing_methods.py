import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.seasonal import seasonal_decompose


# Function for reading CSV file
def read_csv(filepath):
    try:
        return pd.read_csv(filepath, parse_dates=['Datetime'])
    except ValueError:
        return pd.read_csv(filepath, parse_dates=['Date'])


# Function for removing duplicates
def remove_duplicates(df):
    return df.drop_duplicates()


# Function for handling missing values
def handle_missing_values(df, method='ffill'):
    if method == 'ffill':
        return df.ffill()

    elif method == 'mean':
        return df.fillna(df.mean())

    elif method == 'interpolation':
        return df.interpolate()

    elif method == 'knn':
        # Separate numeric and non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        non_numeric_df = df.select_dtypes(exclude=[np.number])

        # Apply KNN Imputer only on numeric columns
        imputer = KNNImputer(n_neighbors=5)
        imputed_numeric_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)

        # Combine back with non-numeric columns
        return pd.concat([imputed_numeric_df, non_numeric_df], axis=1)

    else:
        raise ValueError("Invalid method for handling missing values.")


# Function for removing outliers using the 3σ rule
def remove_outliers(df, columns):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df = df[(df[column] >= mean - 3 * std) & (df[column] <= mean + 3 * std)]
    return df


# Function for smoothing data using moving average
# New columns are created
def smooth_data(df, columns, window=3):
    for column in columns:
        df[column + '_smoothed'] = df[column].rolling(window=window, min_periods=1).mean()
    return df


# Function for robust regression
# New columns are created
def apply_robust_regression(df, columns):
    huber = HuberRegressor()
    for column in columns:
        x = df.index.values.reshape(-1, 1)
        y = df[column].values
        huber.fit(x, y)
        df[column + '_robust'] = huber.predict(x)
    return df


# TODO: For all passed KPSS tests, there are warnings
# C:\Users\zihao\Desktop\Projects\third-year-project\src\data\ADF_and_KPSS.py:35:
# InterpolationWarning: The test statistic is outside of the range of p-values available in the
# look-up table. The actual p-value is smaller than the p-value returned.

# Function for detrending data using differencing
# New columns are created
# handled non-stationarity
def detrend_data(df, columns):
    for column in columns:
        df[column + '_detrended'] = df[column].diff().fillna(0)
    return df


# Function for detrending using polynomial fitting
# New columns are created
# passed ADF but not KPSS
def polynomial_detrend(df, columns, degree=2):
    for column in columns:
        poly_fit = np.polyfit(df.index, df[column], degree)
        trend = np.polyval(poly_fit, df.index)
        df[column + '_polynomial_detrended'] = df[column] - trend
    return df


# Function for detrending data using moving average
# New columns are created
# handled non-stationarity
def detrend_with_moving_average(df, columns, window=12):
    for column in columns:
        trend = df[column].rolling(window=window, min_periods=1).mean()
        df[column + '_moving_avg_detrended'] = df[column] - trend
    return df


# Function for seasonal decomposition
# New columns are created
def decompose_seasonality(df, column, model='additive', period=12):
    decomposition = seasonal_decompose(df[column], model=model, period=period)
    df[column + '_trend'] = decomposition.trend # did not handle non-stationarity
    df[column + '_seasonal'] = decomposition.seasonal # handled non-stationarity
    df[column + '_residual'] = decomposition.resid # handled non-stationarity
    return df


# Function for normalizing data using MinMaxScaler
def normalize_data(df, columns, method='minmax'):
    if method == 'minmax':
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])

    elif method == 'zscore':
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])

    elif method == 'log':
        # Ensure no zero or negative values for log transform
        df[columns] = np.log1p(df[columns] - df[columns].min() + 1)

    elif method == 'unit':
        # Unit normalization (L2 normalization)
        df[columns] = df[columns].apply(lambda x: x / np.sqrt(np.sum(x ** 2)), axis=0)

    else:
        raise ValueError("Invalid normalization method. Choose 'minmax', 'zscore', 'log', or 'unit'.")

    return df


# Function for generating synthetic data points (Data Augmentation)
def augment_data(df, columns, factor=0.1):
    for column in columns:
        noise = np.random.normal(0, factor * df[column].std(), len(df))
        df[column + '_augmented'] = df[column] + noise
    return df
