import os
import numpy as np
import data_preprocessing_methods

input_dir = '../../data/raw'
output_dir = '../../data/preprocessed'

# Main processing loop
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        filepath = os.path.join(input_dir, filename)

        # Read data
        df = data_preprocessing_methods.read_csv(filepath)

        # Remove duplicates
        df = data_preprocessing_methods.remove_duplicates(df)

        # Handle missing values
        # There are no missing values in the data
        #df = data_preprocess_methods.handle_missing_values(df, method='ffill')
        #df = data_preprocess_methods.handle_missing_values(df, method='mean')
        #df = data_preprocess_methods.handle_missing_values(df, method='interpolation')
        df = data_preprocessing_methods.handle_missing_values(df, method='knn')

        # Define columns to clean
        columns_to_clean = df.select_dtypes(include=[np.number]).columns

        # Remove outliers
        df = data_preprocessing_methods.remove_outliers(df, columns_to_clean)

        # Smooth data (using moving average)
        #df = data_preprocess_methods.smooth_data(df, columns_to_clean)

        # Apply robust regression to handle anomalies
        #df = data_preprocess_methods.apply_robust_regression(df, columns_to_clean)

        # Detrend data using differencing
        #df = data_preprocess_methods.detrend_data(df, columns_to_clean)

        # Detrend data using polynomial fitting
        #df = data_preprocess_methods.polynomial_detrend(df, columns_to_clean)

        # Detrend data using moving average
        #df = data_preprocess_methods.detrend_with_moving_average(df, columns_to_clean)

        # Decompose seasonality (e.g., on 'Close' column)
        #df = data_preprocess_methods.decompose_seasonality(df, 'Close')

        # Normalize data
        columns = df.select_dtypes(include=[np.number]).columns
        df = data_preprocessing_methods.normalize_data(df, columns)

        # Generate synthetic data points for augmentation
        #df = data_preprocess_methods.augment_data(df, columns_to_clean)

        # Save preprocessed data
        output_filename = f'preprocessed_{filename}'
        output_filepath = os.path.join(output_dir, output_filename)
        df.to_csv(output_filepath, index=False)

        print(f'Processed {filename} and saved to {output_filename}')
