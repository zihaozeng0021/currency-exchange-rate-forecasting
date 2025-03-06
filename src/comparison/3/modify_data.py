import pandas as pd
import os

# Create the output directory "data" if it doesn't exist
os.makedirs('data', exist_ok=True)

# Read the CSV file with the specified columns
df = pd.read_csv('data/DAT_MT_EURUSE_M1_2017-2020.csv',
                 header=None,
                 names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

# Combine Date and Time into a single DateTime column
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')

# Set the DateTime column as the index and keep only the Close column
df.set_index('DateTime', inplace=True)
df = df[['Close']]

# Sort the index to ensure it is monotonic
df.sort_index(inplace=True)

# Filter data for the respective date ranges
df_10min_range = df.loc['2017':'2018']
df_30min_range = df.loc['2019':'2020']

# Resample the data to a 10-minute interval using the last Close value in each interval
df_10min = df_10min_range.resample('10min').last()

# Resample the data to a 30-minute interval using the last Close value in each interval
df_30min = df_30min_range.resample('30min').last()

# Save the resampled data to CSV files
df_10min.to_csv('data/ERU_USD_10min.csv')
df_30min.to_csv('data/ERU_USD_30min.csv')

print("Done")
