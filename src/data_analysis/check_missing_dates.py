import pandas as pd

# File path
FILE_PATH = "../../data/raw/USDEUR=X_max_1d.csv"

# Load the data
data = pd.read_csv(FILE_PATH)

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Drop duplicate dates
data = data.drop_duplicates(subset='Date')

# Create a DataFrame with all dates within the range of the data
all_dates = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
full_df = pd.DataFrame({'Date': all_dates})

# Merge with the original data to identify missing dates
merged_df = full_df.merge(data, on='Date', how='left', indicator=True)
missing_dates = merged_df[merged_df['_merge'] == 'left_only']['Date']

# Check how many missing dates are weekends
weekend_missing = missing_dates[missing_dates.dt.weekday >= 5]  # 5=Saturday, 6=Sunday
non_weekend_missing = missing_dates[missing_dates.dt.weekday < 5]  # Missing dates that are weekdays

# Results
print(f"Total missing dates: {len(missing_dates)}")
print(f"Missing weekend dates: {len(weekend_missing)}")
print(f"Missing non-weekend dates (weekdays): {len(non_weekend_missing)}")

# Check if all missing dates are weekends
if len(non_weekend_missing) == 0:
    print("All missing dates are weekends.")
else:
    print(f"Not all missing dates are weekends. {len(non_weekend_missing)} weekday(s) are missing.")
