import os
import yfinance as yf
import pandas as pd

# Create the output folder if it doesn't exist
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# Define the date range
start_date = "2000-01-03"
end_date = "2019-03-01"

# Define the tickers and output filenames.
# For JPY/USD, we use 'USDJPY=X' and invert the close prices to get JPY/USD.
ticker_files = {
    "EURUSD=X": "EURUSD.csv",
    #"GBPUSD=X": "GBPUSD.csv",
    #"AUDUSD=X": "AUDUSD.csv",
    #"NZDUSD=X": "NZDUSD.csv",
    #"USDJPY=X": "JPYUSD.csv"  # Inverted to get JPY/USD
}

for ticker, file_name in ticker_files.items():
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print(f"No data found for {ticker}. Skipping.")
        continue

    # For JPY/USD, invert the closing prices (since we get USD/JPY)
    if ticker == "USDJPY=X":
        data["Close"] = 1 / data["Close"]

    # Reset the index so that the date becomes a column.
    data_reset = data.reset_index()

    # If the DataFrame has multi-index columns, flatten them by taking the first level.
    if isinstance(data_reset.columns, pd.MultiIndex):
        data_reset.columns = data_reset.columns.get_level_values(0)

    # Select only the Date and Close columns.
    close_data = data_reset[["Date", "Close"]]

    # Save the CSV file to the data folder.
    file_path = os.path.join(output_folder, file_name)
    close_data.to_csv(file_path, index=False)
    print(f"Saved data to {file_path}")
