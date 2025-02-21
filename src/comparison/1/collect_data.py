import yfinance as yf

# Define the date range
start_date = "2017-12-18"
end_date = "2023-01-27"

# Define the tickers and the output CSV filenames.
# For JPY/USD, we use 'USDJPY=X' and then take the reciprocal.
ticker_files = {
    "EURUSD=X": "EURUSD.csv",
    "GBPUSD=X": "GBPUSD.csv",
    "AUDUSD=X": "AUDUSD.csv",
    "NZDUSD=X": "NZDUSD.csv",
    "USDJPY=X": "JPYUSD.csv"  # Will be inverted to get JPY/USD
}

for ticker, file_name in ticker_files.items():
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        print(f"No data found for {ticker}. Skipping.")
        continue

    # For the Japanese Yen, invert the USD/JPY closing prices to get JPY/USD
    if ticker == "USDJPY=X":
        data["Close"] = 1 / data["Close"]

    # Save only the 'Close' column as CSV
    close_data = data[["Close"]]
    close_data.to_csv(file_name)
    print(f"Saved data to {file_name}")
