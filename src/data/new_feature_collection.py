import os
from datetime import datetime
import yfinance as yf

def main():
    # Define the directory to store the raw data.
    # This path is relative to the script location.
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    os.makedirs(data_dir, exist_ok=True)

    # Define tickers for oil, gold, and FTSE 100.
    tickers = {
        "oil": "CL=F",     # WTI Crude Oil Futures
        "gold": "GC=F",    # Gold Futures
        "ftse": "^FTSE",   # FTSE 100 Index
    }

    # Define the date range for historical data.
    start_date = "2000-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Download data for each ticker and save as CSV.
    for name, ticker in tickers.items():
        print(f"Downloading {name} data for ticker {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date)
        csv_path = os.path.join(data_dir, f"{name}.csv")
        data.to_csv(csv_path)
        print(f"Saved {name} data to {csv_path}")

if __name__ == "__main__":
    main()
