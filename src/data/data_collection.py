import yfinance as yf
import os
import pandas as pd

def main():
    ticker = input("Enter the ticker (e.g. USDEUR=X): ").strip()
    period = input("Enter the period (e.g. max): ").strip()
    interval = input("Enter the interval (e.g. 1d): ").strip()

    outputfile = f"{ticker}_{period}_{interval}"
    output_path = f"../../data/raw/{outputfile}.csv"

    # Prompt for overwrite (default to 'yes')
    if os.path.exists(output_path):
        confirm = input(f"The file {output_path} already exists. Overwrite it? [Y/n]: ").strip().lower()
        if confirm == 'no':
            print("Operation cancelled.")
            return

    # Download data (note group_by='column')
    data = yf.download(ticker, period=period, interval=interval, group_by='column')

    # Check if any data was returned
    if len(data) <= 1:
        print("Download failed or not enough data. No file was saved.")
        return

    # 1) If columns are multi-index for some reason, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ["_".join(col).strip() for col in data.columns]

    # 2) If you see columns like "Close_USDEUR=X", rename them to "Close"
    close_cols = [col for col in data.columns if col.lower().startswith("close_")]
    if len(close_cols) == 1:
        data.rename(columns={close_cols[0]: "Close"}, inplace=True)
    elif "Close" not in data.columns:
        print("No 'Close' column found in data. Columns are:", list(data.columns))
        return

    # 3) Keep only the 'Close' column
    data = data[["Close"]]

    # Print a sample of the data
    print(data)

    # 4) Save to CSV (index=True => "Date,Close")
    data.to_csv(output_path, index=True)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    main()
