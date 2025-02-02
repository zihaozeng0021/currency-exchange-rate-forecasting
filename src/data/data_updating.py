import yfinance as yf
import os
import pandas as pd
import subprocess

input_dir = '../../data/raw'

def update_data():
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Extract parameters from filename
            try:
                ticker, period, interval = filename.replace('.csv', '').split('_')
            except ValueError:
                print(f"Filename {filename} does not match expected pattern. Skipping.")
                continue

            output_path = os.path.join(input_dir, filename)

            # Download data with group_by='column' to avoid multi-level columns
            data = yf.download(ticker, period=period, interval=interval, group_by='column')

            # If data is empty or just 1 row, skip
            if len(data) <= 1:
                print(f"Download failed or no sufficient data for {ticker}. No update performed.")
                continue

            # --- Flatten columns if multi-index ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ["_".join(col).strip() for col in data.columns]

            # Rename "Close_<ticker>" to "Close" for single-ticker data
            for col in list(data.columns):
                if col.lower().startswith("close_"):
                    data.rename(columns={col: "Close"}, inplace=True)

            # If there's still no "Close" column, skip
            if "Close" not in data.columns:
                print(f"No 'Close' column found for {ticker}. Columns are: {list(data.columns)}")
                continue

            # Keep only 'Close'
            data = data[["Close"]]

            # Give the index a name so it shows up as "Date" in CSV
            data.index.name = "Date"

            # Save updated data
            data.to_csv(output_path, header=True, index=True)
            print(f"Data updated and saved to {output_path}")



if __name__ == "__main__":
    update_data()
    subprocess.run(['python', 'data_preprocessing.py'])
