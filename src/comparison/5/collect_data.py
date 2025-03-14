import os
import yfinance as yf
import pandas as pd

# Create the output folder if it doesn't exist
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# Group 1: March 2016 to February 2017
start_date_group1 = "2016-03-01"
end_date_group1 = "2017-02-28"
group1_ticker_files = {
    "AUDJPY=X": "AUDJPY.csv",  # AUD/JPY
    "AUDNZD=X": "AUDNZD.csv",  # AUD/NZD
    "AUDUSD=X": "AUDUSD.csv",  # AUD/USD
    "CADJPY=X": "CADJPY.csv",  # CAD/JPY
    "EURAUD=X": "EURAUD.csv",  # EUR/AUD
    "EURCAD=X": "EURCAD.csv",  # EUR/CAD
    "EURCSK=X": "EURCSK.csv",  # EUR/CSK
    "EURNOK=X": "EURNOK.csv",  # EUR/NOK
    "GBPAUD=X": "GBPAUD.csv",  # GBP/AUD
    "NZDUSD=X": "NZDUSD.csv",  # NZD/USD
    "USDCAD=X": "USDCAD.csv",  # USD/CAD
    "USDNOK=X": "USDNOK.csv",  # USD/NOK
    "USDJPY=X": "USDJPY.csv",  # USD/JPY
    "USDSGD=X": "USDSGD.csv",  # USD/SGD
    "USDZAR=X": "USDZAR.csv",  # USD/ZAR
    "EURGBP=X": "EURGBP.csv"  # EUR/GBP
}

# Group 2: June 2013 to May 2014
start_date_group2 = "2013-06-01"
end_date_group2 = "2014-05-31"
group2_ticker_files = {
    "EURUSD=X": "EURUSD.csv",  # EUR/USD
    "EURJPY=X": "EURJPY.csv",  # EUR/JPY
    "GBPCHF=X": "GBPCHF.csv",  # GBP/CHF
    "GBPUSD=X": "GBPUSD.csv"  # GBP/USD
}


def download_and_save_data(ticker_files, start_date, end_date):
    for ticker, file_name in ticker_files.items():
        print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for {ticker}. Skipping.")
            continue

        # Reset the index so that the date becomes a column
        data_reset = data.reset_index()
        # Flatten multi-index columns if necessary
        if isinstance(data_reset.columns, pd.MultiIndex):
            data_reset.columns = data_reset.columns.get_level_values(0)

        # Select only the Date and Close columns
        close_data = data_reset[["Date", "Close"]]

        # Save the CSV file to the data folder
        file_path = os.path.join(output_folder, file_name)
        close_data.to_csv(file_path, index=False)
        print(f"Saved data to {file_path}")


# Download data for Group 1
download_and_save_data(group1_ticker_files, start_date_group1, end_date_group1)

# Download data for Group 2
download_and_save_data(group2_ticker_files, start_date_group2, end_date_group2)
