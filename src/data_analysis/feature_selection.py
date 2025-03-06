import os
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from datetime import datetime
import warnings

# Suppress warnings from the Granger causality tests
warnings.filterwarnings("ignore")


def load_data(filepath):
    return pd.read_csv(filepath, parse_dates=['Date'])


def standardize_adj_close(df, source_name=""):
    if 'Adj Close' not in df.columns and 'Close' in df.columns:
        print(f"Renaming 'Close' to 'Adj Close' in {source_name} data.")
        df.rename(columns={'Close': 'Adj Close'}, inplace=True)
    return df


def align_data(asset_df, usdeur_df):
    merged = pd.merge(usdeur_df, asset_df, on='Date', how='inner', suffixes=('_usd', '_asset'))
    return merged.dropna()


def perform_spearman_correlation(merged_df):
    return merged_df['Adj Close_usd'].corr(merged_df['Adj Close_asset'], method='spearman')


def perform_granger_tests(merged_df, maxlag=5):
    results = {}
    # USDEUR -> Asset
    gc_usdeur_to_asset = grangercausalitytests(merged_df[['Adj Close_asset', 'Adj Close_usd']], maxlag=maxlag,
                                               verbose=False)
    results['USDEUR_causes_asset'] = {lag: round(test[0]['ssr_ftest'][1], 4) for lag, test in
                                      gc_usdeur_to_asset.items()}

    # Asset -> USDEUR
    gc_asset_to_usdeur = grangercausalitytests(merged_df[['Adj Close_usd', 'Adj Close_asset']], maxlag=maxlag,
                                               verbose=False)
    results['Asset_causes_USDEUR'] = {lag: round(test[0]['ssr_ftest'][1], 4) for lag, test in
                                      gc_asset_to_usdeur.items()}

    return results


def significant_lags(pvalues, alpha=0.05):
    return [lag for lag, p in pvalues.items() if p < alpha]


def print_conclusion(asset, spearman_corr, granger_results):
    print("\n=== Conclusion for {} ===".format(asset.upper()))
    print(f"Spearman Correlation (USDEUR vs {asset}): {spearman_corr:.4f}")

    p_usdeur_asset = granger_results['USDEUR_causes_asset']
    p_asset_usdeur = granger_results['Asset_causes_USDEUR']

    sig_lags_usdeur_asset = significant_lags(p_usdeur_asset)
    sig_lags_asset_usdeur = significant_lags(p_asset_usdeur)

    if sig_lags_usdeur_asset:
        print(f"USDEUR Granger causes {asset} at lag(s): {', '.join(map(str, sig_lags_usdeur_asset))}")
    else:
        print(f"No significant Granger causality found for USDEUR causing {asset} (p ≥ 0.05 at all lags).")

    if sig_lags_asset_usdeur:
        print(f"{asset.capitalize()} Granger causes USDEUR at lag(s): {', '.join(map(str, sig_lags_asset_usdeur))}")
    else:
        print(f"No significant Granger causality found for {asset} causing USDEUR (p ≥ 0.05 at all lags).")

    if sig_lags_usdeur_asset and sig_lags_asset_usdeur:
        print("Result: Bidirectional Granger causality is suggested.")
    elif sig_lags_usdeur_asset:
        print(f"Result: USDEUR appears to Granger cause {asset}, but not vice versa.")
    elif sig_lags_asset_usdeur:
        print(f"Result: {asset.capitalize()} appears to Granger cause USDEUR, but not vice versa.")
    else:
        print("Result: No clear Granger causality in either direction.")


def main():
    # Define the directory containing the CSV files.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))

    # Load and standardize USDEUR data.
    usdeur_file = os.path.join(base_dir, "USDEUR=X_max_1d.csv")
    usdeur_df = load_data(usdeur_file)
    usdeur_df = standardize_adj_close(usdeur_df, source_name="USDEUR")

    # Asset CSV files.
    assets = {
        "oil": "oil.csv",
        "gold": "gold.csv",
        "ftse": "ftse.csv"
    }

    for asset, filename in assets.items():
        asset_file = os.path.join(base_dir, filename)
        asset_df = load_data(asset_file)
        asset_df = standardize_adj_close(asset_df, source_name=asset.upper())

        # Merge data based on Date.
        merged_df = align_data(asset_df, usdeur_df)
        print(f"\n=== Results for {asset.upper()} ===")
        print(f"Number of matching dates: {len(merged_df)}")

        # Compute Spearman correlation and Granger tests.
        spearman_corr = perform_spearman_correlation(merged_df)
        granger_results = perform_granger_tests(merged_df, maxlag=5)

        # Print p-values for reference.
        print("Granger Causality Test p-values (USDEUR causes asset):")
        for lag, pvalue in granger_results['USDEUR_causes_asset'].items():
            print(f"  Lag {lag}: p-value = {pvalue}")
        print("Granger Causality Test p-values (Asset causes USDEUR):")
        for lag, pvalue in granger_results['Asset_causes_USDEUR'].items():
            print(f"  Lag {lag}: p-value = {pvalue}")

        # Print a clear, summarized conclusion.
        print_conclusion(asset, spearman_corr, granger_results)


if __name__ == "__main__":
    main()
