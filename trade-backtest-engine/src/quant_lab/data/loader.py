import yfinance as yf
import pandas as pd

def load_prices(ticker, start_date, end_date) -> pd.DataFrame:
    """
    Load historical stock data for a given ticker and date range.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        raise ValueError(
            f"No data returned for {ticker}. Check ticker symbol or date range."
        )

    
    df.index = pd.to_datetime(df.index)  # Ensure datetime index

    df = df.sort_index()  # Ensures the DataFrame is sorted by date

    df = df[~df.index.duplicated(keep="first")]  # Remove duplicate dates

    return df