import yfinance as yf
import pandas as pd

def load_data(ticker, start_date, end_date):
    """
    Load historical stock data for a given ticker and date range.
    
    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL').
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    df = df.sort_index()  # Ensures the DataFrame is sorted by date

    df = df.dropna()  # Drop rows with missing values

    return df