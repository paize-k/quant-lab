import numpy as np
import pandas as pd

def simple_returns(prices: pd.Series) -> pd.Series:
    """
    Simple returns: r_t = (P_t / P_{t-1}) - 1
    """
    return prices.pct_change().dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Log returns: r_t = ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1)).dropna()

def sma(data: pd.Series, window: int) -> pd.Series:
    """
    Calculate the Simple Moving Average (SMA) for a given data series and window size.
    """

    if window <= 0:
        raise ValueError("Window size must be a positive integer greater than zero.")
    
    return data.rolling(window=window, min_periods = window).mean()

def ema(data: pd.Series, span: int) -> pd.Series:
    """
    Calculate the Exponential Moving Average (EMA) for a given data series and span.
    """
    if span <= 0:
        raise ValueError("Span must be a positive integer greater than zero.")

    return data.ewm(span=span, adjust=False).mean()

def rolling_volatility(returns: pd.Series, window: int, annualise: bool = True, periods_per_year: int = 252) -> pd.Series:
    """
    Calculate the rolling volatility (standard deviation of returns) for a given data series and window size.
    """
    if window <= 0:
        raise ValueError("Window size must be a positive integer greater than zero.")

    vol = returns.rolling(window=window, min_periods = window).std()

    if annualise:
        vol *= np.sqrt(periods_per_year)  # Annualise assuming 252 trading days in a year

    return vol

