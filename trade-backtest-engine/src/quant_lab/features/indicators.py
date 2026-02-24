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

def ma_crossover_position(close_prices, fast_window, slow_window):
    """
    Generate trading positions based on moving average crossover strategy.
    Returns a Series where 1 indicates a long position and 0 indicates no position.
    """
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Window sizes must be positive integers greater than zero.")
    
    if fast_window >= slow_window:
        raise ValueError("Fast window must be smaller than slow window for a valid crossover strategy.")

    fast_ma = sma(close_prices, fast_window)
    slow_ma = sma(close_prices, slow_window)

    # Generate signals: 1 for long, -1 for short, 0 for no position
    signal = pd.Series(0, index=close_prices.index, dtype=int)
    signal[fast_ma > slow_ma] = 1 # Long-only signal 1 = log, 0 = flat

    position = signal.copy()

    return position