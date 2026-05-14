import numpy as np
import pandas as pd
from quant_lab.features.indicators import sma, ema, rolling_volatility, simple_returns, log_returns

def ma_crossover_position(close_prices, fast_window, slow_window):
    """
    Generate trading positions based on moving average crossover strategy.
    Returns a Series where 1 indicates a long position and 0 indicates no position and -1 indicates a short position (if implemented). The position is determined by the crossover of the fast and slow moving averages.
    """

    if isinstance(close_prices, pd.DataFrame):
        if close_prices.shape[1] != 1:
            raise ValueError("close_prices must be a Series or a single-column DataFrame")
        close_prices = close_prices.iloc[:, 0]

    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("Window sizes must be positive integers greater than zero.")
    
    if fast_window >= slow_window:
        raise ValueError("Fast window must be smaller than slow window for a valid crossover strategy.")

    # 1. Calculate MAs
    fast_ma = sma(close_prices, fast_window)
    slow_ma = sma(close_prices, slow_window)

    # 2. Generate Raw Signal
    # Logic: 1 if Fast > Slow, -1 if Fast < Slow, else 0
    signal = np.where(fast_ma < slow_ma, 1, 0)  # Start with long/flat signal
    signal = np.where(fast_ma > slow_ma, -1, signal)
    
    # Convert to Series to keep index alignment
    position = pd.Series(signal, index=close_prices.index)

    # 3. Handle Warm-up: Set signals to 0 where slow_ma is NaN
    position[slow_ma.isna()] = 0

    return position