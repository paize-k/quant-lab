import pandas as pd
import numpy as np

def donchian_breakout_position(close_prices, lookback: int, threshold=0.0):
    """Generate trading positions based on Donchian Channel breakout strategy.  
    Returns a Series where 1 indicates a long position and 0 indicates no position. The position is determined by the breakout of the close price above the upper channel or below the lower channel.
    """
    
    if isinstance(close_prices, pd.DataFrame):
        if close_prices.shape[1] != 1:
            raise ValueError("close_prices must be a Series or a single-column DataFrame")
        close_prices = close_prices.iloc[:, 0]

    if lookback <= 0:
        raise ValueError("Lookback period must be a positive integer greater than zero.")
    
    # Calculate Donchian Channels
    upper_channel = close_prices.shift(1).rolling(lookback).max()
    lower_channel = close_prices.shift(1).rolling(lookback).min()
    
    position = pd.Series(
        np.where(close_prices > upper_channel * (1 + threshold), 1,
        np.where(close_prices < lower_channel * (1 - threshold), -1, np.nan)),
        index=close_prices.index,
        dtype=float
    )

    position[upper_channel.isna()] = 0.0  # Flat during warm-up (channels not yet defined)
    position = position.ffill().fillna(0.0)  # Forward fill positions, then fill any remaining NaNs with 0


    return position

# Example usage:
# data = pd.read_csv('historical_data.csv')
# strategy_df = donchian_breakout(data, lookback=20)
