import numpy as np
import pandas as pd

def generate_trade_log(bt: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a trade log DataFrame from backtest results.
    
    Parameters:
    - bt: DataFrame containing backtest results with columns 'position' and 'trade'.
    
    Returns:
    - A DataFrame containing a log of trades executed, including entry and exit points, position sizes, and timestamps.
    """
    close = bt["close"] 
    position = bt["position"]
    position_shifted = bt["position_shifted"]  # Shift position to align with returns
    trade = bt["trade"]

    entry_signal = (position_shifted == 0) & (position == 1)  # Entry when position goes from 0 to 1
    exit_signal = (position_shifted == 1) & (position == 0)  # Exit when position goes from 1 to 0
    entry_times = bt.index[entry_signal]
    exit_times  = bt.index[exit_signal]

    print("entries:", len(entry_times), "exits:", len(exit_times))
    return entry_times, exit_times