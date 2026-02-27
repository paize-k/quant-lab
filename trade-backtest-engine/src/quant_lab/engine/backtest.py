import numpy as np
import pandas as pd
from quant_lab.features.indicators import sma, ema, rolling_volatility, simple_returns, log_returns


def backtest_log_returns(close_prices: pd.Series, position: pd.Series, cost_per_trade: float = 0.0) -> pd.DataFrame:
    # --- Type checks ---
    if not isinstance(close_prices, pd.Series):
        raise TypeError("close_prices must be a pandas Series")

    if not isinstance(position, pd.Series):
        raise TypeError("position must be a pandas Series")

    # --- Index alignment check ---
    if not close_prices.index.equals(position.index):
        raise ValueError("close_prices and position must have the same index")
    
    # --- Cost per trade check ---
    if not (0.0 <= cost_per_trade < 1.0):
        raise ValueError("cost_per_trade must be in [0, 1)")

    # --- Empty data check ---
    if close_prices.empty:
        raise ValueError("close_prices is empty")

    if position.empty:
        raise ValueError("position is empty")
    
    allowed = {0, 1}
    values = set(position.dropna().unique())
    if not values.issubset(allowed):
        raise ValueError(f"position contains invalid values: {values - allowed}")


    log_ret = log_returns(close_prices)  # Calculate log returns, dropping NaN values from the start
    position_shifted = position.shift(1).fillna(0)  # Shift position to align with returns
    trade = (position - position_shifted).abs()  # Identify trades (changes in position)
    cost_log = trade * np.log(1-cost_per_trade)  # Log cost of trading
    raw_strat = position_shifted * log_ret  # Raw strategy returns without costs
    strat_log = raw_strat + cost_log  # Strategy returns after accounting for costs
    cum_log = strat_log.cumsum()  # Cumulative log returns 
    equity = np.exp(cum_log)  # Equity curve from cumulative log returns

    bt = {
        'close': close_prices,
        'log_return': log_ret,
        'position': position,
        'position_shifted': position_shifted,
        'trade': trade,
        'cost_log': cost_log,
        'strat_log': strat_log,
        'equity': equity

    }
    
    bt = pd.DataFrame(bt)

    # Placeholder return for now
    return bt