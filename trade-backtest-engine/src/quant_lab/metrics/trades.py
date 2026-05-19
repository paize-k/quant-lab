# src/quant_lab/metrics/trades.py

import pandas as pd
import numpy as np
from quant_lab.features.indicators import rolling_volatility, simple_returns, log_returns, sma


def generate_trade_log(bt: pd.DataFrame, vol_window: int = 20, trend_window: int = 20) -> pd.DataFrame:
    """
    Build a trade log from a backtest result DataFrame.

    Derives entries/exits from the 'position' column:
        - Entry: position changes from 0 → non-zero, or flips sign (e.g. -1 → 1)
        - Exit:  position changes from non-zero → 0, or flips sign
    """
    pos = bt["position"]
    close = bt["close"]

    prev_pos = pos.shift(1).fillna(0)

    # Entry: position becomes non-zero AND differs from previous
    entry_mask = (pos != 0) & (pos != prev_pos)

    # Exit: position returns to zero OR flips direction (close old leg)
    exit_mask  = (pos != prev_pos) & (prev_pos != 0)

    entries = bt.index[entry_mask].tolist()
    exits   = bt.index[exit_mask].tolist()

    print(f"entries: {len(entries)}, exits: {len(exits)}")

    paired_entries      = []
    paired_exits        = []
    paired_entry_prices = []
    paired_exit_prices  = []
    paired_entry_pos    = []

    exit_cursor = 0

    for entry_time in entries:
        # Skip exits that are on or before this entry
        while exit_cursor < len(exits) and exits[exit_cursor] <= entry_time:
            exit_cursor += 1

        if exit_cursor < len(exits):
            exit_time = exits[exit_cursor]
            exit_cursor += 1
        else:
            # Open trade — close at last bar
            exit_time = bt.index[-1]

        paired_entries.append(entry_time)
        paired_exits.append(exit_time)
        paired_entry_prices.append(close.loc[entry_time])
        paired_exit_prices.append(close.loc[exit_time])
        paired_entry_pos.append(pos.loc[entry_time])   # +1 long, -1 short

    entry_prices = np.array(paired_entry_prices)
    exit_prices = np.array(paired_exit_prices)
    directions = np.array(paired_entry_pos) # +1 for long, -1 for short

    log_return = np.where(
        directions == 1,  # Long
        np.log(exit_prices / entry_prices),  # Long (profit if price rises)
        np.log(entry_prices / exit_prices)   # Short (profit if price falls)
    )

    holding_bars = [
        bt.index.get_loc(exit_time) - bt.index.get_loc(entry_time)
        for entry_time, exit_time in zip(paired_entries, paired_exits)
    ]

    vol = rolling_volatility(bt["log_return"], window=vol_window)

    trend_ma = sma(bt["close"], window=trend_window)
    trend_strength = (bt["close"] - trend_ma) / trend_ma

    vol_percentile = vol.rank(pct=True)

    vol_at_entry = [vol.loc[time] for time in paired_entries]
    vol_pct_at_entry = [vol_percentile.loc[time] for time in paired_entries]
    trend_at_entry = [trend_strength.loc[time] for time in paired_entries]

    tl = pd.DataFrame({
        "Entry Time":   paired_entries,
        "Exit Time":    paired_exits,
        "Direction":    ["Long" if p == 1 else "Short" for p in paired_entry_pos],
        "Entry Price":  paired_entry_prices,
        "Exit Price":   paired_exit_prices,
        "Log Return":   log_return,
        "Return %": (np.exp(log_return - 1) * 100),  # To be calculated
        "Duration Bars": holding_bars,
        "Duration Days": [(exit - entry).days for entry, exit in zip(paired_entries, paired_exits)],
        "Trend Strength": trend_at_entry,
        "Volatility at entry": vol_at_entry,
        "Volatility Percentile": vol_pct_at_entry
    })

    return tl