# src/quant_lab/metrics/trades.py

import pandas as pd
import numpy as np


def generate_trade_log(bt: pd.DataFrame) -> pd.DataFrame:
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

    tl = pd.DataFrame({
        "Entry Time":   paired_entries,
        "Exit Time":    paired_exits,
        "Direction":    ["Long" if p == 1 else "Short" for p in paired_entry_pos],
        "Entry Price":  paired_entry_prices,
        "Exit Price":   paired_exit_prices,
    })

    # P&L: long profits when price rises, short profits when price falls
    tl["Return %"] = (
        (tl["Exit Price"] - tl["Entry Price"]) / tl["Entry Price"]
        * [1 if p == 1 else -1 for p in paired_entry_pos]
        * 100
    )
    tl["Duration"] = tl["Exit Time"] - tl["Entry Time"]

    return tl