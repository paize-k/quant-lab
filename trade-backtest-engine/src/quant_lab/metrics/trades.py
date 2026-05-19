# src/quant_lab/metrics/trades.py

import pandas as pd
import numpy as np
from quant_lab.features.indicators import rolling_volatility, sma


def generate_trade_log(bt: pd.DataFrame, vol_window: int = 20, trend_window: int = 20) -> pd.DataFrame:
    """
    Build a trade log from a backtest result DataFrame.
    One row per trade, containing only trade-level data.
    """
    pos = bt["position"]
    close = bt["close"]

    prev_pos = pos.shift(1).fillna(0)

    entry_mask = (pos != 0) & (pos != prev_pos)
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
        while exit_cursor < len(exits) and exits[exit_cursor] <= entry_time:
            exit_cursor += 1

        if exit_cursor < len(exits):
            exit_time = exits[exit_cursor]
            exit_cursor += 1
        else:
            exit_time = bt.index[-1]

        paired_entries.append(entry_time)
        paired_exits.append(exit_time)
        paired_entry_prices.append(close.loc[entry_time])
        paired_exit_prices.append(close.loc[exit_time])
        paired_entry_pos.append(pos.loc[entry_time])

    entry_prices = np.array(paired_entry_prices)
    exit_prices  = np.array(paired_exit_prices)
    directions   = np.array(paired_entry_pos)

    log_return = np.where(
        directions == 1,
        np.log(exit_prices / entry_prices),
        np.log(entry_prices / exit_prices)
    )

    holding_bars = [
        bt.index.get_loc(exit_time) - bt.index.get_loc(entry_time)
        for entry_time, exit_time in zip(paired_entries, paired_exits)
    ]

    vol          = rolling_volatility(bt["log_return"], window=vol_window)
    trend_ma     = sma(bt["close"], window=trend_window)
    trend_str    = (bt["close"] - trend_ma) / trend_ma
    vol_pct      = vol.rank(pct=True)

    tl = pd.DataFrame({
        "Entry Time":            paired_entries,
        "Exit Time":             paired_exits,
        "Direction":             ["Long" if p == 1 else "Short" for p in paired_entry_pos],
        "Position":              directions,
        "Entry Price":           entry_prices,
        "Exit Price":            exit_prices,
        "Log Return":            log_return,
        "Return %":              (np.exp(log_return) - 1) * 100,
        "Duration Bars":         holding_bars,
        "Duration Days":         [(ex - en).days for en, ex in zip(paired_entries, paired_exits)],
        "Trend Strength":        [trend_str.loc[t] for t in paired_entries],
        "Volatility at Entry":   [vol.loc[t]       for t in paired_entries],
        "Volatility Percentile": [vol_pct.loc[t]   for t in paired_entries],
    })

    return tl


def evaluate_trades(tl: pd.DataFrame) -> pd.Series:
    """
    Compute summary statistics from a trade log produced by generate_trade_log().
    Returns a Series of scalar metrics.
    """
    log_ret = tl["Log Return"].values

    wins   = log_ret[log_ret > 0]
    losses = log_ret[log_ret < 0]

    total_trades  = len(log_ret)
    win_rate      = float(np.mean(log_ret > 0) * 100)
    loss_rate     = float(np.mean(log_ret < 0) * 100)
    avg_win       = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss      = float(losses.mean()) if len(losses) > 0 else 0.0
    profit_factor = float(-avg_win / avg_loss) if avg_loss < 0 else np.inf
    expected_ret  = float((win_rate / 100) * avg_win + (loss_rate / 100) * avg_loss)
    avg_duration  = float(tl["Duration Bars"].mean())
    best_trade    = float(tl["Return %"].max())
    worst_trade   = float(tl["Return %"].min())

    tm =  pd.Series({
        "Total Trades":    total_trades,
        "Win Rate %":      win_rate,
        "Loss Rate %":     loss_rate,
        "Average Win":     avg_win,
        "Average Loss":    avg_loss,
        "Profit Factor":   profit_factor,
        "Expected Return": expected_ret,
        "Avg Duration Bars": avg_duration,
        "Best Trade %":    best_trade,
        "Worst Trade %":   worst_trade,
    })
    return tm