import numpy as np
import pandas as pd

def evaluate_performance(bt: pd.DataFrame, asset_log_returns: pd.Series, periods_per_year: int = 252):
    """
    Evaluate the performance of a trading strategy and compare it against 
    a baseline Buy & Hold Sharpe ratio.
    
    Parameters:
    - bt: DataFrame containing backtest results with columns 'equity' and 'strat_log'.
    - asset_log_returns: pd.Series of the raw asset's daily log returns (the baseline).
    - periods_per_year: Number of trading periods in a year (default is 252).
    """

    equity = bt["equity"]
    years = len(bt) / periods_per_year

    returns = bt["strat_log"].dropna()  # Strategy returns after costs
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    CAGR = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1 if years > 0 else 0)
    annualized_return = CAGR

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    vol = float(sigma * np.sqrt(periods_per_year)) if sigma > 0 else 0.0  # Annualized volatility
    sharpe_ratio = float((mu / sigma) * np.sqrt(periods_per_year) if sigma > 0 else np.nan)

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = float(drawdown.min())

    total_trades = int(np.abs(bt["trade"]).sum())

    # --- 🚀 BASELINE BUY & HOLD SHARPE CALCULATION ---
    # Align the asset returns to match the exact backtest time frame
    clean_asset_returns = asset_log_returns.loc[returns.index].dropna()
    
    asset_mu = clean_asset_returns.mean()
    asset_sigma = clean_asset_returns.std(ddof=1)
    
    baseline_sharpe = float((asset_mu / asset_sigma) * np.sqrt(periods_per_year) if asset_sigma > 0 else np.nan)
    # ------------------------------------------------

    performance_metrics = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Equity": float(equity.iloc[-1]),
        "Volatility": vol,
        "Strategy Sharpe Ratio": sharpe_ratio,
        "Baseline Sharpe Ratio": baseline_sharpe,  # Added here
        "Max Drawdown": max_drawdown,
        "Total Trades Executed": total_trades
    }
    
    performance_metrics = pd.Series(performance_metrics)

    return performance_metrics