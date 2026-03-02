import numpy as np
import pandas as pd

def evaluate_performance(bt: pd.DataFrame, periods_per_year: int = 252):

    """
    Evaluate the performance of a trading strategy based on backtest results.
    
    Parameters:
    - bt: DataFrame containing backtest results with columns 'equity' and 'trade'.
    - periods_per_year: Number of trading periods in a year (default is 252 for daily data).
    
    Returns:
    - A dictionary containing performance metrics such as total return, annualized return, and total trades executed.
    """

    equity = bt["equity"]
    years = len(bt) / periods_per_year

    returns = bt["strat_log"].dropna()  # Strategy returns after costs
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    CAGR = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1 if years > 0 else 0
    annualized_return = CAGR

    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rfr = 0.0  # Risk-free rate, can be adjusted as needed

    vol = sigma * np.sqrt(periods_per_year)  # Annualized volatility
    sharpe_ratio = (mu/ sigma) * np.sqrt(periods_per_year) if sigma > 0 else np.nan

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    total_trades = bt["trade"].sum()
    
    performance_metrics = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Volatility": vol,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Total Trades Executed": total_trades
    }
    
    performance_metrics = pd.Series(performance_metrics)

    return performance_metrics