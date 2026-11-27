# Milestone 1: codebase audit and scaffold

## What exists today

`src/quant_lab` is a **vectorised, single-asset** research backtester built
around pandas Series:

| Module | What it does |
|---|---|
| `data/loader.py` | `load_prices(ticker, start, end)` wraps `yfinance.download`, sorts and de-duplicates a naive daily index. Columns come back as yfinance's `(Price, Ticker)` MultiIndex. |
| `features/indicators.py` | `simple_returns`, `log_returns`, `sma`, `ema`, `rolling_volatility`. |
| `strategies/*` | `sma_crossover_position`, `ema_crossover_position`, `donchian_breakout_position` return a position Series in {-1, 0, 1}. |
| `engine/backtest.py` | `backtest_log_returns(close, position, cost_per_trade, borrow_cost)` -> DataFrame of `strat_log`, `equity` (starts at 1.0), `drawdown`, `trade`. |
| `metrics/performance.py` | `evaluate_performance` -> total return, CAGR, vol, Sharpe, buy-and-hold Sharpe, max drawdown, trade count. |
| `metrics/trades.py` | `generate_trade_log`, `evaluate_trades` (win rate, avg win/loss, profit factor, expectancy). |

There is **no Bar type, no order/fill model, and no portfolio or ledger
class**. Equity is a unit-less curve, costs are a fraction per trade, and
there is no notion of contracts, units, ticks, pips, margin or sessions.

## Repo vs uploaded folder

The GitHub repo (`main` @ `8920b1d`) and the uploaded
`Quant Lab/trade-backtest-engine/` folder have the same code. The only
differences are Windows CRLF line endings in the uploaded copy.

## Findings worth knowing before Milestone 2

1. **Fill timing.** `position_shifted * log_ret` means a signal computed on
   close `t` is filled at close `t`. That is not look-ahead, but it is
   optimistic versus the new rule (fill at `t+1` open or a verified intra-bar
   stop/limit). Expect the new engines to show slightly worse results than
   the vectorised backtester on the same idea.
2. **Trade counting.** `evaluate_performance` counts `abs(trade).sum()`, so a
   long -> short flip counts as 2. `generate_trade_log` pairs it as one exit
   plus one entry. The two numbers will not match.
3. **Hard-coded 252.** `backtest_log_returns` assumes 252 periods per year for
   borrow cost, and `rolling_volatility` defaults to it. Wrong for intraday
   futures/FX bars, so the new metrics take `periods_per_year` explicitly.
4. **Stale notebook import.** `01_data_explore.ipynb` still has a cell that
   imports `ma_crossover_position` from `sma_crossover`, which no longer exists.
5. **`generate_trade_log` prints** entry/exit counts on every call, which will
   be noisy inside WFO and Monte Carlo loops.
6. **Filename typo.** `strategies/donchain_breakout.py` (kept as-is so the
   notebook keeps working).
7. **Build artefacts were committed** (`__pycache__/*.pyc` for Python 3.14 and
   `quant_lab.egg-info/`). This PR untracks them and adds a `.gitignore`.

## How the new engines build on it

* `core.data.normalize_ohlcv` / `bars_from_dataframe` take the existing
  `load_prices` output (MultiIndex or flat, naive or aware) and produce
  tz-aware `Bar` objects, so the same data source feeds the new engines.
* The existing indicators stay the indicator library; the Milestone 2/3
  strategies will reuse `sma`, `ema` and `rolling_volatility`.
* The existing metrics are the starting point for the shared tear sheet in
  `core` (Milestone 2), extended with Sortino, Calmar, profit factor,
  expectancy and fees/spread paid.
* Nothing existing was changed or removed, so the notebook and the
  vectorised backtester work exactly as before.

## Placement

The new packages live inside the existing installable package as
`quant_lab.core`, `quant_lab.futures` and `quant_lab.forex`, rather than as
new top-level folders. That keeps one `pip install -e .`, lets the engines
import the existing indicators, and keeps your notebook imports valid.
`tests/test_isolation.py` fails the build if `futures` and `forex` ever
import each other, or if `core` imports either.

## Fees, margins and spreads

Tick sizes, tick values, contract months and pip sizes follow exchange and
market convention and live on the specs. Everything that depends on the
broker lives in a swappable profile:

* `quant_lab.futures.config.fees.FuturesBrokerProfile`: commission,
  exchange/NFA/clearing fees, slippage, day/initial/maintenance margin.
* `quant_lab.forex.config.costs.ForexCostProfile`: spread, commission,
  swap per 100k per night, margin rate, rollover spread widening.

`get_contract(symbol)` and `get_pair(symbol)` apply `PAPER_DEFAULT`, a
conservative profile researched on 2026-11-27 from published retail figures
(sources are in each module's docstring). Pass `profile=YOUR_BROKER` once you
have real numbers, or `profile=None` for zero-cost specs. MGC and 6E exchange
fees could not be found and are estimates marked `verified=False`; FX swap
rates are from February 2026 and should be replaced with live rates.
