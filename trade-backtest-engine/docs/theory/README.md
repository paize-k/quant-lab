# Quant Lab theory notes

A teaching companion to the code. Each chapter explains the ideas behind what
has been built (intuition first, then the maths, a worked example, common
pitfalls and self-check questions) and points to the file where each idea
lives.

| Chapter | Covers |
|---|---|
| [0. How to review a pull request](00-how-to-review-a-pr.md) | A four-pass review method and a checklist for quant code |
| [1. Foundations](01-foundations.md) | Simple vs log returns, the bar-by-bar loop, next-bar fills, look-ahead bias, futures ticks and point value, fees, slippage, margin, sessions, continuous contracts and rolls |
| [2. Risk and sizing](02-risk-and-sizing.md) | Fixed-fractional sizing, R-multiples, expectancy, the 1.5R scale-out, circuit breakers |
| [3. Indicators and strategies](03-indicators-and-strategies.md) | EMA, Wilder smoothing, ATR, Donchian, Keltner, ADX, RSI(2), anchored VWAP and bands, ATR percentile; Trend Momentum, VWAP Reversion, ORB, the regime hybrid |
| [4. Measuring and validating](04-measuring-and-validating.md) | Tear sheet metrics (CAGR, Sharpe, Sortino, Calmar, MaxDD, profit factor, expectancy), walk-forward optimisation, Monte Carlo reshuffling, overfitting |
| [5. Forex preview](05-forex-preview.md) | Pips, pip value conversion (GBP account), bid/ask, spreads, swaps, UK margin, for Milestone 3 |

Code paths are relative to `trade-backtest-engine/`. Chapters 1 to 4 describe
the Milestone 2 code (PR #2), so some linked files only exist on `main` once
that PR is merged.

**Suggested order:** chapter 0, then 1 to 4 while reviewing PR #2, then 5
before Milestone 3.
