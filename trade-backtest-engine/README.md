# Quant Lab

## Overview
A modular quantitative research and backtesting platform for equity markets.

## Architecture
- Data ingestion
- Feature engineering
- Strategy signals
- Backtesting engine
- Performance metrics
- Visualization

## Roadmap
- [x] Data loading & exploration
- [ ] Backtesting engine v1
- [ ] Risk metrics
- [ ] Multi-asset portfolios
- [ ] ML-based strategies

## Futures and forex engines (in progress)

Two isolated engines are being added on top of the research backtester:

- `quant_lab.core`: shared `Bar`/`Signal` models, the `BaseStrategy.on_bar(bar, state) -> Signal` contract, time windows and risk-based sizing math.
- `quant_lab.futures`: CME futures (MES, MNQ, MGC, 6E) contract specs, Globex/RTH sessions and risk limits.
- `quant_lab.forex`: spot FX pair specs (pip size, pip value conversion, swaps), UTC session windows and risk limits.

`futures` and `forex` never import each other. See `docs/milestone-1-audit.md` and `docs/milestone-2-futures.md` (futures backtester, strategies, walk-forward and Monte Carlo).

## Current Results
(Coming soon)

## How to Run

```bash
pip install -e ".[dev]"
pytest
```
