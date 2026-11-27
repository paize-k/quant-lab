"""Isolated spot forex engine and live bot.

Must never import from ``quant_lab.futures``. May import from ``quant_lab.core``.

Layout:
    config/      pair specs, UTC session windows, risk limits        (Milestone 1)
    data/        bid/ask-aware candle builder, rollover filter        (Milestone 3)
    strategies/  overlap breakout, MTF pullback, Asian range, hybrid  (Milestone 3)
    execution/   OANDA v20 / IBKR FX / paper adapters, SQLite state   (Milestone 4)
"""
