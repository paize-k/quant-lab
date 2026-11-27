"""Isolated CME futures engine and live bot.

Must never import from ``quant_lab.forex``. May import from ``quant_lab.core``.

Layout:
    config/      contract specs, CME sessions, risk limits        (Milestone 1)
    data/        continuous-contract roll + bar normaliser          (Milestone 2)
    strategies/  trend, VWAP reversion, ORB, hybrid meta-strategy   (Milestone 2)
    execution/   ib_async / Tradovate / paper adapters, OCA, SQLite (Milestone 4)
"""
