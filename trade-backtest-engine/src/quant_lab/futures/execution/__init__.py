"""Futures live execution (Milestone 4).

Planned: a broker-agnostic ``FuturesBrokerAdapter`` interface with a paper
adapter first, then real adapters behind optional imports (Tradovate REST,
TradeStation, tastytrade, NinjaTrader, IBKR via ``ib_async``) so no single
broker library is a hard dependency. Also the OCA bracket manager and the
``futures_state.db`` SQLite state machine.
"""
