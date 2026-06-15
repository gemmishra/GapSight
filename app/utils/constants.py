"""Shared constants and symbol metadata."""

SYMBOL_ALIASES: dict[str, str] = {
    "BANK NIFTY": "BANKNIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
    "BANKNIFTY": "BANKNIFTY",
}

SUPPORTED_SYMBOLS: list[dict[str, object]] = [
    {
        "symbol": "BANKNIFTY",
        "name": "Nifty Bank",
        "instrument_type": "INDEX",
        "exchange": "NSE",
        "model_available": False,
        "is_active": True,
    }
]

SUPPORTED_SYMBOL_IDS: frozenset[str] = frozenset(
    str(symbol["symbol"]) for symbol in SUPPORTED_SYMBOLS
)

YFINANCE_TICKERS: dict[str, str] = {
    "BANKNIFTY": "^NSEBANK",
}
