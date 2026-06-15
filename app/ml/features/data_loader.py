"""Raw market data loading utilities."""

from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.services.symbol_resolver import resolve_symbol

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def _require_columns(df: pd.DataFrame, required_columns: tuple[str, ...]) -> None:
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Raw OHLCV CSV is missing required columns: "
            + ", ".join(missing_columns)
        )


def load_raw_ohlcv(symbol: str) -> pd.DataFrame:
    """Load and clean raw OHLCV data for a supported symbol."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    csv_path = Path(settings.RAW_DATA_DIR) / f"{canonical_symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw OHLCV file not found for {canonical_symbol}: {csv_path}"
        )

    df = pd.read_csv(csv_path)
    df.columns = [column.strip().lower() for column in df.columns]
    _require_columns(df, REQUIRED_OHLCV_COLUMNS)

    cleaned = df.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = (
        cleaned.dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    return cleaned
