"""Market data download and persistence services."""

from datetime import date
from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.ml.features.data_loader import REQUIRED_OHLCV_COLUMNS
from app.services.symbol_resolver import resolve_symbol
from app.utils.constants import YFINANCE_TICKERS


def resolve_yfinance_ticker(symbol: str) -> str:
    """Resolve an internal GapSight symbol to its Yahoo Finance ticker."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    ticker = YFINANCE_TICKERS.get(canonical_symbol)
    if ticker is None:
        raise ValueError(f"No yfinance ticker configured for {canonical_symbol}")

    return ticker


def clean_yfinance_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Format raw yfinance data to GapSight's OHLCV CSV contract."""
    if df.empty:
        raise ValueError("No OHLCV data was downloaded")

    raw = df.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(column[0]).strip().lower() for column in raw.columns]
    else:
        raw.columns = [str(column).strip().lower() for column in raw.columns]

    raw = raw.reset_index()
    raw.columns = [str(column).strip().lower() for column in raw.columns]
    raw = raw.rename(columns={"datetime": "date"})

    missing_columns = sorted(set(REQUIRED_OHLCV_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(
            "Downloaded OHLCV data is missing required columns: "
            + ", ".join(missing_columns)
        )

    cleaned = raw.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.date

    price_columns = ["open", "high", "low", "close"]
    for column in [*price_columns, "volume"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = (
        cleaned.dropna(subset=["date", *price_columns])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )

    if cleaned.empty:
        raise ValueError("No usable OHLCV rows remained after cleaning")

    return cleaned


def download_ohlcv_from_yfinance(
    symbol: str,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance for a supported symbol."""
    ticker = resolve_yfinance_ticker(symbol)
    download_end_date = end_date or date.today().isoformat()

    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "yfinance is required to download market data. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    raw_df = yf.download(
        ticker,
        start=start_date,
        end=download_end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw_df.empty:
        raise ValueError(
            f"No data downloaded for {symbol} ({ticker}) from "
            f"{start_date} to {download_end_date}"
        )

    return clean_yfinance_ohlcv(raw_df)


def save_raw_ohlcv(symbol: str, df: pd.DataFrame) -> str:
    """Save cleaned raw OHLCV data to the expected raw data path."""
    canonical_symbol = resolve_symbol(symbol)
    if canonical_symbol is None:
        raise ValueError(f"Unsupported symbol: {symbol}")

    output_dir = Path(settings.RAW_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{canonical_symbol}.csv"

    output_df = df.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    output_df.to_csv(output_path, index=False)

    return str(output_path)


def download_and_save_raw_ohlcv(
    symbol: str,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> str:
    """Download and save raw OHLCV data for a supported symbol."""
    df = download_ohlcv_from_yfinance(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    return save_raw_ohlcv(symbol, df)
