import pandas as pd

from app.services.market_data_service import (
    clean_yfinance_ohlcv,
    resolve_yfinance_ticker,
    save_raw_ohlcv,
)


def test_banknifty_maps_to_yfinance_ticker() -> None:
    assert resolve_yfinance_ticker("BANKNIFTY") == "^NSEBANK"
    assert resolve_yfinance_ticker("BANK NIFTY") == "^NSEBANK"
    assert resolve_yfinance_ticker("NIFTY BANK") == "^NSEBANK"


def test_clean_yfinance_ohlcv_produces_required_columns() -> None:
    raw_df = pd.DataFrame(
        {
            "Open": ["100.0", "101.0", None],
            "High": ["105.0", "106.0", "107.0"],
            "Low": ["99.0", "100.0", "101.0"],
            "Close": ["104.0", "105.0", "106.0"],
            "Volume": ["1000", "1100", "1200"],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-03"]),
    )
    raw_df.index.name = "Date"

    cleaned = clean_yfinance_ohlcv(raw_df)

    assert cleaned.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
    assert cleaned["date"].astype(str).tolist() == ["2025-01-01", "2025-01-02"]
    assert cleaned["open"].tolist() == [101.0, 100.0]
    assert cleaned["volume"].tolist() == [1100, 1000]


def test_save_raw_ohlcv_writes_expected_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.market_data_service.settings.RAW_DATA_DIR",
        str(tmp_path),
    )
    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 99.0,
                "close": 104.0,
                "volume": 1000,
            }
        ]
    )

    saved_path = save_raw_ohlcv("BANK NIFTY", df)

    assert saved_path == str(tmp_path / "BANKNIFTY.csv")
    saved_df = pd.read_csv(saved_path)
    assert saved_df.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
