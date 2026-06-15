from datetime import datetime, timedelta

import pandas as pd
import pytest

from app.ml.features.data_loader import load_raw_ohlcv
from app.ml.features.feature_builder import BASIC_FEATURE_COLUMNS, build_basic_features
from app.ml.features.label_builder import build_gap_labels
from app.services import symbol_resolver


def _sample_ohlcv(rows: int = 25) -> pd.DataFrame:
    start_date = datetime(2025, 1, 1)
    records = []
    for index in range(rows):
        open_price = 1000 + index * 10
        records.append(
            {
                "date": start_date + timedelta(days=index),
                "open": open_price,
                "high": open_price + 20,
                "low": open_price - 10,
                "close": open_price + 5,
                "volume": 100000 + index,
            }
        )
    return pd.DataFrame(records)


def test_gap_labels_calculate_points_percent_and_direction() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000,
            },
            {
                "date": "2025-01-02",
                "open": 101.0,
                "high": 103.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 1100,
            },
            {
                "date": "2025-01-03",
                "open": 101.8,
                "high": 104.0,
                "low": 100.0,
                "close": 101.0,
                "volume": 1200,
            },
            {
                "date": "2025-01-04",
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 99.0,
                "volume": 1300,
            },
        ]
    )

    labels = build_gap_labels(df)

    assert labels["gap_points"].round(2).tolist() == [1.0, -0.2, -1.0]
    assert labels["gap_percent"].round(2).tolist() == [1.0, -0.2, -0.99]
    assert labels["gap_direction"].tolist() == ["GAP_UP", "FLAT", "GAP_DOWN"]


def test_symbol_resolver_aliases() -> None:
    assert symbol_resolver.resolve_symbol("BANK NIFTY") == "BANKNIFTY"
    assert symbol_resolver.resolve_symbol("NIFTY BANK") == "BANKNIFTY"
    assert symbol_resolver.resolve_symbol("NIFTYBANK") == "BANKNIFTY"
    assert symbol_resolver.resolve_symbol("BANKNIFTY") == "BANKNIFTY"
    assert symbol_resolver.resolve_symbol("NIFTY") is None


def test_build_basic_features_includes_expected_columns() -> None:
    features = build_basic_features(_sample_ohlcv())

    assert set(BASIC_FEATURE_COLUMNS).issubset(features.columns)
    assert not features.empty


def test_load_raw_ohlcv_raises_for_missing_columns(tmp_path, monkeypatch) -> None:
    raw_csv = tmp_path / "BANKNIFTY.csv"
    raw_csv.write_text("date,open,high,low,close\n2025-01-01,1,2,1,2\n")
    monkeypatch.setattr("app.ml.features.data_loader.settings.RAW_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="missing required columns: volume"):
        load_raw_ohlcv("BANKNIFTY")
