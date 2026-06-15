import pandas as pd
import pytest

from app.ml.training.training_utils import (
    chronological_train_test_split,
    get_feature_columns,
    load_training_dataset,
)


def test_feature_column_selection_excludes_metadata_and_targets() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=3),
            "symbol": ["BANKNIFTY"] * 3,
            "previous_close": [1.0, 2.0, 3.0],
            "ma_5": [1.0, 2.0, 3.0],
            "note": ["a", "b", "c"],
            "gap_direction": ["FLAT", "GAP_UP", "GAP_DOWN"],
            "gap_percent": [0.0, 1.0, -1.0],
            "gap_points": [0.0, 10.0, -10.0],
        }
    )

    assert get_feature_columns(df) == ["previous_close", "ma_5"]


def test_chronological_split_keeps_newest_rows_in_test() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-05", "2025-01-01", "2025-01-03", "2025-01-02", "2025-01-04"]
            ),
            "feature": [5, 1, 3, 2, 4],
        }
    )

    train_df, test_df = chronological_train_test_split(df, test_size=0.4)

    assert train_df["date"].astype(str).tolist() == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
    ]
    assert test_df["date"].astype(str).tolist() == ["2025-01-04", "2025-01-05"]


def test_missing_processed_dataset_gives_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.ml.training.training_utils.settings.PROCESSED_DATA_DIR",
        str(tmp_path),
    )

    with pytest.raises(FileNotFoundError, match="Processed training dataset not found"):
        load_training_dataset("BANKNIFTY")
