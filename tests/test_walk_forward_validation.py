from app.ml.training.walk_forward_validation import build_walk_forward_folds


def test_walk_forward_folds_use_expanding_train_and_future_test() -> None:
    folds = build_walk_forward_folds(
        row_count=100,
        min_train_ratio=0.60,
        test_window_ratio=0.05,
        step_ratio=0.05,
    )

    assert folds[0] == {
        "train_start": 0,
        "train_end": 60,
        "test_start": 60,
        "test_end": 65,
    }
    assert folds[1]["train_end"] == 65
    assert folds[1]["test_start"] == 65
    assert folds[-1]["test_end"] == 100
