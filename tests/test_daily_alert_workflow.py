import pytest

from scripts import run_daily_alert


def _prediction(latest_data_date: str = "2026-06-16") -> dict:
    return {
        "symbol": "BANKNIFTY",
        "latest_data_date": latest_data_date,
        "formatted_alert": "BANKNIFTY Pre-Market Prediction",
    }


def test_dry_run_does_not_call_notification_sender(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(
        run_daily_alert,
        "download_and_save_raw_ohlcv",
        lambda symbol: calls.append(("download", symbol)) or "raw.csv",
    )
    monkeypatch.setattr(
        run_daily_alert,
        "save_training_dataset",
        lambda symbol: calls.append(("dataset", symbol)) or "training.csv",
    )
    monkeypatch.setattr(
        run_daily_alert,
        "predict_latest",
        lambda symbol: calls.append(("predict", symbol)) or _prediction(),
    )

    def fail_send(*_args, **_kwargs):
        raise AssertionError("notification sender should not be called")

    monkeypatch.setattr(run_daily_alert, "send_prediction_alert", fail_send)

    result = run_daily_alert.run_daily_alert_workflow(
        "BANKNIFTY",
        channel="discord",
        dry_run=True,
    )

    assert result["success"] is True
    assert result["notification_result"] is None
    assert calls == [
        ("download", "BANKNIFTY"),
        ("dataset", "BANKNIFTY"),
        ("predict", "BANKNIFTY"),
    ]
    assert "Dry run enabled. Notification not sent." in capsys.readouterr().out


def test_retrain_flag_calls_training_pipeline(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(run_daily_alert, "download_and_save_raw_ohlcv", lambda _s: "raw")
    monkeypatch.setattr(run_daily_alert, "save_training_dataset", lambda _s: "dataset")
    monkeypatch.setattr(run_daily_alert, "predict_latest", lambda _s: _prediction())
    monkeypatch.setattr(
        run_daily_alert,
        "train_all_models",
        lambda symbol: calls.append(("train", symbol)) or {"metadata_path": "meta.json"},
    )
    monkeypatch.setattr(
        run_daily_alert,
        "send_prediction_alert",
        lambda _symbol, channel: {
            "success": True,
            "message": "sent",
            "channel": channel,
            "symbol": "BANKNIFTY",
            "status_code": 204,
        },
    )

    result = run_daily_alert.run_daily_alert_workflow(
        "BANKNIFTY",
        channel="discord",
        retrain=True,
    )

    assert calls == [("train", "BANKNIFTY")]
    assert result["training_result"] == {"metadata_path": "meta.json"}


def test_normal_mode_calls_download_dataset_prediction_and_notification(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        run_daily_alert,
        "download_and_save_raw_ohlcv",
        lambda symbol: calls.append(("download", symbol)) or "raw.csv",
    )
    monkeypatch.setattr(
        run_daily_alert,
        "save_training_dataset",
        lambda symbol: calls.append(("dataset", symbol)) or "training.csv",
    )
    monkeypatch.setattr(
        run_daily_alert,
        "predict_latest",
        lambda symbol: calls.append(("predict", symbol)) or _prediction(),
    )
    monkeypatch.setattr(
        run_daily_alert,
        "send_prediction_alert",
        lambda symbol, channel: calls.append(("notify", symbol, channel))
        or {
            "success": True,
            "message": "sent",
            "channel": channel,
            "symbol": symbol,
            "status_code": 204,
        },
    )

    result = run_daily_alert.run_daily_alert_workflow("BANKNIFTY", channel="discord")

    assert result["success"] is True
    assert calls == [
        ("download", "BANKNIFTY"),
        ("dataset", "BANKNIFTY"),
        ("predict", "BANKNIFTY"),
        ("notify", "BANKNIFTY", "discord"),
    ]


def test_daily_workflow_errors_are_handled_cleanly(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["run_daily_alert.py", "--symbol", "NIFTY", "--channel", "discord"],
    )
    monkeypatch.setattr(
        run_daily_alert,
        "run_daily_alert_workflow",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("Unsupported symbol: NIFTY")),
    )

    exit_code = run_daily_alert.main()

    assert exit_code == 1
    assert "Error: Unsupported symbol: NIFTY" in capsys.readouterr().err


def test_market_data_date_warning(monkeypatch, capsys) -> None:
    monkeypatch.setattr(run_daily_alert, "download_and_save_raw_ohlcv", lambda _s: "raw")
    monkeypatch.setattr(run_daily_alert, "save_training_dataset", lambda _s: "dataset")
    monkeypatch.setattr(
        run_daily_alert,
        "predict_latest",
        lambda _s: _prediction(latest_data_date="2026-01-01"),
    )

    run_daily_alert.run_daily_alert_workflow(
        "BANKNIFTY",
        channel="discord",
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert "Latest available data date: 2026-01-01" in output
    assert "Market data may not include today yet" in output
