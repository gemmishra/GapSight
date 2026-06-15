import json

from app.core.config import settings
from scripts.generate_openclaw_tool import build_tool_spec, write_tool_spec


def test_config_default_openclaw_base_url() -> None:
    assert settings.OPENCLAW_BASE_URL == "http://127.0.0.1:8000"


def test_generator_creates_tool_spec_file(tmp_path) -> None:
    output_path = tmp_path / "openclaw" / "gapsight_tool.json"

    result = write_tool_spec(
        "http://127.0.0.1:8000",
        path=output_path,
    )

    assert result["path"] == output_path
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["name"] == "gapsight_predict"


def test_generated_json_uses_provided_base_url() -> None:
    spec = build_tool_spec("https://your-ngrok-url.ngrok-free.app/")

    assert spec["url_template"] == (
        "https://your-ngrok-url.ngrok-free.app/api/v1/openclaw/predict/{symbol}"
    )


def test_generated_json_contains_json_and_alert_endpoints() -> None:
    spec = build_tool_spec("http://127.0.0.1:8000")

    assert spec["url_template"] == (
        "http://127.0.0.1:8000/api/v1/openclaw/predict/{symbol}"
    )
    assert spec["alert_url_template"] == (
        "http://127.0.0.1:8000/api/v1/openclaw/predict/{symbol}/alert"
    )


def test_generated_json_includes_banknifty_examples() -> None:
    spec = build_tool_spec("http://127.0.0.1:8000")
    examples = spec["parameters"]["symbol"]["examples"]

    assert examples == ["BANKNIFTY", "BANK NIFTY", "NIFTY BANK"]
