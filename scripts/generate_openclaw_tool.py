"""Generate the OpenClaw tool spec for GapSight."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings

TOOL_SPEC_PATH = PROJECT_ROOT / "openclaw" / "gapsight_tool.json"


def normalize_base_url(base_url: str) -> str:
    """Normalize a base URL by removing trailing slashes."""
    return base_url.strip().rstrip("/")


def build_tool_spec(base_url: str) -> dict[str, Any]:
    """Build the OpenClaw tool spec dictionary."""
    normalized_base_url = normalize_base_url(base_url)
    return {
        "name": "gapsight_predict",
        "description": (
            "Predict BANKNIFTY pre-market gap direction, gap size, and opening range."
        ),
        "method": "GET",
        "url_template": (
            f"{normalized_base_url}/api/v1/openclaw/predict/{{symbol}}"
        ),
        "alert_url_template": (
            f"{normalized_base_url}/api/v1/openclaw/predict/{{symbol}}/alert"
        ),
        "parameters": {
            "symbol": {
                "type": "string",
                "required": True,
                "default": "BANKNIFTY",
                "examples": [
                    "BANKNIFTY",
                    "BANK NIFTY",
                    "NIFTY BANK",
                ],
            }
        },
        "auth": {
            "type": "bearer",
            "required_when": "ENABLE_OPENCLAW_AUTH=true",
            "header": "Authorization",
            "token_env": "OPENCLAW_API_TOKEN",
        },
        "response_usage": (
            "For normal JSON endpoint, use formatted_alert field as the "
            "user-facing answer. For alert endpoint, return the plain text "
            "response directly."
        ),
    }


def write_tool_spec(base_url: str, path: Path = TOOL_SPEC_PATH) -> dict[str, Any]:
    """Write the generated tool spec and return generation details."""
    spec = build_tool_spec(base_url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return {
        "path": path,
        "base_url": normalize_base_url(base_url),
        "spec": spec,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OpenClaw tool spec.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for GapSight, e.g. local, ngrok, or production URL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url or settings.OPENCLAW_BASE_URL
    result = write_tool_spec(base_url)
    json_endpoint = result["spec"]["url_template"].replace("{symbol}", "BANKNIFTY")
    alert_endpoint = result["spec"]["alert_url_template"].replace(
        "{symbol}",
        "BANKNIFTY",
    )

    print("Generated OpenClaw tool spec:")
    print(f"Path: {result['path'].relative_to(PROJECT_ROOT)}")
    print(f"Base URL: {result['base_url']}")
    print(f"JSON endpoint: {json_endpoint}")
    print(f"Alert endpoint: {alert_endpoint}")
    print(f"Auth enabled: {str(settings.ENABLE_OPENCLAW_AUTH).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
