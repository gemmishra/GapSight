# GapSight OpenClaw Integration

OpenClaw can call GapSight as an external capability for BANKNIFTY pre-market
gap outlooks.

## JSON Endpoint

```text
GET http://127.0.0.1:8000/api/v1/openclaw/predict/BANKNIFTY
```

The response is compact JSON with the selected direction label, confidence
label, gap estimate, expected opening range, model quality, reliability
warnings, and `formatted_alert`.

## Plain-Text Alert Endpoint

```text
GET http://127.0.0.1:8000/api/v1/openclaw/predict/BANKNIFTY/alert
```

Use this when the agent only needs a clean message to return directly to the
user.

## Optional Token Auth

Set these values in `.env`:

```text
OPENCLAW_API_TOKEN=your-token-here
ENABLE_OPENCLAW_AUTH=true
```

When auth is enabled, OpenClaw must send:

```text
Authorization: Bearer your-token-here
```

## Example Prompts

- Predict Bank Nifty
- What is today's BANKNIFTY pre-market outlook?
- Send me the GapSight Bank Nifty alert
