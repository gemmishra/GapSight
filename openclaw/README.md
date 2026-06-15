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

## Local Testing

1. Start FastAPI:

   ```bash
   uvicorn app.main:app --reload
   ```

2. Generate the local tool spec:

   ```bash
   python scripts/generate_openclaw_tool.py --base-url http://127.0.0.1:8000
   ```

3. Test the plain-text endpoint:

   ```text
   http://127.0.0.1:8000/api/v1/openclaw/predict/BANKNIFTY/alert
   ```

## Cloud/OpenClaw Testing With Ngrok

1. Start FastAPI:

   ```bash
   uvicorn app.main:app --reload
   ```

2. Start ngrok:

   ```bash
   ngrok http 8000
   ```

3. Generate the tool spec with your ngrok URL:

   ```bash
   python scripts/generate_openclaw_tool.py --base-url https://your-ngrok-url.ngrok-free.app
   ```

4. Use the generated `openclaw/gapsight_tool.json` in OpenClaw.

If `ENABLE_OPENCLAW_AUTH=false`, no bearer token is needed. If
`ENABLE_OPENCLAW_AUTH=true`, OpenClaw must send:

```text
Authorization: Bearer <OPENCLAW_API_TOKEN>
```

## Example Prompts

- Predict Bank Nifty
- What is today's BANKNIFTY pre-market outlook?
- Send me the GapSight Bank Nifty alert
