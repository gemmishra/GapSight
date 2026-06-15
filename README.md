# GapSight

GapSight is a backend-first machine learning system for predicting pre-market
opening behavior. The current MVP supports BANKNIFTY only, while keeping symbol
resolution, metadata, APIs, and model workflows extensible.

The API returns model-based predictions only. It does not create fake or random
predictions.

## Planned Predictions

- Direction: `GAP_UP`, `FLAT`, or `GAP_DOWN`
- Predicted gap percentage
- Predicted gap points
- Expected opening price and opening range

## Roadmap

Future versions may add NIFTY, FINNIFTY, individual stocks, prompt-based symbol
selection, OpenClaw integration, and Discord or WhatsApp notifications.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and adjust settings if needed.

## Run

```bash
uvicorn app.main:app --reload
```

The API is available at `http://127.0.0.1:8000`, with interactive documentation
at `http://127.0.0.1:8000/docs`.

## API Endpoints

- `GET /` - project information
- `GET /health` - health status
- `GET /api/v1/supported-symbols` - active supported symbols
- `GET /api/v1/predict/BANKNIFTY` - latest BANKNIFTY model prediction
- `GET /api/v1/predict/BANKNIFTY/alert` - plain-text formatted alert

## Historical Data

Raw OHLCV data should be stored at:

```text
app/data/raw/BANKNIFTY.csv
```

The required CSV columns are:

```text
date, open, high, low, close, volume
```

Column names are normalized to lowercase during loading. Dates are parsed as
datetimes, OHLCV values are converted to numeric values, duplicate dates are
dropped, and rows are sorted by date ascending.

## Downloading Historical Data

Install dependencies first:

```bash
pip install -r requirements.txt
```

Download BANKNIFTY daily OHLCV data:

```bash
python scripts/download_data.py --symbol BANKNIFTY --start-date 2015-01-01
```

Then build the processed dataset:

```bash
python scripts/build_dataset.py --symbol BANKNIFTY
```

GapSight uses `yfinance` for development and research data downloads.
Internally, `BANKNIFTY` maps to the Yahoo Finance ticker `^NSEBANK`.

The downloaded raw file is saved at:

```text
app/data/raw/BANKNIFTY.csv
```

The processed training dataset is generated at:

```text
app/data/processed/BANKNIFTY_training.csv
```

## Build Dataset

Create the processed BANKNIFTY training dataset with:

```bash
python scripts/build_dataset.py --symbol BANKNIFTY
```

This reads `app/data/raw/BANKNIFTY.csv` and writes:

```text
app/data/processed/BANKNIFTY_training.csv
```

## Labels

Gap labels compare the current day's open with the previous close:

- `GAP_UP` when gap percent is greater than or equal to `0.40%`
- `GAP_DOWN` when gap percent is less than or equal to `-0.40%`
- `FLAT` when the gap is between those thresholds

The default flat threshold is +/- `0.40%`.

## Training Models

Train and compare BANKNIFTY models from the processed dataset:

```bash
python scripts/train_models.py --symbol BANKNIFTY
```

The classifier learns `gap_direction`, which is one of `GAP_UP`, `FLAT`, or
`GAP_DOWN`. It compares Logistic Regression, Random Forest, and XGBoost, then
selects by macro F1, weighted F1, and accuracy.

The regressor learns `gap_percent`, the percentage gap between the current open
and previous close. It compares Ridge, Random Forest, XGBoost, and a baseline
mean regressor, then selects the best real model by MAE, RMSE, and R2.

Training uses a chronological split, not a random split: the oldest 80% of rows
are used for training, and the newest 20% are used for testing. This better
matches market data workflows and avoids future rows leaking into older
training examples.

Generated files:

- `app/models/BANKNIFTY_gap_direction_classifier.joblib`
- `app/models/BANKNIFTY_gap_size_regressor.joblib`
- `app/models/BANKNIFTY_dataset_diagnostics.json`
- `app/models/BANKNIFTY_classifier_metrics.json`
- `app/models/BANKNIFTY_classifier_model_comparison.json`
- `app/models/BANKNIFTY_regressor_metrics.json`
- `app/models/BANKNIFTY_regressor_model_comparison.json`
- `app/models/BANKNIFTY_walk_forward_validation.json`
- `app/models/BANKNIFTY_model_metadata.json`

## Model Evaluation

GapSight evaluates models with chronological market data only. The standard
training split uses older rows for training and newer rows for testing, and the
walk-forward report uses an expanding window: train on the past, test on the
next future chunk, then move forward.

Classifier evaluation compares Logistic Regression, Random Forest, and XGBoost.
Class imbalance is handled with balanced class weights or sample weights where
supported, and the selected classifier is chosen by macro F1, then weighted F1,
then accuracy.

Regressor evaluation compares Ridge, Random Forest, XGBoost, and a baseline
mean regressor. The selected real regressor is chosen by MAE, then RMSE, then
R2. Reports also show whether the selected regressor beats the baseline MAE.

Useful commands:

```bash
python scripts/build_dataset.py --symbol BANKNIFTY
python scripts/train_models.py --symbol BANKNIFTY
python scripts/model_report.py --symbol BANKNIFTY
```

## Prediction API

Prepare data, features, and trained models:

```bash
python scripts/download_data.py --symbol BANKNIFTY --start-date 2015-01-01
python scripts/build_dataset.py --symbol BANKNIFTY
python scripts/train_models.py --symbol BANKNIFTY
uvicorn app.main:app --reload
```

Then request:

```text
http://127.0.0.1:8000/api/v1/predict/BANKNIFTY
```

For the plain-text alert only:

```text
http://127.0.0.1:8000/api/v1/predict/BANKNIFTY/alert
```

The classifier predicts the gap direction: `GAP_UP`, `FLAT`, or `GAP_DOWN`.
The regressor predicts the gap percentage. Expected open is calculated from the
latest available previous close, and the opening range uses the saved regressor
MAE.

The current gap-size prediction is a baseline estimate, not a guaranteed
forecast. The response includes model-quality flags and explanation notes so
weak or unstable model behavior is visible.

### User-Facing Prediction Fields

Direction confidence is converted into labels:

- `HIGH` when confidence is at least `0.70`
- `MEDIUM` when confidence is at least `0.50` and below `0.70`
- `LOW` when confidence is below `0.50`

Low-confidence predictions should be treated cautiously because the classifier
is not expressing a strong probability preference among direction classes.

The prediction response also includes friendly direction wording, gap-size
interpretation, reliability warnings, and a `formatted_alert` text block. That
alert is intended for future OpenClaw, Discord, or WhatsApp reuse, but no
notification integration is active yet.

## Notifications

GapSight can send the real formatted prediction alert to Discord through a
webhook. Add these values to `.env`:

```text
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
ENABLE_DISCORD_NOTIFICATIONS=true
```

Send an alert from the CLI:

```bash
python scripts/send_alert.py --symbol BANKNIFTY --channel discord
```

Or send through the API:

```text
POST http://127.0.0.1:8000/api/v1/alerts/BANKNIFTY/send?channel=discord
```

Only Discord is active right now. The notification service is structured so
OpenClaw, WhatsApp, Telegram, and other channels can be added later without
changing prediction generation.
