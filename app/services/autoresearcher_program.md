# GlucoAssist Forecast Research Program

## Goal

Improve the glucose forecasting model used in GlucoAssist.
The current model is Ridge regression. Your job is to propose, implement, and evaluate
alternative approaches — one at a time — against a fixed validation set drawn from the
user's local SQLite CGM database.

A candidate model is **promoted** only if it beats all three MAE baselines simultaneously.

## Current Baseline (to beat)

| Horizon | MAE (mg/dL) |
|---------|-------------|
| 30 min  | 17.74       |
| 60 min  | 25.18       |
| 120 min | 28.60       |

These were measured on a dataset of ~25,653 training samples at ~5-minute resolution
(approximately 88 days of continuous CGM data).

## Data Available

All data lives in the local SQLite database at `DATABASE_PATH` (default `/data/glucoassist.db`).

Relevant tables:
- `glucose_readings` — timestamp, glucose_mg_dl, trend
- `garmin_metrics` — date, resting_hr, weight_kg, sleep_hours, stress_level (may be sparse)

A 5-minute cadence should be assumed. Readings may have occasional gaps — handle them
gracefully (forward-fill up to 15 min, then mark as missing).

## Feature Ideas to Explore

### Tier 1 — Low risk, high expected gain
1. **Richer lag window**: extend lags to [1, 2, 3, 6, 12, 18, 24, 36].
2. **Time-of-day encoding**: sin/cos of hour.
3. **Rate-of-change features**: first derivative over 15 and 30 minutes.

### Tier 2 — Model alternatives
4. **LightGBM** with identical feature set.
5. **Separate models per horizon**: 30, 60, 120 min independently.
6. **XGBoost with early stopping**.

### Tier 3 — Garmin integration
7. **Stress × glucose interaction**.
8. **Sleep quality lag**.

### Tier 4 — Architectural changes
9. **Small LSTM (16 hidden units)**.
10. **Ensemble of Ridge + LightGBM**.

## Evaluation Protocol

Walk-forward validation:
- Training window: 14 days
- Validation window: 7 days
- Step: 7 days
- Final MAE = mean across all folds

## Promotion Criteria

Promote if and only if MAE_30 < 17.74 AND MAE_60 < 25.18 AND MAE_120 < 28.60.

## Clinical Safety Note

This is a personal analytics tool, not a medical device. Prefer models with lower
error in the hypoglycaemic range (glucose < 70 mg/dL).
