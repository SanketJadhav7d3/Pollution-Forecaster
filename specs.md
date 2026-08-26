# delhi-mumbai-aqi-mlops

**A pollution forecaster** (not a same-day predictor) — an end-to-end MLOps
project that predicts **tomorrow's AQI category** for Delhi and Mumbai, built
to learn Docker, Kubernetes, MLflow, CI/CD, and drift monitoring on live,
real-world data.

This file is a working brief for a coding agent to pick up and implement
step by step. Each phase should be built, tested, and confirmed working
before moving to the next — don't jump ahead.

---

## 1. What this project is

- **Task type**: Forecasting (time-aware), not static prediction. Target is
  **tomorrow's AQI category** (e.g., Good / Moderate / Unhealthy / Hazardous),
  predicted from today's and recent days' readings.
- **Task shape**: Multi-class classification.
- **Scope**: Two cities to start — **Delhi** and **Mumbai** — with the
  pipeline designed to extend to more Indian cities later without rework.
- **Data source**: [OpenAQ v3 API](https://docs.openaq.org) — free tier,
  requires an API key from https://explore.openaq.org.
- **Algorithm**: Start with `RandomForestClassifier` (reuse the same
  pipeline shape as the earlier Adult Census project for consistency).
  Once the baseline works end-to-end, try `XGBoost`/`LightGBM` — gradient
  boosted trees generally edge out random forests on tabular data with
  engineered lag features. No deep learning — it adds complexity without
  benefit for this data size/shape.
- **Experiment tracking / registry**: MLflow (local first, GCP-hosted later).
- **Cloud**: GCP, using the $300 trial credit. Serverless-first (Cloud
  Functions + Cloud Scheduler + GCS) to avoid always-on costs.

---

## 2. Data fetching

### Cities (radius-based, not bounding box)
Use a small config of `{city_name, latitude, longitude, radius_meters}`
entries rather than hand-picked bounding boxes — easier to extend to more
cities later. Start with:
- Delhi
- Mumbai

Each city's raw data should land in its own subfolder
(`data/raw/openaq/delhi/`, `data/raw/openaq/mumbai/`) so a `city` column
naturally attaches to each row when building the training set — this
becomes the location feature for a shared multi-city model (see §5).

### Two fetch modes, same underlying logic
1. **Backfill (run once, manually)** — pull **3 years** of historical
   daily-aggregated measurements per sensor, per city. This is the
   bootstrap dataset — needed because 3 years of history captures full
   seasonal cycles (Delhi winter smog, monsoon cleanup, stubble-burning
   season, etc.) which shorter windows would miss entirely.
2. **Daily (scheduled, ongoing)** — pull just the last ~2 days per sensor,
   merge into existing per-sensor files rather than overwriting. This is
   the logic that later runs inside a Cloud Function on a schedule.

### Storage policy — keep everything, don't delete
- Raw fetched data is immutable and append-only — never overwritten,
  never deleted. Storage cost for this dataset size (a few MB/year) is a
  non-issue.
- "Old data doesn't matter for training" is handled at **query/training
  time** via a training window (§4), not by deleting anything at storage
  time. Deleting removes your ability to debug, audit, and re-run drift
  comparisons later.
- If cost ever becomes a concern (it won't at this scale), use a GCS
  lifecycle policy to transition old objects to a cheaper storage class —
  reversible, unlike deletion.

### Trigger mechanism (for the live daily fetch)
**Cloud Scheduler → Cloud Function**, the standard GCP serverless cron
pattern:
1. Cloud Scheduler job runs on a daily cron schedule (e.g., `0 3 * * *`).
2. It calls an HTTP-triggered Cloud Function (authenticated via a service
   account / OIDC token, not publicly callable).
3. The function runs the "daily" fetch logic and writes results to GCS.
4. No servers run 24/7 — you only pay for the few seconds of execution
   per day.

---

## 3. Feature engineering

Because this is a forecaster (predicting tomorrow), features must be built
from **lagged** history, not same-day values alone:
- Yesterday's reading, 3-day rolling average, 7-day rolling average
- Rate of change (is the trend rising or falling, how fast)
- Calendar features: month, day-of-week, is-stubble-burning-season flag
- Location: latitude/longitude and/or `city` as a categorical feature —
  lets one shared model generalize across cities instead of needing a
  separate model per city (see §5)

**No deletion of past data** — the training window filter (below) is what
determines what's "in scope" for a given training run, applied at
query time against the full immutable history.

---

## 4. Training window vs. forecasting window vs. retrain frequency

Three separate knobs — don't conflate them:

| Knob | Value | Why |
|---|---|---|
| **Training window** | Rolling **12 months** (not a short 90-day window) | Air quality is strongly seasonal; a short window loses entire seasons on every retrain. Once 2+ years of history exist, consider extending to 24 months or adding year-over-year features instead of just growing the window indefinitely. |
| **Forecasting window** | **Next-day** (t+1) | Matches the daily fetch cadence and is a genuine forecasting task, more useful than same-day classification. |
| **Retrain frequency** | **Weekly**, scheduled (Cloud Scheduler → retrain job) | Daily retraining adds noise for little signal at this data cadence; monthly is too slow to react to real shifts. Start with a scheduled weekly retrain; once drift monitoring (§7) exists, upgrade to a hybrid: weekly floor + drift-triggered early retrain. |

**Chronological splits only** — never randomly shuffle time-ordered data
for train/test. Random shuffling leaks future information into training.

---

## 5. One model, not one per city

Do **not** train separate models per city. Instead:
- Pool all cities' data into one training set
- Add `city` (and/or lat/lon) as an input feature so the model learns
  location-dependent patterns itself (tree-based models are well-suited to
  this — they learn interactions like "city=Delhi AND month=November →
  high PM2.5" without manual encoding)
- This is more data-efficient (cities with sparse history still benefit
  from patterns learned in data-rich cities), simpler to maintain (one
  pipeline, one registry, one monitor instead of N), and generalizes to
  more cities later without restructuring
- If, after measuring, one city/region shows systematically worse error
  than others, that's a finding to investigate — not an upfront assumption
  to design around

---

## 6. Backtesting (validate before waiting on live data)

With 3 years of backfilled history, validate the whole approach
**retrospectively** rather than waiting weeks/months for live data to prove
the pipeline works:

- Use **expanding-window walk-forward validation**
  (`sklearn.model_selection.TimeSeriesSplit`, or hand-rolled by date):
  train on everything up to a point in time, test on the next unseen
  chunk, then expand the training window forward and repeat.
- Fold granularity: **monthly or quarterly** test windows (daily folds
  would be too numerous and too small to be meaningful for a
  classification metric).
- Minimum training size: don't evaluate any fold until at least ~12
  months of training data exists behind it, so every fold's model has
  seen a full season.
- Report **accuracy/F1 per fold over time** (a chart), not a single
  aggregate number — this shows stability/degradation over time and
  surfaces harder periods (e.g., stubble-burning season) honestly.
- Log each fold as its own MLflow run, so the full backtest history is
  visible in the MLflow UI as portfolio evidence.
- **Distinction to keep straight**: backtesting validates the approach
  offline/retrospectively ("did this work when it couldn't see the
  future"). Live drift monitoring (§7) validates online/prospectively
  ("is today's incoming data starting to differ from what the deployed
  model was trained on"). Both matter; they answer different questions.

---

## 7. Drift monitoring (after the baseline pipeline works)

- Use **Evidently** to compare feature distributions between the training
  window and live incoming data.
- Log drift reports as MLflow artifacts.
- Eventually wire drift crossing a threshold into an **early retrain
  trigger**, on top of the weekly scheduled floor (see §4).

---

## 8. Data storage layers (once this moves to GCP)

```
Raw landing zone     → GCS, immutable, exactly as fetched, partitioned by
                        city and fetch date
Processed/training   → cleaned, schema-validated, still dated
                        snapshot
Training snapshot    → the exact slice used for a specific training run,
                        referenced by path/hash, logged as an MLflow param
                        (not re-derived from memory)
```

Log a reference to the exact data used in every MLflow run
(`data_snapshot` path, row count, hash) — not just the model — so any run
in the registry is fully reproducible months later.

---

## 9. Build order (do not skip ahead)

1. **Data fetching** — city-config-based OpenAQ fetcher (Delhi + Mumbai),
   backfill mode (3 years) and daily mode (2-day incremental merge).
   *[in progress — see `src/fetch_openaq.py` for the Delhi-only version to
   generalize to a city list]*
2. **Local data prep** — load raw JSON per city/sensor, clean, engineer
   lag/calendar/location features, build the AQI-category target.
3. **Baseline training locally** — `RandomForestClassifier`, MLflow
   tracking, chronological split, log metrics/artifacts, register model.
4. **Backtesting** — expanding-window walk-forward validation across the
   3-year history, per-fold MLflow runs, accuracy-over-time chart.
5. **Dockerize** — inference service (FastAPI) wrapping the MLflow model.
6. **Deploy fetch pipeline to GCP** — Cloud Function + Cloud Scheduler,
   writing to GCS, using the $300 trial credit (serverless, near-zero
   idle cost).
7. **CI/CD** — GitHub Actions: on push/schedule → retrain → validate →
   if improved, promote in MLflow registry → build & push Docker image →
   deploy.
8. **Deploy inference to GKE (or Cloud Run first, cheaper/simpler)**.
9. **Drift monitoring** — Evidently reports against live traffic, logged
   to MLflow, wired into an early-retrain trigger.

---