# delhi-mumbai-aqi-mlops

Forecasts **tomorrow's AQI category** for Delhi and Mumbai from OpenAQ data.
An end-to-end MLOps project: data fetching, feature engineering, MLflow
tracking, walk-forward backtesting, and (later) Docker/GCP/CI-CD.

See [specs.md](specs.md) for the full brief.

## Setup

```bash
uv sync                      # Python 3.12, creates .venv
cp .env.example .env         # then paste your key from explore.openaq.org
```

## Usage

```bash
uv run python -m src.fetch_openaq discover      # find live sensors per city
uv run python -m src.fetch_openaq backfill --years 2
uv run python -m src.fetch_openaq daily         # incremental, for the scheduler
uv run python -m src.build_dataset              # -> data/processed/city_daily.{parquet,csv}
uv run python -m src.train                      # baseline + per-city + persistence
uv run python -m src.backtest                   # walk-forward folds
uv run python -m src.experiments                # model x feature grid, all logged
uv run python -m src.register_model             # train on all history, register
uv run mlflow ui                                # inspect runs
uv run pytest
```

Cities and pollutants live in [config/cities.yaml](config/cities.yaml); adding
a city or a parameter needs no code change.

## Current state

| Phase | Status |
|---|---|
| 1. Data fetching | done |
| 2. Data prep | done |
| 3. Baseline training | runs; does **not** beat persistence on accuracy |
| 4. Backtesting | done; model comparison grid tracked in MLflow |
| 5-9. Docker, GCP, CI/CD, GKE, drift | not started |

## Metrics: why accuracy is not the headline

About **69% of days carry the same AQI category as the day before**, so "tomorrow
looks like today" scores ~0.73 accuracy for free - while being structurally
incapable of predicting a change (0% recall on change days, every fold). A metric
that ranks a blind forecaster first is measuring the wrong thing, so runs report:

| Metric | Question it answers |
|---|---|
| `skill_score` | What fraction of persistence's errors did the model fix? 0 = no better than doing nothing. |
| `recall_deterioration` | Of days the air genuinely got worse, how many did we call? Persistence: 0. |
| `bad_air_precision` | How many "Poor or worse" warnings were real? A warning system nobody trusts gets ignored. |
| `accuracy` | Sanity check only. |

## Findings so far

- **Data spans 2024-08-25 to 2026-08-24** (730 days/city). The spec's 3-year
  target is not achievable: only two sensors (the diplomatic monitors) reach
  back to 2016, and ~60% of live sensors first reported in 2025.
- **Sensors churn, cities don't.** Stations are retired and re-registered under
  new ids, so no single sensor spans the window. Pooling to a city-day mean
  gives 100% coverage for Delhi and 98.5% for Mumbai.
- **No model beats persistence on accuracy.** Every configuration scores
  negative skill (best: random forest at -0.18). That result held across four
  model families, five feature sets, and both classification and
  regression-then-threshold framings.
- **But every model beats it at the thing a forecast is for.** The registered
  xgboost model catches ~44% of deteriorations at ~0.76 precision; persistence
  catches 0% by construction. The model trades ~7 points of accuracy on easy
  days for the ability to warn at all.
- **Weather features did not help** (115 MB fetched; recall fell from 0.80 to
  0.76 for the random forest, and accuracy fell for every model). Today's wind
  clears today's air; predicting tomorrow needs a weather *forecast*, and
  today's pm25 already encodes today's ventilation more directly. The pipeline
  is built and tested if an NWP source is added later.
- **Gradient boosting beats random forests at transitions, and loses on
  accuracy.** Forests average toward the common case ("no change"), which is
  what accuracy rewards; boosting chases residuals, which is where transitions
  live.
- **The CPCB network has multi-day outages** (9 consecutive days in Jan 2026)
  where only ~6 sensors report. Those days' labels are noisier: the surviving
  subset tracks the full pool in level (r=0.975) but the derived category
  disagrees on ~24% of days.
