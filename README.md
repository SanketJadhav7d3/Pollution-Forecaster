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
| 3. Baseline training | runs; does **not** beat the persistence baseline |
| 4. Backtesting | done |
| 5-9. Docker, GCP, CI/CD, GKE, drift | not started |

## Findings so far

- **Data spans 2024-08-25 to 2026-08-24** (730 days/city). The spec's 3-year
  target is not achievable: only two sensors (the diplomatic monitors) reach
  back to 2016, and ~60% of live sensors first reported in 2025.
- **Sensors churn, cities don't.** Stations are retired and re-registered under
  new ids, so no single sensor spans the window. Pooling to a city-day mean
  gives 100% coverage for Delhi and 98.5% for Mumbai.
- **The model loses to persistence on 4/4 backtest folds** (69.4% vs 72.9% mean
  accuracy). Daily AQI is dominated by persistence plus ventilation; lag and
  calendar features alone do not carry that signal. Weather features are the
  next hypothesis.
- **The CPCB network has multi-day outages** (9 consecutive days in Jan 2026)
  where only ~6 sensors report. Those days' labels are noisier: the surviving
  subset tracks the full pool in level (r=0.975) but the derived category
  disagrees on ~24% of days.
