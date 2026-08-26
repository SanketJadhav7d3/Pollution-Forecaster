# delhi-mumbai-aqi-mlops

Forecasts tomorrow's AQI category for Delhi and Mumbai from OpenAQ data.
An end-to-end MLOps project: fetching, feature engineering, MLflow tracking,
backtesting, and (later) Docker/GCP/CI-CD.

See [specs.md](specs.md) for the full brief.

## Setup

```bash
uv sync
cp .env.example .env      # add your key from explore.openaq.org
```

## Usage

```bash
uv run python -m src.fetch_openaq discover        # find live sensors
uv run python -m src.fetch_openaq backfill --years 2
uv run python -m src.fetch_openaq daily           # incremental
uv run python -m src.build_dataset                # -> data/processed/
uv run python -m src.train                        # baseline
uv run python -m src.backtest                     # walk-forward folds
uv run python -m src.experiments                  # model x feature grid
uv run python -m src.register_model               # register best model
uv run pytest
```

Cities and pollutants are configured in [config/cities.yaml](config/cities.yaml).

## MLflow

Runs are stored in SQLite, so the URI flag is required:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Status

Phases 1-4 done (fetching, prep, training, backtesting).
Phases 5-9 pending (Docker, GCP, CI/CD, GKE, drift).
