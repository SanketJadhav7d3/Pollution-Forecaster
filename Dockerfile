# Multi-stage: the build stage resolves dependencies, the runtime stage carries
# only what serving needs. The training stack (scikit-learn, matplotlib, the
# backtest tooling) lives in the dev group and never reaches the final image -
# which is why the dependency groups were split from the start.

FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Copy only the manifests first: this layer is cached unless dependencies
# actually change, so ordinary code edits skip the whole install.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY config/ ./config/
RUN uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim AS runtime

# Run unprivileged: a container that never needs to write to its own filesystem
# should not be able to.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/config /app/config

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=file:///app/mlruns

USER appuser

# Cloud Run injects PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys,os; \
        sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health').status==200 else 1)"

# Shell form so $PORT expands. One worker: the model loads per process, and
# Cloud Run scales by adding containers rather than threads.
CMD exec uvicorn src.serve:app --host 0.0.0.0 --port ${PORT} --workers 1
