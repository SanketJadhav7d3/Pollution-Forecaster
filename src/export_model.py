"""Export a registered model to a plain directory for baking into an image.

    uv run python -m src.export_model

A container should not reach into a laptop's MLflow store at runtime, and
Cloud Run has no such store at all. Exporting the artifact here means the image
carries one specific model version - the image tag and the model version stay
pinned together, and a rollback is just deploying the previous image.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.register_model import MODEL_NAME

log = logging.getLogger("export_model")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "model_artifact"


def export(model_name: str = MODEL_NAME, version: str | None = None,
           out_dir: Path = DEFAULT_OUT) -> Path:
    client = MlflowClient()
    if version is None:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"no registered versions of '{model_name}'")
        version = str(max(int(v.version) for v in versions))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    uri = f"models:/{model_name}/{version}"
    log.info("downloading %s", uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    shutil.copytree(local, out_dir, dirs_exist_ok=True)

    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)
    (out_dir / "export_info.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "model_version": version,
                "run_id": mv.run_id,
                "data_sha256": run.data.params.get("data_sha256"),
                "trained_rows": run.data.params.get("trained_rows"),
                "date_max": run.data.params.get("date_max"),
            },
            indent=2,
        )
    )
    log.info("exported %s v%s -> %s", model_name, version, out_dir)
    return out_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export a registered model to a directory")
    p.add_argument("--name", default=MODEL_NAME)
    p.add_argument("--version", default=None)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    export(args.name, args.version, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
