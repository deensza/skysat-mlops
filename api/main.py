"""Step 4 – Prediction API.

    POST /predict   one passenger (id + 22 features) -> satisfied yes/no and the probability
    POST /feedback  the real answer for an earlier prediction (step 6)
    GET  /health    200 once a model is loaded (used by Docker and Nginx)
    GET  /model     which registered model version is serving, and its validation metrics

The model comes from the MLflow registry: `skysat-satisfaction@champion`. A background thread
checks the alias every RELOAD_SECONDS; when it points to a new version, the new model is
downloaded and swapped in while requests keep being served (no restart, no downtime).

Step 6: if DATABASE_URL is set, every prediction is written to PostgreSQL (table `predictions`,
monitoring/init.sql) and /feedback fills in the real answer. Grafana reads that table.

Run (inside Docker): uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import socket
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import mlflow
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field, create_model

from skysat import schema
from training.train import load_bundle, predict_proba

MODEL_NAME = "skysat-satisfaction"
ALIAS = "champion"
RELOAD_SECONDS = float(os.environ.get("MODEL_RELOAD_SECONDS", "10"))
REPLICA = socket.gethostname()  # container id: shows which copy answered (step 5)
DATABASE_URL = os.environ.get("DATABASE_URL")

torch.set_num_threads(1)  # one CPU core per API copy; scaling = more copies
log = logging.getLogger("uvicorn.error")

# Request body built from skysat/schema.py, so the API accepts exactly what the model was trained on
fields = {schema.ID_COLUMN: (int, Field(ge=0))}
for col, values in schema.CATEGORICAL_VALUES.items():
    fields[col] = (Literal[tuple(values)], ...)
for col in schema.NUMERIC_FEATURES:
    fields[col] = (int, Field(ge=0))
for col in schema.RATING_FEATURES:
    fields[col] = (int, Field(ge=schema.RATING_MIN, le=schema.RATING_MAX))
Passenger = create_model("Passenger", __config__=ConfigDict(extra="forbid"), **fields)


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: uuid.UUID
    satisfied: bool


class ServingModel:
    """The model currently in use. Replaced as a whole, so a request never sees half an update."""

    def __init__(self):
        self.current = None  # (version, run_id, bundle, loaded_at)

    def refresh(self) -> None:
        client = MlflowClient()
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        if self.current and self.current[0] == version.version:
            return
        with tempfile.TemporaryDirectory() as tmp:
            folder = mlflow.artifacts.download_artifacts(artifact_uri=version.source, dst_path=tmp)
            bundle = load_bundle(Path(folder) / "model.pt")
        before = self.current[0] if self.current else "none"
        self.current = (version.version, version.run_id, bundle, datetime.now(timezone.utc).isoformat())
        log.info("Serving %s version %s (was %s)", MODEL_NAME, version.version, before)

    def keep_fresh(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception as err:  # MLflow down or no champion yet: keep the old model, retry
                log.warning("Model refresh failed: %s", err)
            time.sleep(RELOAD_SECONDS)


serving = ServingModel()
db = ConnectionPool(DATABASE_URL, min_size=1, max_size=8, open=False) if DATABASE_URL else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=serving.keep_fresh, daemon=True).start()
    if db:
        db.open()
    yield
    if db:
        db.close()


def log_prediction(row: dict) -> None:
    """Monitoring must not break serving: if the database is down, log a warning and go on."""
    try:
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO predictions (prediction_id, passenger_id, features, probability, predicted,"
                " model_version, replica, latency_ms) VALUES (%(prediction_id)s, %(passenger_id)s,"
                " %(features)s, %(probability)s, %(predicted)s, %(model_version)s, %(replica)s, %(latency_ms)s)",
                row,
            )
    except Exception as err:
        log.warning("Could not log prediction: %s", err)


app = FastAPI(
    title="SkySat prediction API",
    description="Predicts whether an airline passenger is satisfied. Model: MLflow registry, alias `champion`.",
    version="1.0",
    lifespan=lifespan,
)


def require_model():
    if serving.current is None:
        raise HTTPException(status_code=503, detail="No model loaded yet (is MLflow up and is there a champion?)")
    return serving.current


@app.post("/predict")
def predict(passenger: Passenger):
    started = time.perf_counter()
    version, _, bundle, _ = require_model()
    body = passenger.model_dump()
    probability = float(predict_proba(bundle, pd.DataFrame([body]))[0])
    prediction_id = uuid.uuid4()
    if db:
        log_prediction({
            "prediction_id": prediction_id,
            "passenger_id": body[schema.ID_COLUMN],
            "features": Jsonb({col: body[col] for col in schema.FEATURES}),
            "probability": probability,
            "predicted": int(probability >= 0.5),
            "model_version": version,
            "replica": REPLICA,
            "latency_ms": (time.perf_counter() - started) * 1000,
        })
    return {
        "prediction_id": str(prediction_id),
        schema.ID_COLUMN: body[schema.ID_COLUMN],
        "satisfied": probability >= 0.5,
        "probability": round(probability, 4),
        "model_version": version,
        "replica": REPLICA,
    }


@app.post("/feedback")
def feedback(item: Feedback):
    """The passenger's real answer (from the survey), linked to an earlier prediction."""
    if not db:
        raise HTTPException(status_code=503, detail="Prediction logging is off (no DATABASE_URL)")
    with db.connection() as conn:
        updated = conn.execute(
            "UPDATE predictions SET actual = %s, labelled_at = now() WHERE prediction_id = %s",
            (int(item.satisfied), item.prediction_id),
        ).rowcount
    if updated == 0:
        raise HTTPException(status_code=404, detail="Unknown prediction_id")
    return {"prediction_id": str(item.prediction_id), "recorded": True}


@app.get("/health")
def health():
    version, *_ = require_model()
    return {"status": "ok", "model_version": version, "replica": REPLICA}


@app.get("/model")
def model_info():
    version, run_id, bundle, loaded_at = require_model()
    return {
        "name": MODEL_NAME,
        "alias": ALIAS,
        "version": version,
        "run_id": run_id,
        "loaded_at": loaded_at,
        "validation_metrics": bundle["meta"]["metrics"],
        "replica": REPLICA,
    }
