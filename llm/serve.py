"""Complaint-box API: the LoRA-tuned LLM (llm/train_lora.py) behind HTTP, on CPU.

    POST /score       one airline review -> score out of 10, recommended yes/no, the main complaint
    GET  /health      200 once a model is loaded (used by Docker and Nginx), 503 before
    GET  /scores/recent?hours=N   aggregates of the last N hours of the prediction log (never text): the drift check reads it
    GET  /model       which adapter version is serving, on which base model and prompt version, since when

The base model (Qwen2.5-1.5B-Instruct, ~3 GB) is downloaded once from Hugging Face into HF_HOME and stays frozen;
the adapter comes from the MLflow registry: `skysat-reviews@champion`. A background thread checks the alias every
MODEL_RELOAD_SECONDS; a new version is loaded as a second adapter on the same base, switched to, and the old
adapter is deleted: a new model is a small download, never a second copy of the base model.
LLM_ADAPTER_DIR=<folder> serves a local adapter instead and does not use MLflow at all.

One review at a time (a lock): on CPU one generation already uses every core, and there is one model in memory.

If DATABASE_URL is set, each answer is logged to PostgreSQL (table llm_scores, created here): model version, latency,
the three answers and, if the caller sent them, the true score and recommendation. Never the review, the title or the
airline: the review dataset is not ours to store.

Run: uvicorn llm.serve:app --host 0.0.0.0 --port 8000   (Docker: llm/Dockerfile;  locally: make llm-serve)
"""

import json
import logging
import os
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from llm.gate import MAX_ERROR
from llm.reviews import format_prompt, parse_output
from llm.train_lora import (ALIAS, BASE_MODEL, INFO_FILE, MAX_NEW_TOKENS, MODEL_NAME, PROMPT_VERSION, champion_adapter,
                            chat)

# torch, transformers and peft are imported inside the functions that need them: the tests (and CI) have none

RELOAD_SECONDS = float(os.environ.get("MODEL_RELOAD_SECONDS", "300"))
ADAPTER_DIR = os.environ.get("LLM_ADAPTER_DIR") or None  # a local adapter folder: no MLflow at all
DTYPE = os.environ.get("LLM_DTYPE", "bfloat16")
QUEUE_SECONDS = float(os.environ.get("LLM_QUEUE_SECONDS", "100"))  # waiting longer than this for the model: 503
DATABASE_URL = os.environ.get("DATABASE_URL")
DTYPES = ("bfloat16", "float32")
if DTYPE not in DTYPES:
    raise ValueError(f"LLM_DTYPE must be one of {DTYPES}, not {DTYPE!r}")

log = logging.getLogger("uvicorn.error")



class ScoreRequest(BaseModel):
    """One review. true_score / true_recommended are optional: the caller sends them when it knows them (the replay
    does), so the dashboard can show how far off the model is on live traffic."""
    model_config = ConfigDict(extra="forbid")
    airline: str = Field("", max_length=200)
    cabin: str = Field("", max_length=100)
    travel_type: str = Field("", max_length=100)
    title: str = Field("", max_length=500)
    review: str = Field(min_length=1, max_length=50_000)
    true_score: int | None = Field(None, ge=1, le=10)
    true_recommended: Literal["yes", "no"] | None = None


def prompt_row(req: ScoreRequest) -> dict:
    """The request as the row format_prompt() reads (the same prompt the model was tuned on)."""
    return {"AirlineName": req.airline, "CabinType": req.cabin, "TravelType": req.travel_type, "Title": req.title,
            "Review": req.review}


class IncompatibleAdapter(ValueError):
    """The adapter was tuned for another base model or prompt: its answers would not be what the gate scored."""


def read_info(folder) -> dict:
    """The adapter's skysat_llm.json (written by llm/train_lora.py); raises IncompatibleAdapter if it does not fit."""
    path = Path(folder) / INFO_FILE
    if not path.exists():
        raise IncompatibleAdapter(f"{path} not found: not an adapter from llm/train_lora.py")
    info = json.loads(path.read_text())
    if info.get("base_model") != BASE_MODEL or info.get("prompt_version") != PROMPT_VERSION:
        raise IncompatibleAdapter(f"adapter in {folder} is for base_model={info.get('base_model')!r}, prompt_version="
                                  f"{info.get('prompt_version')!r}; this code serves {BASE_MODEL!r}, {PROMPT_VERSION}")
    return info


# --- The three heavy steps (torch / transformers / peft). The tests replace them with fakes. ---

def load_base(dtype: str):
    """(tokenizer, base model) on CPU; downloaded into HF_HOME the first time."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=getattr(torch, dtype)).to("cpu").eval()
    return tok, model


def attach_adapter(model, folder: str, name: str, old: str | None):
    """First adapter: wrap the base model. Later ones: load next to the old one on the same base, switch, delete
    the old one. If the new one fails, the old one stays active."""
    if old is None:
        from peft import PeftModel
        return PeftModel.from_pretrained(model, folder, adapter_name=name).eval()
    try:
        model.load_adapter(folder, adapter_name=name)
        model.set_adapter(name)
    except Exception:
        if name in model.peft_config:
            model.set_adapter(old)
            model.delete_adapter(name)
        raise
    model.delete_adapter(old)
    return model.eval()


def generate(tok, model, prompt: str) -> tuple[str, int]:
    """Greedy, as the gate scored it (llm/train_lora.py generate()). Returns (the three lines, generated tokens)."""
    import torch
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
    new = out[0, enc.input_ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), int(new.shape[0])


def mlflow_client():
    import mlflow
    return mlflow.MlflowClient()


class ScoreModel:
    """One base model in memory with one active adapter. `current` is replaced as a whole, under the lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.tok = self.base = self.model = None
        self.current = None  # {"version", "adapter", "info", "source", "loaded_at"}
        self.rejected = None  # a champion version we refused: not downloaded again every check

    def install(self, folder: str, version: str, source: str) -> None:
        info = read_info(folder)  # before any loading: a refused adapter changes nothing
        if self.base is None:
            log.info("Loading base model %s (%s, CPU); the first start downloads ~3 GB", BASE_MODEL, DTYPE)
            started = time.perf_counter()
            self.tok, self.base = load_base(DTYPE)
            log.info("Base model loaded in %.0f s", time.perf_counter() - started)
        name = "v_" + re.sub(r"\W", "_", version)
        with self.lock:
            old = self.current["adapter"] if self.current else None
            self.model = attach_adapter(self.model if old else self.base, folder, name, old)
            before = self.current["version"] if self.current else "none"
            self.current = {"version": version, "adapter": name, "info": info, "source": source,
                            "loaded_at": datetime.now(timezone.utc).isoformat()}
        log.info("Serving %s version %s (was %s)", MODEL_NAME, version, before)

    def refresh(self) -> None:
        if ADAPTER_DIR:
            try:
                self.install(ADAPTER_DIR, "local", f"local folder {ADAPTER_DIR}")
            except IncompatibleAdapter as err:
                log.error("Refused: %s", err)
            return
        client = mlflow_client()
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
        if version == self.rejected or (self.current and self.current["version"] == version):
            return
        champion = champion_adapter(client, BASE_MODEL)  # None: tuned for another base model or prompt version
        if champion is None:
            self.rejected = version
            log.error("Refused %s version %s: not for %s / prompt_version %s; keeping %s", MODEL_NAME, version,
                      BASE_MODEL, PROMPT_VERSION, self.current["version"] if self.current else "no model")
            return
        version, folder = champion
        try:
            self.install(folder, version, f"models:/{MODEL_NAME}@{ALIAS}")
        except IncompatibleAdapter as err:
            self.rejected = version
            log.error("Refused: %s", err)
        finally:
            shutil.rmtree(folder, ignore_errors=True)  # the adapter is in memory now

    def keep_fresh(self) -> None:
        while True:
            try:
                self.refresh()
            except Exception as err:  # MLflow down, no champion yet, download failed: keep the old one, retry
                log.warning("Model refresh failed: %s", err)
            if ADAPTER_DIR and self.current:
                return  # a local adapter does not change
            time.sleep(RELOAD_SECONDS)


serving = ScoreModel()
db = ConnectionPool(DATABASE_URL, min_size=1, max_size=2, open=False) if DATABASE_URL else None
_table_ready = False

# Only numbers and fixed labels (complaint is one of the seven sub-rating names or "none"): never the review text
LOG_COLUMNS = ["model_version", "latency_ms", "score", "recommended", "complaint", "true_score", "true_recommended"]
CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS llm_scores (id bigserial PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),"
    " model_version text, latency_ms integer, score integer, recommended text, complaint text,"
    " true_score integer, true_recommended text)"
)
INSERT = (f"INSERT INTO llm_scores ({', '.join(LOG_COLUMNS)})"
          f" VALUES ({', '.join(f'%({c})s' for c in LOG_COLUMNS)})")


def log_score(row: dict) -> None:
    """Monitoring must not break serving: if the database is down, log a warning and go on."""
    global _table_ready
    try:
        with db.connection(timeout=5) as conn:
            if not _table_ready:
                conn.execute(CREATE_TABLE)
                _table_ready = True
            conn.execute(INSERT, {c: row[c] for c in LOG_COLUMNS})
    except Exception as err:
        log.warning("Could not log score: %s", err)


# Aggregates only. A malformed score counts as an error of gate.MAX_ERROR (as in the gate), a malformed
# recommendation as wrong; the labelled counts say how many rows each measure is computed on.
SUMMARY = """
SELECT count(*) AS n,
       avg(score) AS mean_score,
       count(true_score) AS labelled_score,
       avg(CASE WHEN true_score IS NULL THEN NULL WHEN score IS NULL THEN %(max_error)s
                ELSE abs(score - true_score) END) AS mae,
       count(true_recommended) AS labelled_recommended,
       avg(CASE WHEN true_recommended IS NULL THEN NULL
                WHEN recommended = true_recommended THEN 1.0 ELSE 0.0 END) AS recommended_accuracy
FROM llm_scores WHERE created_at >= now() - %(hours)s * interval '1 hour'
"""
COMPLAINTS = """
SELECT coalesce(complaint, 'malformed') AS complaint, count(*) AS n FROM llm_scores
WHERE created_at >= now() - %(hours)s * interval '1 hour' GROUP BY 1 ORDER BY 2 DESC, 1
"""
VERSIONS = """
SELECT model_version, count(*) AS n FROM llm_scores
WHERE created_at >= now() - %(hours)s * interval '1 hour' GROUP BY 1 ORDER BY 1
"""


def _num(value, digits=4):
    return None if value is None else round(float(value), digits)


def recent_scores(conn, hours: float) -> dict:
    """Aggregates over the last `hours` hours of llm_scores: counts, mean score, MAE and recommended accuracy where
    the true labels were sent, share per complaint. Numbers only, never text."""
    params = {"hours": hours, "max_error": MAX_ERROR}
    row = conn.execute(SUMMARY, params).fetchone()
    n, mean_score, labelled_score, mae, labelled_rec, rec_acc = row
    complaints = {name: _num(count / n) for name, count in conn.execute(COMPLAINTS, params).fetchall()} if n else {}
    versions = {str(v): int(count) for v, count in conn.execute(VERSIONS, params).fetchall()}
    return {"hours": hours, "count": int(n), "mean_score": _num(mean_score, 2),
            "labelled_score": int(labelled_score), "mae": _num(mae),
            "labelled_recommended": int(labelled_rec), "recommended_accuracy": _num(rec_acc),
            "complaint_share": complaints, "model_versions": versions}


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=serving.keep_fresh, daemon=True).start()
    if db:
        db.open()
    yield
    if db:
        db.close()


app = FastAPI(
    title="SkySat complaint-box API",
    description=f"Reads an airline review: score, recommended, main complaint. {BASE_MODEL} + LoRA adapter `{MODEL_NAME}@{ALIAS}`.",
    version="1.0",
    lifespan=lifespan,
)


def require_model() -> dict:
    if serving.current is None:
        raise HTTPException(status_code=503, detail="No model loaded yet (downloading the base model, or no champion)")
    return serving.current


@app.post("/score")
def score(req: ScoreRequest):
    require_model()
    prompt = chat(serving.tok, format_prompt(prompt_row(req)))
    if not serving.lock.acquire(timeout=QUEUE_SECONDS):
        raise HTTPException(status_code=503, detail="Busy: too many reviews waiting, try again")
    try:
        started = time.perf_counter()
        current = serving.current
        text, n_tokens = generate(serving.tok, serving.model, prompt)
        latency_ms = int((time.perf_counter() - started) * 1000)
    finally:
        serving.lock.release()
    log.info("Score: %d tokens in %d ms (%.1f tokens/s)", n_tokens, latency_ms, n_tokens / max(latency_ms, 1) * 1000)
    parsed = parse_output(text)
    if db:
        log_score({"model_version": current["version"], "latency_ms": latency_ms, **parsed,
                   "true_score": req.true_score, "true_recommended": req.true_recommended})
    return {**parsed, "model_version": current["version"], "latency_ms": latency_ms}


@app.get("/scores/recent")
def scores_recent(hours: float = Query(24, gt=0, le=24 * 366)):
    """What the model answered in the last `hours` hours, as numbers (docs/llm.md, Monitoring and drift check)."""
    global _table_ready
    if db is None:
        raise HTTPException(status_code=503, detail="No prediction log: DATABASE_URL is not set")
    try:
        with db.connection(timeout=5) as conn:
            if not _table_ready:
                conn.execute(CREATE_TABLE)
                _table_ready = True
            return recent_scores(conn, hours)
    except Exception as err:
        log.warning("Could not read scores: %s", err)
        raise HTTPException(status_code=503, detail="The prediction log is not reachable") from err


@app.get("/health")
def health():
    return {"status": "ok", "model_version": require_model()["version"]}


@app.get("/model")
def model_info():
    current = require_model()
    return {
        "name": MODEL_NAME,
        "alias": ALIAS,
        "version": current["version"],
        "source": current["source"],
        "base_model": BASE_MODEL,
        "prompt_version": current["info"]["prompt_version"],
        "trained_on": current["info"].get("split"),
        "dtype": DTYPE,
        "loaded_at": current["loaded_at"],
    }
