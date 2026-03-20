"""
glucoassist-autoresearcher — standalone glucose research engine.

Exposes an HTTP API consumed by GlucoAssist when AUTORESEARCHER_URL is configured.
Reads glucose data from a shared SQLite database and runs the ML walk-forward CV
loop, proposing experiments via a locally-hosted or commercial LLM.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services import autoresearcher as ar_service

log = logging.getLogger(__name__)

VERSION = "0.1.0"

_CREATE_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS autoresearcher_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT    NOT NULL,
    experiment_id  INTEGER,
    timestamp      TEXT,
    description    TEXT,
    mae_30         REAL,
    mae_60         REAL,
    mae_90         REAL,
    mae_120        REAL,
    promoted       INTEGER DEFAULT 0,
    elapsed_s      REAL,
    feature_config TEXT,
    model_config   TEXT,
    notes          TEXT
)
"""

_LOG_COLS = (
    "id", "run_id", "experiment_id", "timestamp", "description",
    "mae_30", "mae_60", "mae_90", "mae_120",
    "promoted", "elapsed_s", "feature_config", "model_config", "notes",
)


def _db_path() -> str:
    return os.environ.get("DATABASE_PATH", "/data/glucoassist.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        con = sqlite3.connect(_db_path())
        con.execute(_CREATE_LOG_TABLE)
        con.commit()
        con.close()
        log.info("autoresearcher_log table ready at %s", _db_path())
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not initialise log table: %s", exc)
    yield


app = FastAPI(
    title="GlucoAssist Autoresearcher",
    version=VERSION,
    lifespan=lifespan,
)


@app.get("/api/version")
def get_version() -> dict:
    """Return service version — used by GlucoAssist for compatibility checks."""
    return {"version": VERSION}


class RunRequest(BaseModel):
    n_experiments: int = 10
    program_md: str | None = None
    llm_provider: str = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_api_key: str = ""  # Bearer token for authenticated Ollama proxies (e.g. Helios)
    openai_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"


@app.post("/api/run", status_code=202)
def start_run(body: RunRequest) -> dict:
    """Start an ad-hoc research run. Returns 409 if a run is already in progress."""
    try:
        run_id = ar_service.start_run(
            db_path=_db_path(),
            n_experiments=body.n_experiments,
            ollama_url=body.ollama_url,
            ollama_model=body.ollama_model,
            ollama_api_key=body.ollama_api_key,
            program_md=body.program_md,
            llm_provider=body.llm_provider,
            openai_url=body.openai_url,
            openai_api_key=body.openai_api_key,
            openai_model=body.openai_model,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run_id, "message": f"Started {body.n_experiments} experiments"}


@app.get("/api/status")
def get_status() -> dict:
    """Return current run state: idle | running | error."""
    return ar_service.get_status()


@app.delete("/api/run")
def cancel_run() -> dict:
    """Request cancellation of the running loop (stops after current experiment)."""
    stopped = ar_service.request_stop()
    if not stopped:
        raise HTTPException(status_code=404, detail="No run is currently in progress")
    return {"message": "Stop requested — will halt after current experiment completes"}


@app.get("/api/log")
def get_log(
    limit: int = 50, offset: int = 0, run_id: str | None = None
) -> list[dict]:
    """Return experiment results, optionally filtered by run_id."""
    try:
        con = sqlite3.connect(_db_path())
        cols = ", ".join(_LOG_COLS)
        if run_id:
            rows = con.execute(
                f"SELECT {cols} FROM autoresearcher_log "
                "WHERE run_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (run_id, limit, offset),
            ).fetchall()
        else:
            rows = con.execute(
                f"SELECT {cols} FROM autoresearcher_log "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        con.close()
        return [dict(zip(_LOG_COLS, r)) for r in rows]
    except Exception:  # noqa: BLE001
        return []
