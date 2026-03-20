"""API-level tests for the autoresearcher service."""
from __future__ import annotations

import os
import sqlite3
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import VERSION, app
from app.services import autoresearcher as ar_service

client = TestClient(app)


def test_version():
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert resp.json()["version"] == VERSION


def test_status_idle_on_start():
    resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["state"] in ("idle", "error")


def test_cancel_no_run_returns_404():
    with ar_service._lock:
        ar_service._state.state = "idle"
    resp = client.delete("/api/run")
    assert resp.status_code == 404


def test_start_run_returns_202():
    with patch.object(ar_service, "start_run", return_value="mock-run-id"):
        resp = client.post("/api/run", json={"n_experiments": 1})
    assert resp.status_code == 202
    assert resp.json()["run_id"] == "mock-run-id"


def test_start_run_conflict_returns_409():
    with patch.object(ar_service, "start_run", side_effect=RuntimeError("already running")):
        resp = client.post("/api/run", json={"n_experiments": 1})
    assert resp.status_code == 409


def test_log_returns_list(tmp_path):
    db = tmp_path / "test.db"
    con = sqlite3.connect(db)
    con.execute(
        """CREATE TABLE autoresearcher_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, experiment_id INTEGER,
            timestamp TEXT, description TEXT, mae_30 REAL, mae_60 REAL, mae_90 REAL,
            mae_120 REAL, promoted INTEGER, elapsed_s REAL,
            feature_config TEXT, model_config TEXT, notes TEXT
        )"""
    )
    con.commit()
    con.close()
    with patch.dict(os.environ, {"DATABASE_PATH": str(db)}):
        resp = client.get("/api/log")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_log_missing_db_returns_empty_list(tmp_path):
    with patch.dict(os.environ, {"DATABASE_PATH": str(tmp_path / "nonexistent.db")}):
        resp = client.get("/api/log")
    assert resp.status_code == 200
    assert resp.json() == []
