"""
GlucoAssist Autoresearcher Service
------------------------------------
Autonomous model improvement loop using a locally-hosted Ollama LLM or any
OpenAI-compatible API (Claude, Gemini, OpenAI, etc.).
Proposes experiments, runs walk-forward CV, promotes winners.

Runs ad-hoc in a background thread (never scheduled — LLM may be offline).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

HORIZONS: dict[str, int] = {"30": 6, "60": 12, "90": 18, "120": 24}
CV_TRAIN_DAYS = 14
CV_VAL_DAYS = 7
CV_STEP_DAYS = 7
READINGS_PER_DAY = 288

_PROGRAM_MD_PATH = Path(__file__).parent / "autoresearcher_program.md"


@dataclass
class _RunState:
    state: str = "idle"
    run_id: str = ""
    progress: int = 0
    total: int = 0
    error_message: str = ""
    stop_requested: bool = False
    results: list[dict] = field(default_factory=list)


_state = _RunState()
_lock = threading.Lock()
_thread_lock = threading.Lock()


def get_status() -> dict[str, Any]:
    return {
        "state": _state.state,
        "run_id": _state.run_id,
        "progress": _state.progress,
        "total": _state.total,
        "error_message": _state.error_message,
    }


def request_stop() -> bool:
    with _lock:
        if _state.state == "running":
            _state.stop_requested = True
            return True
        return False


class LLMUnreachableError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


def _load_glucose(db_path: str):
    try:
        import pandas as pd
        con = sqlite3.connect(db_path)
        df = pd.read_sql(
            "SELECT timestamp, glucose_mg_dl FROM glucose_readings ORDER BY timestamp",
            con, parse_dates=["timestamp"],
        )
        con.close()
        df = df.set_index("timestamp").sort_index()
        df = df.resample("5min").mean().ffill(limit=3)
        return df
    except ImportError as exc:
        raise RuntimeError("pandas is required for autoresearcher.") from exc


def _load_garmin(db_path: str):
    try:
        import pandas as pd
        con = sqlite3.connect(db_path)
        try:
            df = pd.read_sql(
                "SELECT date, resting_hr, sleep_hours, stress_level FROM garmin_metrics ORDER BY date",
                con, parse_dates=["date"],
            )
        except Exception:
            df = pd.DataFrame()
        con.close()
        return df
    except ImportError:
        import pandas as pd
        return pd.DataFrame()


def _build_features(df, garmin, config: dict):
    import numpy as np
    import pandas as pd

    g = df["glucose_mg_dl"].copy()
    feats = pd.DataFrame(index=df.index)

    for lag in config.get("lags", [1, 2, 3, 6, 12]):
        feats[f"lag_{lag}"] = g.shift(lag)

    if config.get("roc_features", False):
        feats["roc_15"] = g.diff(3)
        feats["roc_30"] = g.diff(6)

    if config.get("time_of_day", False):
        hour = df.index.hour + df.index.minute / 60
        feats["tod_sin"] = np.sin(2 * np.pi * hour / 24)
        feats["tod_cos"] = np.cos(2 * np.pi * hour / 24)

    if config.get("garmin_features", False) and not garmin.empty:
        garmin_daily = garmin.set_index("date")
        feats["date"] = df.index.normalize()
        feats = feats.join(garmin_daily[["sleep_hours", "stress_level"]], on="date", how="left")
        feats.drop(columns=["date"], inplace=True)

    feats["glucose"] = g
    return feats.dropna()


def _build_model(config: dict, x_train, y_train):
    algo = config.get("algorithm", "ridge")
    if algo == "ridge":
        m = Ridge(alpha=config.get("alpha", 1.0))
        m.fit(x_train, y_train)
        return m
    elif algo == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise RuntimeError("lightgbm is required. pip install lightgbm") from exc
        m = lgb.LGBMRegressor(
            n_estimators=config.get("n_estimators", 200),
            learning_rate=config.get("learning_rate", 0.05),
            num_leaves=config.get("num_leaves", 31),
            verbosity=-1,
        )
        m.fit(x_train, y_train)
        return m
    raise ValueError(f"Unknown algorithm: {algo!r} (supported: ridge, lightgbm)")


def _walk_forward_cv(feats, model_config: dict) -> dict[str, float]:
    import pandas as pd

    train_n = CV_TRAIN_DAYS * READINGS_PER_DAY
    val_n   = CV_VAL_DAYS   * READINGS_PER_DAY
    step_n  = CV_STEP_DAYS  * READINGS_PER_DAY
    total   = len(feats)

    feature_cols = [c for c in feats.columns if c != "glucose"]
    all_maes: dict[str, list[float]] = {h: [] for h in HORIZONS}

    start = 0
    while start + train_n + val_n <= total:
        train_slice = feats.iloc[start : start + train_n]
        val_slice   = feats.iloc[start + train_n : start + train_n + val_n]

        scaler = StandardScaler()
        x_train_s = pd.DataFrame(scaler.fit_transform(train_slice[feature_cols]), columns=feature_cols)
        x_val_s   = pd.DataFrame(scaler.transform(val_slice[feature_cols]),       columns=feature_cols)
        g_val = val_slice["glucose"].values

        for horizon_label, steps in HORIZONS.items():
            y_train = train_slice["glucose"].shift(-steps).dropna()
            x_tr    = x_train_s.iloc[: len(y_train)]
            model   = _build_model(model_config, x_tr, y_train.values)
            preds   = model.predict(x_val_s)
            if steps < len(g_val):
                actuals       = g_val[steps:]
                preds_aligned = preds[: len(actuals)]
                all_maes[horizon_label].append(mean_absolute_error(actuals, preds_aligned))

        start += step_n

    return {h: float(np.mean(v)) if v else float("inf") for h, v in all_maes.items()}


def _propose_experiment(program_md, research_log, ollama_url, ollama_model, ollama_api_key="") -> dict:
    log_summary = json.dumps(research_log[-5:], indent=2) if research_log else "[]"
    prompt = (
        "You are running the GlucoAssist forecast research loop.\n\n"
        f"<program>\n{program_md}\n</program>\n\n"
        f"<log>\n{log_summary}\n</log>\n\n"
        "Propose the SINGLE next experiment. Respond ONLY with a valid JSON object with:\n"
        '  "description": "one sentence",\n'
        '  "feature_config": {"lags": [ints], "time_of_day": bool, "roc_features": bool, "garmin_features": bool},\n'
        '  "model_config": {"algorithm": "ridge"|"lightgbm", "alpha": 1.0, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31}'
    )
    try:
        headers: dict = {"Content-Type": "application/json"}
        if ollama_api_key:
            headers["Authorization"] = f"Bearer {ollama_api_key}"
        resp = requests.post(
            ollama_url.rstrip("/") + "/api/generate",
            headers=headers,
            json={"model": ollama_model, "prompt": prompt, "format": "json", "stream": False, "think": False},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise LLMUnreachableError(f"Cannot connect to Ollama at {ollama_url}. Is it running?") from exc
    except requests.exceptions.HTTPError as exc:
        raise LLMResponseError(f"Ollama returned error: {exc}") from exc

    payload = resp.json()
    raw = payload.get("response", "").strip() or payload.get("message", {}).get("content", "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"Ollama returned non-JSON: {raw[:200]}") from exc


def _propose_experiment_openai(program_md, research_log, api_url, api_key, model) -> dict:
    log_summary = json.dumps(research_log[-5:], indent=2) if research_log else "[]"
    prompt = (
        "You are running the GlucoAssist forecast research loop.\n\n"
        f"<program>\n{program_md}\n</program>\n\n"
        f"<log>\n{log_summary}\n</log>\n\n"
        "Propose the SINGLE next experiment. Respond ONLY with a valid JSON object with:\n"
        '  "description": "one sentence",\n'
        '  "feature_config": {"lags": [ints], "time_of_day": bool, "roc_features": bool, "garmin_features": bool},\n'
        '  "model_config": {"algorithm": "ridge"|"lightgbm", "alpha": 1.0, "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31}'
    )
    try:
        resp = requests.post(
            api_url.rstrip("/") + "/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise LLMUnreachableError(f"Cannot connect to OpenAI-compatible API at {api_url}.") from exc
    except requests.exceptions.HTTPError as exc:
        raise LLMResponseError(f"API returned error: {exc}") from exc

    raw = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"API returned non-JSON: {raw[:200]}") from exc


def _should_promote(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    return all(candidate[h] < baseline[h] for h in HORIZONS)


def _persist_result(db_path: str, run_id: str, exp_idx: int, result: dict) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO autoresearcher_log
            (run_id, experiment_id, timestamp, description,
             mae_30, mae_60, mae_90, mae_120,
             promoted, elapsed_s, feature_config, model_config, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, exp_idx,
            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            result["description"],
            result.get("mae_30"), result.get("mae_60"),
            result.get("mae_90"), result.get("mae_120"),
            1 if result["promoted"] else 0,
            result.get("elapsed_s"),
            json.dumps(result.get("feature_config")),
            json.dumps(result.get("model_config")),
            result.get("notes"),
        ),
    )
    con.commit()
    con.close()


def _save_promoted_config(db_path: str, proposal: dict) -> None:
    config_path = Path(db_path).parent / "promoted_model_config.json"
    with open(config_path, "w") as f:
        json.dump(proposal, f, indent=2)
    log.info("Promoted config saved to %s", config_path)


def get_default_program_md() -> str:
    return _PROGRAM_MD_PATH.read_text()


def _run_loop(
    db_path, n_experiments, ollama_url, ollama_model, run_id, program_md,
    llm_provider="ollama", ollama_api_key="",
    openai_url="", openai_api_key="", openai_model="gpt-4o",
) -> None:
    global _state  # noqa: PLW0603

    try:
        log.info("Loading glucose data from %s", db_path)
        df     = _load_glucose(db_path)
        garmin = _load_garmin(db_path)
        log.info("%d readings loaded", len(df))

        con  = sqlite3.connect(db_path)
        rows = con.execute(
            "SELECT description, mae_30, mae_60, mae_90, mae_120, promoted "
            "FROM autoresearcher_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        con.close()
        prior_log = [
            {"description": r[0], "mae_30": r[1], "mae_60": r[2], "mae_90": r[3], "mae_120": r[4], "promoted": bool(r[5])}
            for r in reversed(rows)
        ]

        baseline  = {"30": 17.74, "60": 25.18, "90": 27.73, "120": 28.60}
        meta_path = Path(db_path).parent / "model_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                for h_label, h_key in [("30","h30"),("60","h60"),("90","h90"),("120","h120")]:
                    if h_key in meta.get("mae", {}):
                        baseline[h_label] = meta["mae"][h_key]
            except Exception as exc:
                log.debug("Could not read model_meta.json: %s", exc)

        for i in range(n_experiments):
            with _lock:
                if _state.stop_requested:
                    _state.state = "idle"
                    _state.stop_requested = False
                    return
                _state.progress = i + 1

            log.info("Experiment %d/%d", i + 1, n_experiments)

            try:
                if llm_provider == "openai_compatible":
                    proposal = _propose_experiment_openai(program_md, prior_log, openai_url, openai_api_key, openai_model)
                else:
                    proposal = _propose_experiment(program_md, prior_log, ollama_url, ollama_model, ollama_api_key)
            except LLMUnreachableError as exc:
                with _lock:
                    _state.state = "error"
                    _state.error_message = str(exc)
                log.error("%s", exc)
                return
            except (LLMResponseError, Exception) as exc:
                log.warning("Proposal failed for exp %d: %s", i + 1, exc)
                result = {"description": f"[Proposal failed: {exc}]", "promoted": False, "notes": str(exc)}
                _persist_result(db_path, run_id, i + 1, result)
                prior_log.append(result)
                continue

            try:
                feats   = _build_features(df, garmin, proposal.get("feature_config", {}))
                t0      = time.time()
                maes    = _walk_forward_cv(feats, proposal.get("model_config", {}))
                elapsed = time.time() - t0
            except Exception as exc:
                log.warning("CV failed for exp %d: %s", i + 1, exc)
                result = {
                    "description":   proposal.get("description", "?"),
                    "promoted":      False,
                    "notes":         f"CV error: {exc}",
                    "feature_config": proposal.get("feature_config"),
                    "model_config":   proposal.get("model_config"),
                }
                _persist_result(db_path, run_id, i + 1, result)
                prior_log.append(result)
                continue

            promoted = _should_promote(maes, baseline)
            if promoted:
                baseline = dict(maes)
                _save_promoted_config(db_path, proposal)

            result = {
                "description":    proposal.get("description", "?"),
                "mae_30":         maes.get("30"),  "mae_60":   maes.get("60"),
                "mae_90":         maes.get("90"),  "mae_120":  maes.get("120"),
                "promoted":       promoted,
                "elapsed_s":      round(elapsed, 1),
                "feature_config": proposal.get("feature_config"),
                "model_config":   proposal.get("model_config"),
            }
            _persist_result(db_path, run_id, i + 1, result)
            prior_log.append(result)
            log.info("Exp %d MAE 30:%.2f 60:%.2f 90:%.2f 120:%.2f %s",
                     i+1, maes.get("30",0), maes.get("60",0), maes.get("90",0), maes.get("120",0),
                     "PROMOTED" if promoted else "not promoted")

        with _lock:
            _state.state    = "idle"
            _state.progress = n_experiments
        log.info("Run %s complete (%d experiments)", run_id, n_experiments)

    except Exception as exc:
        with _lock:
            _state.state         = "error"
            _state.error_message = str(exc)
        log.exception("Unexpected error in run %s", run_id)


def start_run(
    db_path, n_experiments, ollama_url, ollama_model,
    program_md=None, llm_provider="ollama", ollama_api_key="",
    openai_url="", openai_api_key="", openai_model="gpt-4o",
) -> str:
    """Start a research run in a background thread. Raises RuntimeError if already running."""
    if program_md is None:
        program_md = get_default_program_md()
    if not _thread_lock.acquire(blocking=False):
        raise RuntimeError("A research run is already in progress")

    run_id = str(uuid.uuid4())
    with _lock:
        _state.state         = "running"
        _state.run_id        = run_id
        _state.progress      = 0
        _state.total         = n_experiments
        _state.error_message = ""
        _state.stop_requested = False

    def _worker():
        try:
            _run_loop(
                db_path, n_experiments, ollama_url, ollama_model, run_id, program_md,
                llm_provider=llm_provider, ollama_api_key=ollama_api_key,
                openai_url=openai_url, openai_api_key=openai_api_key, openai_model=openai_model,
            )
        finally:
            _thread_lock.release()

    threading.Thread(target=_worker, name=f"autoresearcher-{run_id[:8]}", daemon=True).start()
    log.info("Started run %s (%d experiments)", run_id, n_experiments)
    return run_id
