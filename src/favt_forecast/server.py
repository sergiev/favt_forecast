"""Локальный API и веб-интерфейс прогноза."""
from __future__ import annotations

import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from favt_forecast.airports import resolve
from favt_forecast.config import HORIZON_HOURS
from favt_forecast.db import connect, counts, get_state, latest_model_run, load_events
from favt_forecast.model import load_predictor, model_ready
from favt_forecast.service import PROGRESS, auto_interval, auto_sync_enabled, ingest_and_update, update_settings
from favt_forecast.util import iso, parse_dt

STATIC = Path(__file__).resolve().parent / "static" / "index.html"
_STOP = threading.Event()
_THREAD: threading.Thread | None = None
_CACHE: dict = {"stamp": None, "predictor": None}


class SettingsIn(BaseModel):
    auto_sync: bool | None = None
    auto_sync_sec: int | None = Field(default=None, ge=60, le=3600)


def _stamp() -> str:
    conn = connect()
    try:
        current = counts(conn)
        run = latest_model_run(conn)
        trained = run["trained_at"] if run else ""
    finally:
        conn.close()
    return f"{trained}:{current['messages']}:{current['events']}:{current['intervals']}"


def _predictor():
    if not model_ready():
        raise HTTPException(status_code=409, detail="Модель ещё не обучена")
    stamp = _stamp()
    if _CACHE["stamp"] != stamp or _CACHE["predictor"] is None:
        try:
            _CACHE["predictor"] = load_predictor()
            _CACHE["stamp"] = stamp
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail="Модель ещё не обучена") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _CACHE["predictor"]


def _airport(value: str) -> str:
    airport = resolve(value)
    if airport is None:
        raise HTTPException(status_code=404, detail=f"Неизвестный аэропорт: {value}")
    return airport.icao


def _safe_ingest() -> None:
    try:
        ingest_and_update()
    except Exception as exc:
        PROGRESS["error"] = f"{type(exc).__name__}: {exc}"
        PROGRESS["stage"] = "error"


def _loop() -> None:
    if not _STOP.is_set():
        _safe_ingest()
    while not _STOP.is_set():
        if _STOP.wait(auto_interval()):
            return
        if auto_sync_enabled():
            _safe_ingest()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _THREAD
    _STOP.clear()
    _THREAD = threading.Thread(target=_loop, name="favt-sync", daemon=True)
    _THREAD.start()
    yield
    _STOP.set()


app = FastAPI(title="Прогноз ограничений аэропортов РФ", lifespan=_lifespan)


@app.get("/api/status")
def status() -> dict:
    conn = connect()
    try:
        current = counts(conn)
        run = latest_model_run(conn)
        model = None
        if run is not None:
            metrics = json.loads(run["metrics_json"])
            near = (metrics.get("near") or {}).get("metrics") or {}
            far = (metrics.get("far") or {}).get("metrics") or {}
            model = {
                "trained_at": run["trained_at"],
                "device": run["device"],
                "train_seconds": run["train_seconds"],
                "logloss_near": near.get("logloss"),
                "logloss_far": far.get("logloss"),
                "brier_near": near.get("brier"),
                "brier_far": far.get("brier"),
                "brier_closed_near": near.get("brier_closed"),
                "brier_closed_far": far.get("brier_closed"),
                "brier_prior_near": near.get("brier_prior"),
                "base_closed_near": near.get("base_closed"),
                "inference_ms_point": metrics.get("inference_ms_point"),
                "inference_ms_horizon": metrics.get("inference_ms_horizon"),
                "inference_ms_network": metrics.get("inference_ms_network"),
                "alpha_near": (metrics.get("near") or {}).get("alpha"),
                "alpha_far": (metrics.get("far") or {}).get("alpha"),
            }
        return {
            "messages": current["messages"],
            "events": current["events"],
            "intervals": current["intervals"],
            "misses": current["misses"],
            "last_message_at": current["last_message_at"],
            "last_sync_at": get_state(conn, "last_sync_at"),
            "backfill_complete": get_state(conn, "backfill_complete", "0") == "1",
            "auto_sync": get_state(conn, "auto_sync", "1") != "0",
            "auto_sync_sec": auto_interval(),
            "sync": dict(PROGRESS),
            "model": model,
        }
    finally:
        conn.close()


@app.get("/api/airports")
def airports(cluster: str | None = None, q: str | None = None) -> dict:
    predictor = _predictor()
    rows = predictor.summaries()
    issued = iso(predictor.cube()["now"])
    if cluster:
        rows = [row for row in rows if row["cluster"] == cluster]
    if q:
        needle = q.casefold()
        rows = [
            row
            for row in rows
            if needle in row["label"].casefold()
            or needle in row["icao"].casefold()
            or needle in row["iata"].casefold()
            or needle in row["city"].casefold()
            or needle in row["name"].casefold()
        ]
    return {"airports": rows, "issued_at": issued}


@app.get("/api/forecast")
def forecast(airport: str = Query(...), horizon_hours: int = HORIZON_HOURS) -> dict:
    horizon_hours = max(1, min(HORIZON_HOURS, horizon_hours))
    return _predictor().forecast(_airport(airport), horizon_hours)


@app.get("/api/predict_point")
def predict_point(airport: str = Query(...), target_ts: str = Query(...)) -> dict:
    try:
        target = parse_dt(target_ts)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не разобран момент времени: {target_ts}") from exc
    return _predictor().predict_point(_airport(airport), target)


@app.get("/api/events")
def events(limit: int = 40) -> dict:
    limit = max(1, min(200, limit))
    conn = connect()
    try:
        rows = []
        for row in load_events(conn, limit):
            airport = resolve(row["icao"])
            rows.append(
                {
                    "msg_id": row["msg_id"],
                    "icao": row["icao"],
                    "iata": airport.iata if airport else "",
                    "name": airport.name if airport else row["icao"],
                    "label": airport.label if airport else row["icao"],
                    "event_type": row["event_type"],
                    "event_time": row["event_time"],
                    "posted_at": row["posted_at"],
                    "time_source": row["time_source"],
                    "snippet": row["snippet"],
                }
            )
        return {"events": rows}
    finally:
        conn.close()


@app.post("/api/sync")
def sync_now() -> dict:
    if PROGRESS["running"]:
        return {"already_running": True}
    threading.Thread(target=_safe_ingest, name="favt-sync-manual", daemon=True).start()
    return {"started": True}


@app.post("/api/settings")
def settings(body: SettingsIn) -> dict:
    return update_settings(auto_sync=body.auto_sync, auto_sync_sec=body.auto_sync_sec)


@app.get("/")
def index() -> FileResponse:
    if not STATIC.exists():
        raise HTTPException(status_code=500, detail="Нет static/index.html")
    return FileResponse(STATIC)
