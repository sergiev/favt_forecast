"""Долив ленты, пересборка интервалов и дообучение."""
from __future__ import annotations

import hashlib
import threading
from collections import Counter

from favt_forecast.db import connect, counts, get_state, iter_messages, replace_derived, set_state
from favt_forecast.model import model_ready, train
from favt_forecast.parser import event_row, interval_row, rebuild_from_messages
from favt_forecast.scraper import sync_channel
from favt_forecast.util import iso, now_utc, parse_dt

_LOCK = threading.Lock()
PROGRESS = {
    "running": False,
    "stage": "idle",
    "pages": 0,
    "new": 0,
    "updated": 0,
    "embeds": 0,
    "error": None,
    "detail": "",
}


def auto_sync_enabled() -> bool:
    conn = connect()
    try:
        return get_state(conn, "auto_sync", "1") != "0"
    finally:
        conn.close()


def auto_interval() -> int:
    conn = connect()
    try:
        raw = get_state(conn, "auto_sync_sec", "180") or "180"
    finally:
        conn.close()
    try:
        return max(60, min(3600, int(raw)))
    except ValueError:
        return 180


def update_settings(*, auto_sync: bool | None = None, auto_sync_sec: int | None = None) -> dict:
    conn = connect()
    try:
        if auto_sync is not None:
            set_state(conn, "auto_sync", "1" if auto_sync else "0")
        if auto_sync_sec is not None:
            set_state(conn, "auto_sync_sec", str(max(60, min(3600, int(auto_sync_sec)))))
        conn.commit()
        return {"auto_sync": auto_sync_enabled(), "auto_sync_sec": auto_interval()}
    finally:
        conn.close()


def derived_digest(conn) -> str:
    events = conn.execute(
        "SELECT icao, event_type, event_time FROM airport_events ORDER BY event_time, icao, event_type"
    ).fetchall()
    intervals = conn.execute(
        "SELECT icao, state, start_utc, COALESCE(end_utc, ''), inferred_end "
        "FROM state_intervals ORDER BY icao, start_utc"
    ).fetchall()
    blob = repr([tuple(row) for row in events] + [tuple(row) for row in intervals]).encode()
    return hashlib.sha1(blob).hexdigest()


def reparse() -> dict:
    conn = connect()
    try:
        messages = [
            (int(row["msg_id"]), parse_dt(row["posted_at"]), row["text"] or "", row["reply_text"] or "")
            for row in iter_messages(conn)
        ]
        events, intervals, misses = rebuild_from_messages(messages, now_utc())
        replace_derived(
            conn,
            [event_row(event) for event in events],
            [interval_row(item) for item in intervals],
            misses,
        )
        conn.commit()
        kinds = Counter(event.event_type for event in events)
        summary = {
            "messages": len(messages),
            "events": len(events),
            "intervals": len(intervals),
            "misses": len(misses),
            "by_type": dict(kinds),
        }
        print(
            f"разбор: сообщений {summary['messages']}, событий {summary['events']} {dict(kinds)}, "
            f"интервалов {summary['intervals']}, без аэропорта {summary['misses']}"
        )
        return summary
    finally:
        conn.close()


def ingest_and_update(continue_backfill: bool = True, retrain: bool = True, offline: bool = False) -> dict:
    if not _LOCK.acquire(blocking=False):
        return {"already_running": True}
    try:
        PROGRESS.update(
            running=True,
            stage="start",
            pages=0,
            new=0,
            updated=0,
            embeds=0,
            error=None,
            detail="",
        )
        stats = {"new": 0, "updated": 0, "pages": 0, "embeds": 0, "backfill_complete": False}
        if not offline:

            def on_progress(stage: str, snap: dict) -> None:
                PROGRESS.update(
                    stage=stage,
                    pages=snap.get("pages", 0),
                    new=snap.get("new", 0),
                    updated=snap.get("updated", 0),
                    embeds=snap.get("embeds", 0),
                )
                pages = int(snap.get("pages") or 0)
                if pages == 1 or pages % 10 == 0:
                    print(
                        f"{stage}: страниц {pages}, новых {snap.get('new', 0)}, "
                        f"правок {snap.get('updated', 0)}, embed {snap.get('embeds', 0)}",
                        flush=True,
                    )

            stats = sync_channel(continue_backfill=continue_backfill, on_progress=on_progress)
            print(
                f"лента: новых {stats['new']}, правок {stats['updated']}, "
                f"страниц {stats['pages']}, embed {stats['embeds']}, "
                f"архив {'полный' if stats['backfill_complete'] else 'неполный'}",
                flush=True,
            )
        changed = int(stats.get("new") or 0) + int(stats.get("updated") or 0)
        conn = connect()
        try:
            current = counts(conn)
            before = derived_digest(conn) if current["events"] or current["intervals"] else ""
        finally:
            conn.close()
        need_parse = offline or changed > 0 or (current["messages"] and not current["events"])
        parsed = None
        digest_changed = False
        if need_parse:
            PROGRESS["stage"] = "parse"
            parsed = reparse()
            PROGRESS["detail"] = (
                f"события {parsed['events']}, интервалы {parsed['intervals']}, пропуски {parsed['misses']}"
            )
            conn = connect()
            try:
                digest_changed = derived_digest(conn) != before
            finally:
                conn.close()
        interval_n = parsed["intervals"] if parsed else current["intervals"]
        metrics = None
        if retrain and (digest_changed or (not model_ready() and interval_n >= 30)):
            PROGRESS["stage"] = "train"
            metrics = train()
        PROGRESS["stage"] = "done"
        return {
            "already_running": False,
            "stats": stats,
            "parsed": parsed,
            "model": metrics,
        }
    except Exception as exc:
        PROGRESS["error"] = f"{type(exc).__name__}: {exc}"
        PROGRESS["stage"] = "error"
        raise
    finally:
        PROGRESS["running"] = False
        _LOCK.release()
