from __future__ import annotations

import sqlite3
from typing import Iterable, Iterator

from favt_forecast.config import DB_PATH, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_messages (
    msg_id INTEGER PRIMARY KEY,
    posted_at TEXT NOT NULL,
    text TEXT NOT NULL,
    reply_text TEXT NOT NULL DEFAULT '',
    views INTEGER,
    text_hash TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS airport_events (
    id INTEGER PRIMARY KEY,
    msg_id INTEGER NOT NULL,
    icao TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_time TEXT NOT NULL,
    posted_at TEXT NOT NULL,
    time_source TEXT NOT NULL,
    snippet TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_icao_time ON airport_events(icao, event_time);
CREATE INDEX IF NOT EXISTS idx_events_msg ON airport_events(msg_id);

CREATE TABLE IF NOT EXISTS state_intervals (
    id INTEGER PRIMARY KEY,
    icao TEXT NOT NULL,
    state TEXT NOT NULL,
    start_utc TEXT NOT NULL,
    end_utc TEXT,
    inferred_end INTEGER NOT NULL DEFAULT 0,
    start_msg_id INTEGER,
    end_msg_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_intervals_icao_start ON state_intervals(icao, start_utc);

CREATE TABLE IF NOT EXISTS sync_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY,
    trained_at TEXT NOT NULL,
    device TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    n_events INTEGER,
    n_intervals INTEGER,
    train_seconds REAL
);

CREATE TABLE IF NOT EXISTS parse_misses (
    msg_id INTEGER PRIMARY KEY,
    reason TEXT NOT NULL,
    snippet TEXT NOT NULL
);
"""


def connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.executescript(SCHEMA)
    return conn


def get_state(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    return str(row["value"])


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO sync_state(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def message_hashes(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute("SELECT msg_id, text_hash FROM raw_messages").fetchall()
    return {int(r["msg_id"]): str(r["text_hash"]) for r in rows}


def max_msg_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(msg_id) AS m FROM raw_messages").fetchone()
    if row is None or row["m"] is None:
        return 0
    return int(row["m"])


def upsert_message(
    conn: sqlite3.Connection,
    *,
    msg_id: int,
    posted_at: str,
    text: str,
    reply_text: str,
    views: int | None,
    text_hash: str,
    fetched_at: str,
) -> str:
    row = conn.execute("SELECT text_hash FROM raw_messages WHERE msg_id = ?", (msg_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO raw_messages(msg_id, posted_at, text, reply_text, views, text_hash, fetched_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (msg_id, posted_at, text, reply_text, views, text_hash, fetched_at),
        )
        return "new"
    if str(row["text_hash"]) == text_hash:
        return "same"
    conn.execute(
        "UPDATE raw_messages SET posted_at=?, text=?, reply_text=?, views=?, text_hash=?, fetched_at=? "
        "WHERE msg_id=?",
        (posted_at, text, reply_text, views, text_hash, fetched_at, msg_id),
    )
    return "updated"


def iter_messages(conn: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    return conn.execute(
        "SELECT msg_id, posted_at, text, reply_text FROM raw_messages ORDER BY msg_id"
    )


def replace_derived(
    conn: sqlite3.Connection,
    events: Iterable[tuple],
    intervals: Iterable[tuple],
    misses: Iterable[tuple],
) -> None:
    conn.execute("DELETE FROM airport_events")
    conn.execute("DELETE FROM state_intervals")
    conn.execute("DELETE FROM parse_misses")
    conn.executemany(
        "INSERT INTO airport_events(msg_id, icao, event_type, event_time, posted_at, time_source, snippet) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)",
        events,
    )
    conn.executemany(
        "INSERT INTO state_intervals(icao, state, start_utc, end_utc, inferred_end, start_msg_id, end_msg_id) "
        "VALUES(?, ?, ?, ?, ?, ?, ?)",
        intervals,
    )
    conn.executemany(
        "INSERT INTO parse_misses(msg_id, reason, snippet) VALUES(?, ?, ?)",
        misses,
    )


def load_intervals(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT icao, state, start_utc, end_utc, inferred_end, start_msg_id, end_msg_id "
            "FROM state_intervals ORDER BY icao, start_utc, id"
        )
    )


def load_events(conn: sqlite3.Connection, limit: int = 40) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT msg_id, icao, event_type, event_time, posted_at, time_source, snippet "
            "FROM airport_events ORDER BY event_time DESC, id DESC LIMIT ?",
            (limit,),
        )
    )


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    def one(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    last = conn.execute("SELECT MAX(posted_at) FROM raw_messages").fetchone()[0]
    return {
        "messages": one("SELECT COUNT(*) FROM raw_messages"),
        "events": one("SELECT COUNT(*) FROM airport_events"),
        "intervals": one("SELECT COUNT(*) FROM state_intervals"),
        "misses": one("SELECT COUNT(*) FROM parse_misses"),
        "last_message_at": last,
    }


def insert_model_run(
    conn: sqlite3.Connection,
    *,
    trained_at: str,
    device: str,
    metrics_json: str,
    n_events: int,
    n_intervals: int,
    train_seconds: float,
) -> None:
    conn.execute(
        "INSERT INTO model_runs(trained_at, device, metrics_json, n_events, n_intervals, train_seconds) "
        "VALUES(?, ?, ?, ?, ?, ?)",
        (trained_at, device, metrics_json, n_events, n_intervals, train_seconds),
    )


def latest_model_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT trained_at, device, metrics_json, n_events, n_intervals, train_seconds "
        "FROM model_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
