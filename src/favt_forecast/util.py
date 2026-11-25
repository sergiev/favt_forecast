from __future__ import annotations

from datetime import datetime, timedelta, timezone

from favt_forecast.config import MSK_OFFSET_HOURS

MSK = timezone(timedelta(hours=MSK_OFFSET_HOURS))
UTC = timezone.utc


def as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime без таймзоны")
    return dt.astimezone(UTC)


def parse_dt(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=MSK)
    return dt.astimezone(UTC)


def iso(dt: datetime) -> str:
    return as_utc(dt).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_hour(dt: datetime) -> datetime:
    dt = as_utc(dt)
    return dt.replace(minute=0, second=0, microsecond=0)


def ceil_hour(dt: datetime) -> datetime:
    floored = floor_hour(dt)
    if floored == as_utc(dt).replace(microsecond=0) and dt.second == 0 and dt.minute == 0:
        return floored
    return floored + timedelta(hours=1)


def now_utc() -> datetime:
    return datetime.now(UTC)
