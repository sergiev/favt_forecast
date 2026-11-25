"""Признаки для прогноза состояния аэропорта в момент t0 + h."""
from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np

from favt_forecast.airports import AIRPORTS
from favt_forecast.parser import CLOSED, Interval, OPEN, RESTRICTED
from favt_forecast.util import MSK, UTC, floor_hour

STATE_CODE = {OPEN: 0, RESTRICTED: 1, CLOSED: 2}
WINDOWS = (3, 6, 12, 24, 72, 168, 336, 720)
START_WINDOWS = (24, 72, 168, 336, 720)
LAGS = (24, 48, 72, 168)
RADII = (250, 500, 1000)
SPATIAL_W = (1, 3, 6, 24)
SURV_HORIZON = 96


def _feature_names() -> tuple[str, ...]:
    names = [
        "h",
        "log1p_h",
        "flag_h_le_6",
        "flag_h_le_48",
        "flag_h_gt_48",
        "sin_hour_msk",
        "cos_hour_msk",
        "sin_hour_local",
        "cos_hour_local",
        "sin_dow",
        "cos_dow",
        "sin_doy",
        "cos_doy",
        "is_weekend",
        "is_night",
        "cur_restricted",
        "cur_closed",
        "log1p_age",
        "surv_cont",
        "log1p_since_close",
    ]
    for width in WINDOWS:
        names.append(f"closed_frac_{width}")
        names.append(f"restr_frac_{width}")
    for width in START_WINDOWS:
        names.append(f"close_starts_{width}")
    for lag in LAGS:
        names.append(f"lag_closed_{lag}")
        names.append(f"lag_restr_{lag}")
    for radius in RADII:
        for width in SPATIAL_W:
            names.append(f"nei{radius}_closed_{width}")
            names.append(f"nei{radius}_restr_{width}")
    for width in SPATIAL_W:
        names.append(f"clu_closed_{width}")
        names.append(f"clu_restr_{width}")
    names.extend(
        [
            "prior14_restr",
            "prior14_closed",
            "prior60_restr",
            "prior60_closed",
            "global14_closed",
        ]
    )
    return tuple(names)


FEATURE_NAMES = _feature_names()
FEATURE_INDEX = {name: pos for pos, name in enumerate(FEATURE_NAMES)}

FACTOR_SPECS = (
    ("h", "Горизонт, ч", "num"),
    ("cur_closed", "Сейчас закрыт", "bool"),
    ("cur_restricted", "Сейчас по согласованию", "bool"),
    ("log1p_age", "Часов в текущем состоянии", "hours"),
    ("surv_cont", "P(состояние продолжится)", "pct"),
    ("log1p_since_close", "Часов с последнего закрытия", "hours"),
    ("closed_frac_24", "Доля закрытия за 24 ч", "pct"),
    ("closed_frac_168", "Доля закрытия за 7 суток", "pct"),
    ("closed_frac_720", "Доля закрытия за 30 суток", "pct"),
    ("close_starts_168", "Закрытий за 7 суток", "num"),
    ("nei250_closed_6", "Соседи ≤250 км, закрытие за 6 ч", "pct"),
    ("nei500_closed_24", "Соседи ≤500 км, закрытие за 24 ч", "pct"),
    ("clu_closed_24", "Кластер, закрытие за 24 ч", "pct"),
    ("prior14_closed", "Прайор закрытия этого часа, 14 суток", "pct"),
    ("prior60_closed", "Прайор закрытия этого часа, 60 суток", "pct"),
    ("global14_closed", "Вся сеть, прайор этого часа", "pct"),
    ("lag_closed_24", "Сутки назад в этот час был закрыт", "bool"),
    ("lag_closed_168", "Неделю назад в этот час был закрыт", "bool"),
    ("is_night", "Ночной час МСК", "bool"),
)


def explain_row(row: np.ndarray) -> list[dict]:
    factors = []
    for name, label, kind in FACTOR_SPECS:
        raw = float(row[FEATURE_INDEX[name]])
        if kind == "hours":
            value = float(np.expm1(raw))
            display = f"{value:.1f}"
        elif kind == "pct":
            value = raw
            display = f"{100 * raw:.1f}%"
        elif kind == "bool":
            value = raw
            display = "да" if raw >= 0.5 else "нет"
        else:
            value = raw
            display = f"{raw:.2f}"
        factors.append({"key": name, "label": label, "value": value, "display": display})
    return factors


def _haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _norm_prob(mass: np.ndarray) -> np.ndarray:
    lifted = mass + 0.5
    return (lifted / lifted.sum(axis=-1, keepdims=True)).astype(np.float32)


def _survival_curve(durations: list[float]) -> np.ndarray:
    grid = np.arange(SURV_HORIZON + 1, dtype=np.float32)
    if not durations:
        return np.exp(-grid / 6.0).astype(np.float32)
    values = np.asarray(durations, dtype=np.float32)
    return np.array([(values > hour).mean() for hour in grid], dtype=np.float32)


class Timeline:
    def __init__(self, intervals: list[Interval], moment: datetime):
        self.airports = list(AIRPORTS)
        self.n = len(self.airports)
        self.index = {airport.icao: pos for pos, airport in enumerate(self.airports)}
        self.offsets = np.array([airport.tz_offset for airport in self.airports], dtype=np.float32)
        if intervals:
            earliest = min(item.start for item in intervals) - timedelta(days=2)
        else:
            earliest = moment - timedelta(days=14)
        self.grid_start = floor_hour(earliest)
        self.grid_end = floor_hour(moment)
        if self.grid_end <= self.grid_start:
            self.grid_end = self.grid_start + timedelta(hours=1)
        self.n_hours = int((self.grid_end - self.grid_start).total_seconds() // 3600) + 1
        self._build_state(intervals)
        self._build_prefixes()
        self._build_space()
        self._build_calendar()
        self._build_survival(intervals)
        self._build_priors()
        self._index_intervals(intervals)

    def hour_index(self, moment: datetime) -> int:
        sec = (moment - self.grid_start).total_seconds()
        return int(math.floor(sec / 3600))

    def _point_index(self, moment: datetime) -> int:
        sec = (moment - self.grid_start).total_seconds()
        return math.ceil(sec / 3600 - 1e-9)

    def _build_state(self, intervals: list[Interval]) -> None:
        hours = self.n_hours
        self.state = np.zeros((self.n, hours), dtype=np.int8)
        self.age = np.ones((self.n, hours), dtype=np.float32)
        starts = np.zeros((self.n, hours), dtype=np.float32)
        grid_unix = self.grid_start.timestamp() + np.arange(hours, dtype=np.float64) * 3600.0
        far_end = self.grid_end + timedelta(hours=1)
        for code in (1, 2):
            for item in intervals:
                if STATE_CODE[item.state] != code or item.icao not in self.index:
                    continue
                end = item.end or far_end
                i0 = max(0, self._point_index(item.start))
                i1 = min(hours, max(i0, self._point_index(end)))
                if i1 <= i0:
                    continue
                row = self.index[item.icao]
                self.state[row, i0:i1] = code
                slots = np.arange(i0, i1)
                self.age[row, slots] = (grid_unix[slots] - item.start.timestamp()) / 3600.0
                if code == 2:
                    starts[row, i0] += 1.0
        for hour in range(1, hours):
            same = self.state[:, hour] == self.state[:, hour - 1]
            open_mask = self.state[:, hour] == 0
            self.age[:, hour] = np.where(
                open_mask,
                np.where(same, self.age[:, hour - 1] + 1.0, 1.0),
                self.age[:, hour],
            )
        self.since_close = np.full((self.n, hours), 720.0, dtype=np.float32)
        last = np.full(self.n, -1, dtype=np.int32)
        for hour in range(hours):
            closed = self.state[:, hour] == 2
            elapsed = np.where(last < 0, 720.0, hour - last).astype(np.float32)
            self.since_close[:, hour] = np.where(closed, 0.0, np.minimum(elapsed, 720.0))
            last = np.where(closed, hour, last)
        self._close_starts = starts

    def _build_prefixes(self) -> None:
        hours = self.n_hours
        self.ps_closed = np.zeros((self.n, hours + 1), dtype=np.float32)
        self.ps_restr = np.zeros((self.n, hours + 1), dtype=np.float32)
        self.ps_starts = np.zeros((self.n, hours + 1), dtype=np.float32)
        self.ps_closed[:, 1:] = np.cumsum(self.state == 2, axis=1)
        self.ps_restr[:, 1:] = np.cumsum(self.state == 1, axis=1)
        self.ps_starts[:, 1:] = np.cumsum(self._close_starts, axis=1)

    def _build_space(self) -> None:
        lat = np.array([airport.lat for airport in self.airports], dtype=np.float64)
        lon = np.array([airport.lon for airport in self.airports], dtype=np.float64)
        dist = _haversine_km(lat, lon)
        eye = np.eye(self.n, dtype=bool)
        self.nei_closed: dict[int, np.ndarray] = {}
        self.nei_restr: dict[int, np.ndarray] = {}
        self.nei_count: dict[int, np.ndarray] = {}
        for radius in RADII:
            mask = ((dist <= radius) & ~eye).astype(np.float32)
            self.nei_closed[radius] = mask @ self.ps_closed
            self.nei_restr[radius] = mask @ self.ps_restr
            self.nei_count[radius] = mask.sum(axis=1)
        clusters = sorted({airport.cluster for airport in self.airports})
        self.cluster_names = clusters
        cluster_of = np.array(
            [clusters.index(airport.cluster) for airport in self.airports], dtype=np.int32
        )
        self.cluster_of = cluster_of
        n_clusters = len(clusters)
        self.cluster_ps_closed = np.zeros((n_clusters, self.n_hours + 1), dtype=np.float32)
        self.cluster_ps_restr = np.zeros((n_clusters, self.n_hours + 1), dtype=np.float32)
        self.cluster_size = np.zeros(n_clusters, dtype=np.float32)
        for cluster_id in range(n_clusters):
            members = np.flatnonzero(cluster_of == cluster_id)
            self.cluster_size[cluster_id] = float(len(members))
            if len(members):
                self.cluster_ps_closed[cluster_id] = self.ps_closed[members].sum(axis=0)
                self.cluster_ps_restr[cluster_id] = self.ps_restr[members].sum(axis=0)

    def _build_calendar(self) -> None:
        hours = self.n_hours
        msk_hour = np.zeros(hours, dtype=np.int16)
        dow = np.zeros(hours, dtype=np.int8)
        doy = np.zeros(hours, dtype=np.int16)
        for hour in range(hours):
            moment = (self.grid_start + timedelta(hours=hour)).astimezone(MSK)
            msk_hour[hour] = moment.hour
            dow[hour] = moment.weekday()
            doy[hour] = moment.timetuple().tm_yday
        self.msk_hour = msk_hour
        self.dow = dow
        self.doy = doy

    def _build_survival(self, intervals: list[Interval]) -> None:
        per_closed: list[list[float]] = [[] for _ in range(self.n)]
        per_restr: list[list[float]] = [[] for _ in range(self.n)]
        all_closed: list[float] = []
        all_restr: list[float] = []
        for item in intervals:
            if item.end is None or item.icao not in self.index:
                continue
            duration = (item.end - item.start).total_seconds() / 3600.0
            if duration <= 0:
                continue
            bucket = per_closed if item.state == CLOSED else per_restr if item.state == RESTRICTED else None
            pool = all_closed if item.state == CLOSED else all_restr if item.state == RESTRICTED else None
            if bucket is None or pool is None:
                continue
            bucket[self.index[item.icao]].append(duration)
            pool.append(duration)
        global_closed = _survival_curve(all_closed)
        global_restr = _survival_curve(all_restr)
        self.surv_closed = np.zeros((self.n, SURV_HORIZON + 1), dtype=np.float32)
        self.surv_restr = np.zeros((self.n, SURV_HORIZON + 1), dtype=np.float32)
        for row in range(self.n):
            self.surv_closed[row] = _shrink(per_closed[row], global_closed)
            self.surv_restr[row] = _shrink(per_restr[row], global_restr)

    def _build_priors(self) -> None:
        hours = self.n_hours
        n_days = (hours - 1) // 24 + 1
        mass14 = np.zeros((self.n, 24, 3), dtype=np.float32)
        mass60 = np.zeros((self.n, 24, 3), dtype=np.float32)
        global_mass = np.zeros((24, 3), dtype=np.float32)
        decay14 = np.float32(0.5 ** (1 / (14 * 24)))
        decay60 = np.float32(0.5 ** (1 / (60 * 24)))
        self.prior14 = np.zeros((n_days, self.n, 24, 3), dtype=np.float32)
        self.prior60 = np.zeros((n_days, self.n, 24, 3), dtype=np.float32)
        self.global14_closed = np.zeros((n_days, 24), dtype=np.float32)
        rows = np.arange(self.n)
        for hour in range(hours):
            if hour % 24 == 0:
                day = hour // 24
                self.prior14[day] = _norm_prob(mass14)
                self.prior60[day] = _norm_prob(mass60)
                probs = _norm_prob(global_mass)
                self.global14_closed[day] = probs[:, 2]
            hod = int(self.msk_hour[hour])
            state = self.state[:, hour]
            mass14 *= decay14
            mass60 *= decay60
            global_mass *= decay14
            mass14[rows, hod, state] += 1.0
            mass60[rows, hod, state] += 1.0
            global_mass[hod] += np.bincount(state, minlength=3)

    def _index_intervals(self, intervals: list[Interval]) -> None:
        grouped: list[list[tuple[datetime, datetime, int, bool]]] = [[] for _ in range(self.n)]
        far = datetime(9999, 1, 1, tzinfo=UTC)
        for item in intervals:
            row = self.index.get(item.icao)
            if row is None:
                continue
            grouped[row].append((item.start, item.end or far, STATE_CODE[item.state], item.inferred_end))
        for row in grouped:
            row.sort(key=lambda item: item[0])
        self.by_airport = grouped

    def exact(self, row: int, moment: datetime) -> tuple[int, float, float, bool]:
        code = 0
        age = None
        inferred = False
        last_close_end: datetime | None = None
        last_end: datetime | None = None
        for start, end, state, was_inferred in self.by_airport[row]:
            if state == 2 and end <= moment and end.year < 9000:
                last_close_end = end
            if end.year < 9000 and end <= moment and (last_end is None or end > last_end):
                last_end = end
            if start <= moment < end and state >= code:
                code = state
                age = (moment - start).total_seconds() / 3600.0
                inferred = was_inferred
        if age is None:
            base = last_end or self.grid_start
            age = max(0.0, (moment - base).total_seconds() / 3600.0)
            for index, (start, end, _state, was_inferred) in enumerate(self.by_airport[row]):
                if not was_inferred or end.year > 9000 or not (end <= moment < end + timedelta(hours=48)):
                    continue
                nxt = self.by_airport[row][index + 1][0] if index + 1 < len(self.by_airport[row]) else None
                if nxt is None or moment < nxt:
                    inferred = True
                    break
        since = 0.0 if code == 2 else 720.0
        if code != 2 and last_close_end is not None:
            since = min(720.0, max(0.0, (moment - last_close_end).total_seconds() / 3600.0))
        return code, float(age), float(since), inferred

    def _window(self, prefix: np.ndarray, idx: np.ndarray, hour: int, width: int) -> np.ndarray:
        left = max(0, hour - width)
        span = hour - left
        if span <= 0:
            return np.zeros(len(idx), dtype=np.float32)
        return ((prefix[idx, hour] - prefix[idx, left]) / float(span)).astype(np.float32)

    def _neighbor_frac(self, radius: int, kind: str, idx: np.ndarray, hour: int, width: int) -> np.ndarray:
        prefix = self.nei_closed[radius] if kind == "closed" else self.nei_restr[radius]
        left = max(0, hour - width)
        span = hour - left
        if span <= 0:
            return np.zeros(len(idx), dtype=np.float32)
        count = self.nei_count[radius][idx]
        num = prefix[idx, hour] - prefix[idx, left]
        den = span * np.maximum(count, 1.0)
        return np.where(count > 0, num / den, 0.0).astype(np.float32)

    def _cluster_frac(self, kind: str, idx: np.ndarray, hour: int, width: int) -> np.ndarray:
        prefix = self.cluster_ps_closed if kind == "closed" else self.cluster_ps_restr
        own = self.ps_closed if kind == "closed" else self.ps_restr
        left = max(0, hour - width)
        span = hour - left
        if span <= 0:
            return np.zeros(len(idx), dtype=np.float32)
        clusters = self.cluster_of[idx]
        num = (prefix[clusters, hour] - prefix[clusters, left]) - (own[idx, hour] - own[idx, left])
        others = np.maximum(self.cluster_size[clusters] - 1.0, 1.0)
        frac = num / (span * others)
        return np.where(self.cluster_size[clusters] > 1, frac, 0.0).astype(np.float32)

    def _survival_feature(self, idx: np.ndarray, hour: int, horizon: float, state: np.ndarray, age: np.ndarray) -> np.ndarray:
        elapsed = np.clip(np.floor(age), 0, SURV_HORIZON).astype(np.int32)
        ahead = np.clip(np.floor(age + horizon), 0, SURV_HORIZON).astype(np.int32)
        out = np.zeros(len(idx), dtype=np.float32)
        for code, table in ((1, self.surv_restr), (2, self.surv_closed)):
            mask = state == code
            if not np.any(mask):
                continue
            base = np.maximum(table[idx[mask], elapsed[mask]], 1e-3)
            out[mask] = np.clip(table[idx[mask], ahead[mask]] / base, 0.0, 1.0)
        return out

    def design(
        self,
        hour: int,
        horizon: float,
        idx: np.ndarray,
        now: datetime | None = None,
    ) -> np.ndarray:
        hour = int(np.clip(hour, 0, self.n_hours - 1))
        idx = np.asarray(idx, dtype=np.int32)
        origin = self.grid_start + timedelta(hours=hour)
        target = origin + timedelta(hours=float(horizon))
        count = len(idx)
        msk = target.astimezone(MSK)
        hour_frac = msk.hour + msk.minute / 60.0 + msk.second / 3600.0
        utc = target.astimezone(UTC)
        utc_frac = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
        local = (utc_frac + self.offsets[idx]) % 24.0
        dow = msk.weekday()
        doy = msk.timetuple().tm_yday
        hod = int(msk.hour) % 24
        state = self.state[idx, hour].astype(np.int16)
        age = self.age[idx, hour].astype(np.float32)
        since = self.since_close[idx, hour].astype(np.float32)
        if now is not None:
            state = state.copy()
            age = age.copy()
            since = since.copy()
            for pos, row in enumerate(idx):
                code, exact_age, exact_since, _flag = self.exact(int(row), now)
                state[pos] = code
                age[pos] = exact_age
                since[pos] = exact_since
        day = min(hour // 24, self.prior14.shape[0] - 1)
        cols: dict[str, np.ndarray] = {
            "h": np.full(count, horizon, dtype=np.float32),
            "log1p_h": np.full(count, math.log1p(max(horizon, 0.0)), dtype=np.float32),
            "flag_h_le_6": np.full(count, 1.0 if horizon <= 6 else 0.0, dtype=np.float32),
            "flag_h_le_48": np.full(count, 1.0 if horizon <= 48 else 0.0, dtype=np.float32),
            "flag_h_gt_48": np.full(count, 1.0 if horizon > 48 else 0.0, dtype=np.float32),
            "sin_hour_msk": np.full(count, math.sin(2 * math.pi * hour_frac / 24), dtype=np.float32),
            "cos_hour_msk": np.full(count, math.cos(2 * math.pi * hour_frac / 24), dtype=np.float32),
            "sin_hour_local": np.sin(2 * np.pi * local / 24).astype(np.float32),
            "cos_hour_local": np.cos(2 * np.pi * local / 24).astype(np.float32),
            "sin_dow": np.full(count, math.sin(2 * math.pi * dow / 7), dtype=np.float32),
            "cos_dow": np.full(count, math.cos(2 * math.pi * dow / 7), dtype=np.float32),
            "sin_doy": np.full(count, math.sin(2 * math.pi * doy / 365.25), dtype=np.float32),
            "cos_doy": np.full(count, math.cos(2 * math.pi * doy / 365.25), dtype=np.float32),
            "is_weekend": np.full(count, 1.0 if dow >= 5 else 0.0, dtype=np.float32),
            "is_night": np.full(count, 1.0 if hour_frac < 6 or hour_frac >= 22 else 0.0, dtype=np.float32),
            "cur_restricted": (state == 1).astype(np.float32),
            "cur_closed": (state == 2).astype(np.float32),
            "log1p_age": np.log1p(np.clip(age, 0, 720)).astype(np.float32),
            "surv_cont": self._survival_feature(idx, hour, horizon, state, age),
            "log1p_since_close": np.log1p(np.clip(since, 0, 720)).astype(np.float32),
            "prior14_restr": self.prior14[day, idx, hod, 1],
            "prior14_closed": self.prior14[day, idx, hod, 2],
            "prior60_restr": self.prior60[day, idx, hod, 1],
            "prior60_closed": self.prior60[day, idx, hod, 2],
            "global14_closed": np.full(count, self.global14_closed[day, hod], dtype=np.float32),
        }
        for width in WINDOWS:
            cols[f"closed_frac_{width}"] = self._window(self.ps_closed, idx, hour, width)
            cols[f"restr_frac_{width}"] = self._window(self.ps_restr, idx, hour, width)
        for width in START_WINDOWS:
            left = max(0, hour - width)
            cols[f"close_starts_{width}"] = (self.ps_starts[idx, hour] - self.ps_starts[idx, left]).astype(np.float32)
        for lag in LAGS:
            src = hour - lag
            if src < 0:
                cols[f"lag_closed_{lag}"] = np.zeros(count, dtype=np.float32)
                cols[f"lag_restr_{lag}"] = np.zeros(count, dtype=np.float32)
            else:
                cols[f"lag_closed_{lag}"] = (self.state[idx, src] == 2).astype(np.float32)
                cols[f"lag_restr_{lag}"] = (self.state[idx, src] == 1).astype(np.float32)
        for radius in RADII:
            for width in SPATIAL_W:
                cols[f"nei{radius}_closed_{width}"] = self._neighbor_frac(radius, "closed", idx, hour, width)
                cols[f"nei{radius}_restr_{width}"] = self._neighbor_frac(radius, "restr", idx, hour, width)
        for width in SPATIAL_W:
            cols[f"clu_closed_{width}"] = self._cluster_frac("closed", idx, hour, width)
            cols[f"clu_restr_{width}"] = self._cluster_frac("restr", idx, hour, width)
        missing = [name for name in FEATURE_NAMES if name not in cols]
        if missing:
            raise RuntimeError(f"нет признаков: {missing}")
        matrix = np.column_stack([cols[name] for name in FEATURE_NAMES]).astype(np.float32)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def sample_origins(self, rng: np.random.Generator, cap: int) -> np.ndarray:
        hours = self.n_hours
        min_hour = min(24, max(1, hours // 10))
        max_hour = hours - 1
        if max_hour <= min_hour:
            return np.arange(0, max_hour, dtype=np.int32)
        keep = np.zeros(hours, dtype=bool)
        keep[min_hour:max_hour:12] = True
        changed = np.zeros(hours, dtype=bool)
        changed[1:] = np.any(self.state[:, 1:] != self.state[:, :-1], axis=0)
        dilated = changed.copy()
        for shift in range(1, 4):
            dilated[shift:] |= changed[:-shift]
        keep |= dilated
        active = np.flatnonzero(np.any(self.state != 0, axis=0))
        keep[active[::2]] = True
        origins = np.flatnonzero(keep)
        origins = origins[(origins >= min_hour) & (origins < max_hour)]
        if len(origins) > cap:
            must = origins[dilated[origins]]
            rest = origins[~dilated[origins]]
            if len(must) > cap:
                must = np.sort(rng.choice(must, cap, replace=False))
                rest = np.array([], dtype=int)
            else:
                need = cap - len(must)
                if len(rest) > need:
                    rest = np.sort(rng.choice(rest, need, replace=False))
            origins = np.sort(np.concatenate([must, rest]))
        return origins.astype(np.int32)

    def select_airports(self, hour: int, rng: np.random.Generator) -> np.ndarray:
        left = max(0, hour - 48)
        recent_closed = (self.ps_closed[:, hour] - self.ps_closed[:, left]) > 0
        recent_restr = (self.ps_restr[:, hour] - self.ps_restr[:, left]) > 0
        current = self.state[:, hour] != 0
        neighbor = self._neighbor_frac(500, "closed", np.arange(self.n), hour, 6) > 0
        hot_mask = current | recent_closed | recent_restr | neighbor
        hot = np.flatnonzero(hot_mask)
        if len(hot) > 28:
            hot = rng.choice(hot, 28, replace=False)
        cold = np.flatnonzero(~hot_mask)
        if len(cold):
            picked = np.unique(cold[(np.arange(8) * 13 + int(hour)) % len(cold)])
        else:
            picked = np.array([], dtype=int)
        if len(hot) == 0 and len(picked) == 0:
            return np.arange(min(8, self.n), dtype=np.int32)
        return np.unique(np.concatenate([hot, picked])).astype(np.int32)


def _shrink(local: list[float], global_curve: np.ndarray) -> np.ndarray:
    if not local:
        return global_curve
    own = _survival_curve(local)
    weight = float(len(local))
    return ((weight * own + 8.0 * global_curve) / (weight + 8.0)).astype(np.float32)


def sample_training(
    timeline: Timeline,
    horizons: tuple[int, ...],
    max_rows: int,
    max_origins: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    origins = timeline.sample_origins(rng, max_origins)
    blocks_x: list[np.ndarray] = []
    blocks_y: list[np.ndarray] = []
    blocks_h: list[np.ndarray] = []
    blocks_t: list[np.ndarray] = []
    for origin in origins:
        chosen = timeline.select_airports(int(origin), rng)
        for horizon in horizons:
            target = int(origin) + int(horizon)
            if target >= timeline.n_hours:
                continue
            matrix = timeline.design(int(origin), float(horizon), chosen)
            blocks_x.append(matrix)
            blocks_y.append(timeline.state[chosen, target].astype(np.int32))
            blocks_h.append(np.full(len(chosen), horizon, dtype=np.float32))
            blocks_t.append(np.full(len(chosen), target, dtype=np.int32))
    if not blocks_x:
        raise RuntimeError("нет строк для обучения: в базе слишком короткая история")
    matrix = np.vstack(blocks_x).astype(np.float32)
    target = np.concatenate(blocks_y).astype(np.int32)
    horizon_arr = np.concatenate(blocks_h).astype(np.float32)
    label_time = np.concatenate(blocks_t).astype(np.int32)
    return _subsample(matrix, target, horizon_arr, label_time, max_rows, rng)


def _subsample(
    matrix: np.ndarray,
    target: np.ndarray,
    horizon: np.ndarray,
    label_time: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(target) <= max_rows:
        return matrix, target, horizon, label_time
    rare = np.flatnonzero(target != 0)
    common = np.flatnonzero(target == 0)
    keep_rare = min(len(rare), int(max_rows * 0.55))
    if len(rare) > keep_rare:
        rare = rng.choice(rare, keep_rare, replace=False)
    keep_common = max_rows - len(rare)
    if len(common) > keep_common:
        weights = label_time[common].astype(np.float64)
        weights = weights - weights.min() + 1.0
        weights /= weights.sum()
        common = rng.choice(common, keep_common, replace=False, p=weights)
    picked = np.concatenate([rare, common])
    rng.shuffle(picked)
    return matrix[picked], target[picked], horizon[picked], label_time[picked]


def run_self_tests() -> None:
    start = datetime(2024, 1, 1, 1, 30, tzinfo=UTC)
    end = datetime(2024, 1, 1, 3, 0, tzinfo=UTC)
    intervals = [
        Interval("UUWW", CLOSED, start, end, False, 1, 2),
    ]
    moment = datetime(2024, 1, 3, 0, 0, tzinfo=UTC)
    timeline = Timeline(intervals, moment)
    row = timeline.index["UUWW"]
    # Точки сетки: 01:00 открыт, 02:00 закрыт, 03:00 уже открыт.
    base = timeline.hour_index(datetime(2024, 1, 1, 0, 0, tzinfo=UTC))
    assert timeline.state[row, base + 1] == 0
    assert timeline.state[row, base + 2] == 2
    assert timeline.state[row, base + 3] == 0
    assert abs(float(timeline.age[row, base + 2]) - 0.5) < 1e-3
    hour = base + 4
    frac = timeline._window(timeline.ps_closed, np.array([row]), hour, 6)
    # В окне из 6 часов ровно один час закрытия.
    assert abs(float(frac[0]) - 1.0 / 6.0) < 1e-4
    neighbor = timeline.index["UUDD"]
    nei = timeline._neighbor_frac(250, "closed", np.array([neighbor]), hour, 6)
    assert float(nei[0]) > 0.0
    matrix = timeline.design(hour, 2.5, np.array([row, neighbor]))
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()
    code, age, _since, _flag = timeline.exact(row, datetime(2024, 1, 1, 2, 0, tzinfo=UTC))
    assert code == 2 and abs(age - 0.5) < 1e-6
    code, _age, _since, _flag = timeline.exact(row, datetime(2024, 1, 1, 4, 0, tzinfo=UTC))
    assert code == 0


if __name__ == "__main__":
    run_self_tests()
    print("features ok", len(FEATURE_NAMES))
