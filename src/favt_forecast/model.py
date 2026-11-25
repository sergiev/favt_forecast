"""CatBoost на двух горизонтах, калибровка и пространственно-временной прайор."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

from favt_forecast.airports import AIRPORTS, BY_ICAO
from favt_forecast.config import (
    BLEND_HI,
    BLEND_LO,
    FAR_HORIZONS,
    HORIZON_HOURS,
    MAX_ORIGINS,
    MAX_TRAIN_ROWS,
    MODEL_DIR,
    NEAR_HORIZONS,
    ensure_dirs,
)
from favt_forecast.db import connect, counts, insert_model_run, load_intervals
from favt_forecast.features import (
    FEATURE_INDEX,
    FEATURE_NAMES,
    Timeline,
    explain_row,
    sample_training,
)
from favt_forecast.parser import CLOSED, Interval, OPEN, RESTRICTED
from favt_forecast.util import MSK, floor_hour, iso, now_utc, parse_dt

CLASSES = (OPEN, RESTRICTED, CLOSED)
BUNDLE_PATH = MODEL_DIR / "bundle.joblib"


def _intervals_from_db(conn) -> list[Interval]:
    rows = []
    for row in load_intervals(conn):
        rows.append(
            Interval(
                icao=row["icao"],
                state=row["state"],
                start=parse_dt(row["start_utc"]),
                end=parse_dt(row["end_utc"]) if row["end_utc"] else None,
                inferred_end=bool(row["inferred_end"]),
                start_msg_id=row["start_msg_id"],
                end_msg_id=row["end_msg_id"],
            )
        )
    return rows


def heuristic(matrix: np.ndarray) -> np.ndarray:
    closed = np.clip(matrix[:, FEATURE_INDEX["prior14_closed"]], 0, 1)
    restricted = np.clip(matrix[:, FEATURE_INDEX["prior14_restr"]], 0, 1)
    survival = np.clip(matrix[:, FEATURE_INDEX["surv_cont"]], 0, 1)
    current_closed = matrix[:, FEATURE_INDEX["cur_closed"]] >= 0.5
    current_restricted = matrix[:, FEATURE_INDEX["cur_restricted"]] >= 0.5
    opened = np.clip(1.0 - closed - restricted, 0.01, None)
    proba = np.stack([opened, restricted, closed], axis=1)
    proba /= proba.sum(axis=1, keepdims=True)
    if np.any(current_closed):
        stay = survival[current_closed]
        rest = 1.0 - stay
        base = proba[current_closed, :2]
        share = base / np.clip(base.sum(axis=1, keepdims=True), 1e-6, None)
        proba[current_closed, 0] = rest * share[:, 0]
        proba[current_closed, 1] = rest * share[:, 1]
        proba[current_closed, 2] = stay
    restricted_now = current_restricted & ~current_closed
    if np.any(restricted_now):
        stay = survival[restricted_now]
        rest = 1.0 - stay
        proba[restricted_now, 1] = stay
        proba[restricted_now, 0] = rest * 0.8
        proba[restricted_now, 2] = rest * 0.2
    proba = np.clip(proba, 1e-6, None)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba.astype(np.float32)


def _nll(target: np.ndarray, proba: np.ndarray) -> float:
    clipped = np.clip(proba, 1e-6, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(clipped[np.arange(len(target)), target])))


def _brier(target: np.ndarray, proba: np.ndarray) -> float:
    eye = np.eye(3, dtype=np.float64)[target]
    return float(np.mean(np.sum((proba - eye) ** 2, axis=1)))


def _brier_class(target: np.ndarray, proba: np.ndarray, cls: int) -> float:
    truth = (target == cls).astype(np.float64)
    return float(np.mean((proba[:, cls] - truth) ** 2))


def _apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-6, 1.0)) / max(temperature, 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def _fit_temperature(target: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    best_t = 1.0
    best = _nll(target, proba)
    for temperature in np.linspace(0.65, 2.4, 24):
        score = _nll(target, _apply_temperature(proba, float(temperature)))
        if score < best:
            best = score
            best_t = float(temperature)
    return best_t, best


def _fit_isotonic(target: np.ndarray, proba: np.ndarray) -> list[IsotonicRegression | None]:
    models: list[IsotonicRegression | None] = []
    for cls in range(3):
        column = proba[:, cls]
        if len(np.unique(column)) < 3 or int((target == cls).sum()) < 20:
            models.append(None)
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(column, (target == cls).astype(np.float64))
        models.append(model)
    return models


def _apply_isotonic(proba: np.ndarray, models: list[IsotonicRegression | None]) -> np.ndarray:
    out = proba.copy()
    for cls, model in enumerate(models):
        if model is not None:
            out[:, cls] = model.predict(proba[:, cls])
    out = np.clip(out, 1e-6, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _best_alpha(target: np.ndarray, model_p: np.ndarray, prior_p: np.ndarray) -> float:
    best_a = 1.0
    best = _brier(target, model_p)
    for alpha in np.linspace(0.0, 1.0, 21):
        mixed = alpha * model_p + (1.0 - alpha) * prior_p
        mixed /= mixed.sum(axis=1, keepdims=True)
        score = _brier(target, mixed)
        if score < best - 1e-9:
            best = score
            best_a = float(alpha)
    return best_a


def _align(model: CatBoostClassifier, proba: np.ndarray) -> np.ndarray:
    full = np.zeros((len(proba), 3), dtype=np.float64)
    for column, cls in enumerate(model.classes_):
        full[:, int(cls)] = proba[:, column]
    total = full.sum(axis=1, keepdims=True)
    total[total <= 0] = 1.0
    return full / total


def _reliability(target: np.ndarray, proba: np.ndarray, cls: int = 2, bins: int = 8) -> list[dict]:
    column = proba[:, cls]
    truth = (target == cls).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        lo, hi = float(edges[index]), float(edges[index + 1])
        if index == bins - 1:
            mask = (column >= lo) & (column <= hi)
        else:
            mask = (column >= lo) & (column < hi)
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(mask.sum()),
                "p_mean": float(column[mask].mean()),
                "y_rate": float(truth[mask].mean()),
            }
        )
    return rows


def _detect_device() -> str:
    forced = os.environ.get("FAVT_DEVICE", "auto").strip().upper()
    if forced in {"CPU", "GPU"}:
        return forced
    try:
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(24, 6)).astype(np.float32)
        target = np.array([0, 1, 2] * 8, dtype=np.int32)
        probe = CatBoostClassifier(
            iterations=1,
            depth=2,
            loss_function="MultiClass",
            task_type="GPU",
            devices="0",
            allow_writing_files=False,
            verbose=False,
        )
        probe.fit(matrix, target)
        return "GPU"
    except Exception:
        return "CPU"


def _fit(matrix, target, weight, valid_x, valid_y, device: str) -> tuple[CatBoostClassifier, str]:
    params = {
        "loss_function": "MultiClass",
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 8.0,
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
    }
    use_valid = (
        valid_x is not None
        and len(valid_x) >= 40
        and len(np.unique(valid_y)) >= 2
        and len(np.unique(target)) >= 2
    )
    if use_valid:
        params["early_stopping_rounds"] = 40

    def once(task: str) -> CatBoostClassifier:
        extra = (
            {"task_type": "GPU", "devices": "0"}
            if task == "GPU"
            else {"task_type": "CPU", "thread_count": max(1, (os.cpu_count() or 2) - 1)}
        )
        model = CatBoostClassifier(**params, **extra)
        if use_valid:
            model.fit(matrix, target, sample_weight=weight, eval_set=(valid_x, valid_y))
        else:
            model.fit(matrix, target, sample_weight=weight)
        return model

    if device == "GPU":
        try:
            return once("GPU"), "GPU"
        except Exception as exc:
            print(f"GPU-обучение не стартовало ({exc.__class__.__name__}: {exc}). Повтор на CPU.")
            return once("CPU"), "CPU"
    return once("CPU"), "CPU"


def _weights(target: np.ndarray) -> np.ndarray:
    weight = np.ones(len(target), dtype=np.float32)
    weight[target == 1] = 4.0
    weight[target == 2] = 5.0
    return weight


def _split(label_time: np.ndarray, n_hours: int) -> tuple[np.ndarray, int]:
    cutoff = n_hours - 14 * 24
    if cutoff < int(n_hours * 0.55):
        cutoff = int(n_hours * 0.8)
    train = label_time < cutoff
    if train.sum() < 100 or (~train).sum() < 30:
        cutoff = int(np.quantile(label_time, 0.8))
        train = label_time < cutoff
    return train, int(cutoff)


def _calibrate(model: CatBoostClassifier | None, matrix_v, target_v) -> dict:
    if model is None or matrix_v is None or len(target_v) < 30:
        return {"has_model": model is not None, "use_temp": False, "temperature": 1.0, "use_iso": False, "isotonic": None, "alpha": 0.0 if model is None else 1.0}
    raw = _align(model, model.predict_proba(matrix_v))
    temperature, nll_temp = _fit_temperature(target_v, raw)
    tempered = _apply_temperature(raw, temperature)
    isotonic = _fit_isotonic(target_v, tempered)
    calibrated = _apply_isotonic(tempered, isotonic) if any(item is not None for item in isotonic) else tempered
    candidates = {
        "raw": (_nll(target_v, raw), raw, False, False),
        "temp": (nll_temp, tempered, True, False),
        "iso": (_nll(target_v, calibrated), calibrated, True, True),
    }
    best_name = min(candidates, key=lambda name: candidates[name][0])
    _score, chosen, use_temp, use_iso = candidates[best_name]
    if not use_iso:
        isotonic = None
    if not use_temp:
        temperature = 1.0
    alpha = _best_alpha(target_v, chosen, heuristic(matrix_v))
    return {
        "has_model": True,
        "use_temp": use_temp,
        "temperature": temperature,
        "use_iso": use_iso,
        "isotonic": isotonic,
        "alpha": alpha,
    }


def _apply_calibrator(model: CatBoostClassifier | None, calibrator: dict, matrix: np.ndarray) -> np.ndarray:
    if model is None or not calibrator.get("has_model"):
        return heuristic(matrix)
    proba = _align(model, model.predict_proba(matrix))
    if calibrator.get("use_temp"):
        proba = _apply_temperature(proba, float(calibrator["temperature"]))
    if calibrator.get("use_iso") and calibrator.get("isotonic"):
        proba = _apply_isotonic(proba, calibrator["isotonic"])
    alpha = float(calibrator.get("alpha", 1.0))
    if alpha < 0.999:
        proba = alpha * proba + (1.0 - alpha) * heuristic(matrix)
    proba = np.clip(proba, 1e-6, None)
    proba /= proba.sum(axis=1, keepdims=True)
    return proba.astype(np.float32)


class Predictor:
    def __init__(self, timeline: Timeline, models: dict, bundle: dict):
        self.timeline = timeline
        self.models = models
        self.bundle = bundle
        self._cube_key: str | None = None
        self._cube: dict | None = None

    def predict_proba(self, matrix: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        horizons = np.asarray(horizons, dtype=np.float64)
        out = np.zeros((len(horizons), 3), dtype=np.float32)
        near = horizons <= BLEND_LO
        far = horizons >= BLEND_HI
        blend = ~near & ~far
        if np.any(near):
            out[near] = _apply_calibrator(self.models.get("near"), self.bundle["near"], matrix[near])
        if np.any(far):
            out[far] = _apply_calibrator(self.models.get("far"), self.bundle["far"], matrix[far])
        if np.any(blend):
            p_near = _apply_calibrator(self.models.get("near"), self.bundle["near"], matrix[blend])
            p_far = _apply_calibrator(self.models.get("far"), self.bundle["far"], matrix[blend])
            weight = ((BLEND_HI - horizons[blend]) / (BLEND_HI - BLEND_LO)).astype(np.float32)
            mixed = weight[:, None] * p_near + (1.0 - weight[:, None]) * p_far
            mixed /= mixed.sum(axis=1, keepdims=True)
            out[blend] = mixed
        return out

    def _origin(self, moment: datetime) -> tuple[int, datetime]:
        hour = self.timeline.hour_index(floor_hour(moment))
        hour = int(np.clip(hour, 0, self.timeline.n_hours - 1))
        return hour, self.timeline.grid_start + timedelta(hours=hour)

    def cube(self, horizon: int = HORIZON_HOURS) -> dict:
        now = now_utc()
        key = f"{int(now.timestamp()) // 60}:{horizon}"
        if self._cube_key == key and self._cube is not None:
            return self._cube
        origin_hour, origin = self._origin(now)
        start = floor_hour(now) + timedelta(hours=1)
        if start <= origin:
            start = origin + timedelta(hours=1)
        targets = [start + timedelta(hours=step) for step in range(horizon)]
        horizons = np.array([(item - origin).total_seconds() / 3600.0 for item in targets], dtype=np.float64)
        idx = np.arange(self.timeline.n, dtype=np.int32)
        blocks = [self.timeline.design(origin_hour, float(value), idx, now=now) for value in horizons]
        matrix = np.vstack(blocks)
        horizon_per_row = np.repeat(horizons.astype(np.float32), self.timeline.n)
        proba = self.predict_proba(matrix, horizon_per_row)
        cube = proba.reshape(horizon, self.timeline.n, 3).transpose(1, 0, 2)
        self._cube = {
            "now": now,
            "origin": origin,
            "targets": targets,
            "cube": cube,
        }
        self._cube_key = key
        return self._cube

    def forecast(self, icao: str, horizon: int = HORIZON_HOURS) -> dict:
        started = time.perf_counter()
        now = now_utc()
        origin_hour, origin = self._origin(now)
        start = floor_hour(now) + timedelta(hours=1)
        if start <= origin:
            start = origin + timedelta(hours=1)
        targets = [start + timedelta(hours=step) for step in range(horizon)]
        horizons = np.array([(item - origin).total_seconds() / 3600.0 for item in targets], dtype=np.float64)
        row = self.timeline.index[icao]
        blocks = [
            self.timeline.design(origin_hour, float(value), np.array([row], dtype=np.int32), now=now)
            for value in horizons
        ]
        proba = self.predict_proba(np.vstack(blocks), horizons)
        points = []
        for moment, probs in zip(targets, proba):
            msk = moment.astimezone(MSK)
            points.append(
                {
                    "ts": iso(moment),
                    "hour_msk": int(msk.hour),
                    "dow_msk": int(msk.weekday()),
                    "p_open": float(probs[0]),
                    "p_restricted": float(probs[1]),
                    "p_closed": float(probs[2]),
                    "observed": False,
                    "inferred": False,
                }
            )
        history = self._observed_hours(row, targets[0] - timedelta(hours=24 * 7), targets[0])
        code, _age, _since, inferred = self.timeline.exact(row, now)
        return {
            "airport": BY_ICAO[icao].public(),
            "issued_at": iso(now),
            "origin_ts": iso(origin),
            "current_state": CLASSES[code],
            "current_inferred": inferred,
            "inference_ms": (time.perf_counter() - started) * 1000.0,
            "history": history,
            "points": points,
        }

    def _observed_hours(self, row: int, start: datetime, stop: datetime) -> list[dict]:
        points = []
        moment = start
        while moment < stop:
            code, _age, _since, inferred = self.timeline.exact(row, moment)
            probs = [0.0, 0.0, 0.0]
            probs[code] = 1.0
            msk = moment.astimezone(MSK)
            points.append(
                {
                    "ts": iso(moment),
                    "hour_msk": int(msk.hour),
                    "dow_msk": int(msk.weekday()),
                    "p_open": probs[0],
                    "p_restricted": probs[1],
                    "p_closed": probs[2],
                    "observed": True,
                    "inferred": bool(inferred),
                }
            )
            moment += timedelta(hours=1)
        return points

    def predict_point(self, icao: str, target: datetime) -> dict:
        started = time.perf_counter()
        row = self.timeline.index[icao]
        now = now_utc()
        code_now, _age, _since, inferred_now = self.timeline.exact(row, now)
        if target <= now:
            code, _age, _since, inferred = self.timeline.exact(row, target)
            proba = np.zeros(3, dtype=np.float64)
            proba[code] = 1.0
            payload = _point_payload(icao, target, now, now, 0.0, proba, "observed", inferred, [])
            payload["current_state"] = CLASSES[code_now]
            payload["current_inferred"] = inferred_now
            payload["inference_ms"] = (time.perf_counter() - started) * 1000.0
            return payload
        origin_hour, origin = self._origin(now)
        horizon = (target - origin).total_seconds() / 3600.0
        matrix = self.timeline.design(origin_hour, horizon, np.array([row], dtype=np.int32), now=now)
        proba = self.predict_proba(matrix, np.array([horizon]))[0]
        payload = _point_payload(
            icao,
            target,
            now,
            origin,
            horizon,
            proba,
            "model",
            False,
            explain_row(matrix[0]),
        )
        payload["current_state"] = CLASSES[code_now]
        payload["current_inferred"] = inferred_now
        payload["beyond_horizon"] = horizon > HORIZON_HOURS
        payload["inference_ms"] = (time.perf_counter() - started) * 1000.0
        return payload

    def summaries(self) -> list[dict]:
        grid = self.cube(HORIZON_HOURS)
        out = []
        hours_24 = min(24, grid["cube"].shape[1])
        for airport in AIRPORTS:
            row = self.timeline.index[airport.icao]
            series = grid["cube"][row]
            code, _age, _since, inferred = self.timeline.exact(row, grid["now"])
            out.append(
                {
                    **airport.public(),
                    "current_state": CLASSES[code],
                    "current_inferred": inferred,
                    "p_open_24h": float(series[:hours_24, 0].mean()),
                    "p_restricted_24h": float(series[:hours_24, 1].mean()),
                    "p_closed_24h": float(series[:hours_24, 2].mean()),
                    "peak_p_closed_24h": float(series[:hours_24, 2].max()),
                    "p_closed_14d": float(series[:, 2].mean()),
                    "peak_p_closed_14d": float(series[:, 2].max()),
                }
            )
        return out


def _point_payload(icao, target, now, origin, horizon, proba, source, inferred, factors) -> dict:
    return {
        "airport": BY_ICAO[icao].public(),
        "target_ts": iso(target),
        "issued_at": iso(now),
        "origin_ts": iso(origin),
        "horizon_hours": float(horizon),
        "source": source,
        "inferred": bool(inferred),
        "probabilities": {
            OPEN: float(proba[0]),
            RESTRICTED: float(proba[1]),
            CLOSED: float(proba[2]),
        },
        "factors": factors,
    }


def _train_branch(name: str, timeline: Timeline, horizons: tuple[int, ...], device: str) -> tuple[CatBoostClassifier | None, dict, dict]:
    print(f"сэмпл {name}: горизонты {horizons[0]}..{horizons[-1]} ч")
    matrix, target, _horizon, label_time = sample_training(
        timeline, horizons, MAX_TRAIN_ROWS, MAX_ORIGINS, seed=42 if name == "near" else 43
    )
    train_mask, cutoff = _split(label_time, timeline.n_hours)
    valid_mask = ~train_mask
    info = {
        "rows": int(len(target)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(valid_mask.sum()),
        "cutoff_hour": cutoff,
        "class_counts": {CLASSES[c]: int((target == c).sum()) for c in range(3)},
    }
    print(
        f"  строк {info['rows']}: train {info['train_rows']}, val {info['val_rows']}, "
        f"классы {info['class_counts']}"
    )
    if len(np.unique(target[train_mask])) < 2:
        print(f"  {name}: меньше двух классов, остаётся прайор")
        return None, {"has_model": False, "use_temp": False, "temperature": 1.0, "use_iso": False, "isotonic": None, "alpha": 0.0}, info
    model, used = _fit(
        matrix[train_mask],
        target[train_mask],
        _weights(target[train_mask]),
        matrix[valid_mask] if valid_mask.any() else None,
        target[valid_mask] if valid_mask.any() else None,
        device,
    )
    info["device"] = used
    calibrator = _calibrate(
        model,
        matrix[valid_mask] if valid_mask.any() else None,
        target[valid_mask] if valid_mask.any() else None,
    )
    info["alpha"] = calibrator["alpha"]
    info["temperature"] = calibrator["temperature"]
    if valid_mask.any():
        proba = _apply_calibrator(model, calibrator, matrix[valid_mask])
        raw = _align(model, model.predict_proba(matrix[valid_mask]))
        prior = heuristic(matrix[valid_mask])
        yv = target[valid_mask]
        info["metrics"] = {
            "logloss": _nll(yv, proba),
            "brier": _brier(yv, proba),
            "brier_closed": _brier_class(yv, proba, 2),
            "brier_restricted": _brier_class(yv, proba, 1),
            "brier_model": _brier(yv, raw),
            "brier_prior": _brier(yv, prior),
            "base_closed": float((yv == 2).mean()),
            "base_restricted": float((yv == 1).mean()),
            "reliability_closed": _reliability(yv, proba, 2),
        }
        print(
            f"  val logloss {info['metrics']['logloss']:.4f}, "
            f"brier {info['metrics']['brier']:.4f}, "
            f"brier closed {info['metrics']['brier_closed']:.4f} "
            f"(прайор {info['metrics']['brier_prior']:.4f}), alpha {calibrator['alpha']:.2f}"
        )
    return model, calibrator, info


def train() -> dict:
    ensure_dirs()
    started = time.perf_counter()
    conn = connect()
    try:
        intervals = _intervals_from_db(conn)
        moment = now_utc()
        if len(intervals) < 30:
            raise RuntimeError(
                f"мало интервалов для обучения: {len(intervals)}. Сначала нужен долив ленты."
            )
        print(f"интервалов {len(intervals)}, строю сетку…")
        timeline = Timeline(intervals, moment)
        print(f"сетка {timeline.n_hours} ч с {iso(timeline.grid_start)}")
        device = _detect_device()
        print(f"устройство: {device}")
        near_model, near_cal, near_info = _train_branch("near", timeline, NEAR_HORIZONS, device)
        far_model, far_cal, far_info = _train_branch("far", timeline, FAR_HORIZONS, device)
        used = near_info.get("device") or far_info.get("device") or device
        bundle = {
            "near": near_cal,
            "far": far_cal,
            "feature_names": list(FEATURE_NAMES),
            "icaos": [airport.icao for airport in AIRPORTS],
            "trained_at": iso(moment),
            "device": used,
        }
        _store_models(near_model, far_model, bundle)
        predictor = Predictor(timeline, {"near": near_model, "far": far_model}, bundle)
        t0 = time.perf_counter()
        predictor.predict_point(AIRPORTS[0].icao, moment + timedelta(hours=6))
        point_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        predictor.forecast(AIRPORTS[0].icao, HORIZON_HOURS)
        horizon_ms = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        predictor.summaries()
        network_ms = (time.perf_counter() - t2) * 1000.0
        metrics = {
            "device": used,
            "trained_at": bundle["trained_at"],
            "train_seconds": time.perf_counter() - started,
            "near": {key: value for key, value in near_info.items() if key != "device"},
            "far": {key: value for key, value in far_info.items() if key != "device"},
            "inference_ms_point": point_ms,
            "inference_ms_horizon": horizon_ms,
            "inference_ms_network": network_ms,
            "grid_hours": timeline.n_hours,
            "intervals": len(intervals),
        }
        bundle["metrics"] = metrics
        _store_models(near_model, far_model, bundle)
        stats = counts(conn)
        insert_model_run(
            conn,
            trained_at=bundle["trained_at"],
            device=used,
            metrics_json=json.dumps(metrics, ensure_ascii=False),
            n_events=int(stats["events"]),
            n_intervals=len(intervals),
            train_seconds=float(metrics["train_seconds"]),
        )
        conn.commit()
        print(
            f"обучение {metrics['train_seconds']:.1f} с, "
            f"точка {point_ms:.0f} мс, горизонт 14д {horizon_ms:.0f} мс, "
            f"вся сеть {network_ms:.0f} мс"
        )
        return metrics
    finally:
        conn.close()


def _store_models(near, far, bundle: dict) -> None:
    ensure_dirs()
    for name, model in (("near", near), ("far", far)):
        path = MODEL_DIR / f"{name}.cbm"
        if model is None:
            if path.exists():
                path.unlink()
            continue
        temporary = path.with_suffix(".cbm.tmp")
        model.save_model(str(temporary))
        os.replace(temporary, path)
    temporary = BUNDLE_PATH.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temporary)
    os.replace(temporary, BUNDLE_PATH)


def load_predictor() -> Predictor:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("модель не обучена")
    bundle = joblib.load(BUNDLE_PATH)
    if bundle.get("icaos") != [airport.icao for airport in AIRPORTS]:
        raise RuntimeError("справочник аэропортов изменился, нужно переобучить модель")
    if bundle.get("feature_names") != list(FEATURE_NAMES):
        raise RuntimeError("набор признаков изменился, нужно переобучить модель")
    conn = connect()
    try:
        timeline = Timeline(_intervals_from_db(conn), now_utc())
    finally:
        conn.close()
    models = {}
    for name in ("near", "far"):
        path = MODEL_DIR / f"{name}.cbm"
        if bundle[name].get("has_model") and path.exists():
            model = CatBoostClassifier()
            model.load_model(str(path))
            models[name] = model
    return Predictor(timeline, models, bundle)


def model_ready() -> bool:
    return BUNDLE_PATH.exists()


def saved_metrics() -> dict | None:
    if not BUNDLE_PATH.exists():
        return None
    bundle = joblib.load(BUNDLE_PATH)
    return bundle.get("metrics")


def format_metrics(metrics: dict) -> str:
    lines = [
        f"устройство: {metrics.get('device')}",
        f"обучено: {metrics.get('trained_at')}",
        f"обучение, с: {metrics.get('train_seconds', 0):.1f}",
        f"инференс точки, мс: {metrics.get('inference_ms_point', 0):.1f}",
        f"инференс 14 суток одного аэропорта, мс: {metrics.get('inference_ms_horizon', 0):.1f}",
        f"инференс сети, мс: {metrics.get('inference_ms_network', 0):.1f}",
    ]
    for name in ("near", "far"):
        block = metrics.get(name) or {}
        part = block.get("metrics") or {}
        if not part:
            lines.append(f"{name}: нет валидации")
            continue
        lines.append(
            f"{name}: logloss {part['logloss']:.4f}, brier {part['brier']:.4f}, "
            f"brier closed {part['brier_closed']:.4f}, prior {part['brier_prior']:.4f}, "
            f"base closed {part['base_closed']:.4f}"
        )
        lines.append("  калибровка P(CLOSED):")
        for row in part.get("reliability_closed") or []:
            lines.append(
                f"    [{row['lo']:.2f}, {row['hi']:.2f}) n={row['n']} "
                f"p={row['p_mean']:.3f} факт={row['y_rate']:.3f}"
            )
    return "\n".join(lines)
