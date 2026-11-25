"""Команды: sync, train, predict, backtest, serve, selftest."""
from __future__ import annotations

import argparse
import sys

from favt_forecast.airports import resolve
from favt_forecast.model import format_metrics, load_predictor, model_ready, saved_metrics, train
from favt_forecast.service import ingest_and_update
from favt_forecast.util import parse_dt


def _airport(value: str) -> str:
    airport = resolve(value)
    if airport is None:
        raise SystemExit(f"неизвестный аэропорт: {value}")
    return airport.icao


def cmd_sync(args: argparse.Namespace) -> None:
    ingest_and_update(
        continue_backfill=not args.new_only,
        retrain=not args.no_train,
        offline=args.offline,
    )


def cmd_train(_args: argparse.Namespace) -> None:
    metrics = train()
    print(format_metrics(metrics))


def cmd_predict(args: argparse.Namespace) -> None:
    if not model_ready():
        raise SystemExit("модель не обучена: uv run favt-forecast sync")
    predictor = load_predictor()
    icao = _airport(args.airport)
    if args.at:
        result = predictor.predict_point(icao, parse_dt(args.at))
        probs = result["probabilities"]
        print(f"{result['airport']['label']} ({icao})")
        print(f"момент {result['target_ts']}  горизонт {result['horizon_hours']:.2f} ч  источник {result['source']}")
        print(
            f"P(открыт) {probs['OPEN']:.3f}  "
            f"P(по согласованию) {probs['RESTRICTED']:.3f}  "
            f"P(закрыт) {probs['CLOSED']:.3f}"
        )
        if result.get("inferred"):
            print("интервал закрыт по таймауту 24 ч: публикация о снятии не найдена")
        for factor in result.get("factors") or []:
            print(f"  {factor['label']}: {factor['display']}")
        print(f"инференс {result['inference_ms']:.1f} мс")
        return
    forecast = predictor.forecast(icao, args.horizon)
    print(f"{forecast['airport']['label']} ({icao}) сейчас {forecast['current_state']}")
    print(f"опора {forecast['origin_ts']}  расчёт {forecast['inference_ms']:.1f} мс")
    for index, point in enumerate(forecast["points"]):
        if index >= 24 and (index - 24) % 6 != 0:
            continue
        print(
            f"{point['ts']}  "
            f"откр {point['p_open']:.3f}  "
            f"согл {point['p_restricted']:.3f}  "
            f"закр {point['p_closed']:.3f}"
        )


def cmd_backtest(args: argparse.Namespace) -> None:
    if args.refit or not model_ready():
        train()
    metrics = saved_metrics()
    if not metrics:
        raise SystemExit("нет сохранённых метрик")
    print(format_metrics(metrics))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("favt_forecast.server:app", host=args.host, port=args.port, reload=False)


def cmd_selftest(_args: argparse.Namespace) -> None:
    from favt_forecast.features import run_self_tests as features_ok
    from favt_forecast.parser import run_self_tests as parser_ok
    from favt_forecast.scraper import run_self_tests as scraper_ok

    parser_ok()
    scraper_ok()
    features_ok()
    print("selftest ok")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(prog="uv run favt-forecast")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sync = sub.add_parser("sync", help="долить t.me/s/favt_info, разобрать и дообучить")
    sync.add_argument("--new-only", action="store_true", help="не продолжать исторический бэкфилл")
    sync.add_argument("--offline", action="store_true", help="только переразобрать уже скачанное и дообучить")
    sync.add_argument("--no-train", action="store_true")
    sync.set_defaults(func=cmd_sync)

    train_cmd = sub.add_parser("train", help="обучить модель на текущей базе")
    train_cmd.set_defaults(func=cmd_train)

    predict = sub.add_parser("predict", help="вероятности OPEN / RESTRICTED / CLOSED")
    predict.add_argument("--airport", required=True, help="ICAO, IATA или название")
    predict.add_argument("--at", help="момент Y, ISO; без зоны считается МСК")
    predict.add_argument("--horizon", type=int, default=336)
    predict.set_defaults(func=cmd_predict)

    backtest = sub.add_parser("backtest", help="показать валидацию на последних 14 сутках меток")
    backtest.add_argument("--refit", action="store_true")
    backtest.set_defaults(func=cmd_backtest)

    serve = sub.add_parser("serve", help="HTTP API и веб-интерфейс")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.set_defaults(func=cmd_serve)

    test = sub.add_parser("selftest")
    test.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
