"""Разбор постов @favt_info в события и интервалы состояний аэропортов."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from favt_forecast.airports import BY_ALIAS, BY_ICAO
from favt_forecast.config import INFER_GAP_HOURS
from favt_forecast.util import MSK, UTC, iso

OPEN = "OPEN"
RESTRICTED = "RESTRICTED"
CLOSED = "CLOSED"
STATES = (OPEN, RESTRICTED, CLOSED)

_CYR_TO_LAT = str.maketrans(
    {
        "а": "a",
        "в": "b",
        "е": "e",
        "к": "k",
        "м": "m",
        "н": "h",
        "о": "o",
        "р": "p",
        "с": "c",
        "т": "t",
        "у": "y",
        "х": "x",
    }
)

_OPEN_RE = re.compile(r"(?<![а-я])(?<!не\s)снят[аоы]?(?![а-я])")
_CLOSED_VERB_RE = re.compile(r"(?<![а-я])введен[аоы]?(?![а-я])")
_LIMIT_RE = re.compile(r"ограничен")
_OPS_RE = re.compile(r"прием|выпуск|работ")
_NOT_OPERATING_RE = re.compile(r"не\s+принима\w*\s+и\s+не\s+отправ")
_RESTR_RE = re.compile(r"по\s+согласованию")
_RECEIVE_RE = re.compile(r"принима|прием|отправ|обслужива|рейс|выпуск")
_NARRATIVE_RE = re.compile(
    r"возможн|корректир|расслед|запасн|пассажир|погиб|памятн|истори|ушел|ушли|ушло"
)
_SKIP_SENTENCE_RE = re.compile(
    r"[^.!?]*(?:запасн|корректир|возможн|расслед|пассажир|погиб)[^.!?]*[.!?]?"
)
_RANGE_RE = re.compile(
    r"\d{1,2}[:.]\d{2}\s*(?:мск|msk)?\s*до\s*\d{1,2}[:.]\d{2}(?:\s*(?:мск|msk))?",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    r"(?<![0-9])(\d{1,2})[:.](\d{2})\s*(?:мск|msk)(?![а-яa-z])",
    re.IGNORECASE,
)
_CODE_RE = re.compile(r"(?<![0-9a-zа-я])([a-zа-я]{4})(?![0-9a-zа-я])")


def _compile_aliases() -> tuple[re.Pattern[str], dict[str, str]]:
    items = sorted(BY_ALIAS.items(), key=lambda item: len(item[0]), reverse=True)
    mapping: dict[str, str] = {}
    parts: list[str] = []
    for alias, airport in items:
        if alias in mapping:
            continue
        mapping[alias] = airport.icao
        parts.append(re.escape(alias).replace(r"\ ", r"\s+"))
    pattern = re.compile(
        r"(?<![0-9a-zа-я])(" + "|".join(parts) + r")(?![0-9a-zа-я])"
    )
    return pattern, mapping


_ALIAS_RE, _ALIAS_MAP = _compile_aliases()
_ICAO_LOWER = {icao.lower(): icao for icao in BY_ICAO}


@dataclass(frozen=True)
class Event:
    msg_id: int
    icao: str
    event_type: str
    event_time: datetime
    posted_at: datetime
    time_source: str
    snippet: str


@dataclass(frozen=True)
class Interval:
    icao: str
    state: str
    start: datetime
    end: datetime | None
    inferred_end: bool
    start_msg_id: int | None
    end_msg_id: int | None


def norm_paragraph(text: str) -> str:
    value = text.lower().replace("ё", "е")
    value = re.sub(r"[-–—]", " ", value)
    value = re.sub(r"[^\w\s:.+]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_paragraphs(text: str) -> list[str]:
    value = text.replace("\xa0", " ").replace("\u200b", "")
    value = value.replace("ё", "е").replace("Ё", "Е")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    if not value:
        return []
    lines: list[str] = []
    for block in re.split(r"\n\s*\n", value):
        for line in block.split("\n"):
            cleaned = re.sub(r"^[\s\-–—•·]+", "", line).strip()
            if cleaned:
                lines.append(cleaned)
    return lines


def classify(text: str) -> str | None:
    # Снятие ограничений на трассах и в зоне, без аэропорта: это не статус аэродрома.
    if re.search(r"маршрутах воздушного|воздушного пространства", text) and not _RECEIVE_RE.search(text):
        return None
    has_open = _OPEN_RE.search(text) is not None and _LIMIT_RE.search(text) is not None
    has_closed = (
        _CLOSED_VERB_RE.search(text) is not None
        and _LIMIT_RE.search(text) is not None
        and _OPS_RE.search(text) is not None
    ) or (
        _NOT_OPERATING_RE.search(text) is not None and find_spans(text) != []
    )
    has_restr = _RESTR_RE.search(text) is not None and _RECEIVE_RE.search(text) is not None
    if has_open and has_closed:
        open_at = _OPEN_RE.search(text).start()
        closed_at = _CLOSED_VERB_RE.search(text).start() if _CLOSED_VERB_RE.search(text) else 10**9
        return OPEN if open_at < closed_at else CLOSED
    if has_open:
        return OPEN
    if has_restr and not has_closed:
        return RESTRICTED
    if has_closed and has_restr:
        if re.search(r"принима\w*(?:\s+\w+){0,4}\s+и\s+отправ", text):
            return RESTRICTED
        return CLOSED
    if has_closed:
        return CLOSED
    if has_restr:
        return RESTRICTED
    return None


def _canon_code(token: str) -> str | None:
    mapped = token.translate(_CYR_TO_LAT)
    return _ICAO_LOWER.get(mapped)


def _mask(text: str) -> str:
    def blank(match: re.Match[str]) -> str:
        return " " * len(match.group(0))

    masked = _SKIP_SENTENCE_RE.sub(blank, text)
    return _RANGE_RE.sub(blank, masked)


def find_spans(text: str) -> list[tuple[int, int, str]]:
    masked = _mask(text)
    candidates: list[tuple[int, int, str, int]] = []
    for match in _ALIAS_RE.finditer(masked):
        key = norm_paragraph(match.group(1))
        icao = _ALIAS_MAP.get(key)
        if icao:
            candidates.append((match.start(), match.end(), icao, match.end() - match.start()))
    for match in _CODE_RE.finditer(masked):
        icao = _canon_code(match.group(1))
        if icao:
            candidates.append((match.start(), match.end(), icao, 4))
    candidates.sort(key=lambda item: (-item[3], item[0]))
    occupied: list[tuple[int, int]] = []
    chosen: list[tuple[int, int, str]] = []
    for start, end, icao, _length in candidates:
        if any(start < stop and end > begin for begin, stop in occupied):
            continue
        occupied.append((start, end))
        chosen.append((start, end, icao))
    chosen.sort()
    return chosen


def airports_with_times(text: str) -> list[tuple[str, tuple[int, int] | None]]:
    masked = _mask(text)
    spans = _collapse_runs(find_spans(text))
    times: list[tuple[int, int, int, int]] = []
    for match in _TIME_RE.finditer(masked):
        hour, minute = int(match.group(1)), int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            times.append((match.start(), match.end(), hour, minute))
    chosen: dict[str, tuple[int, int] | None] = {}
    order: list[str] = []
    for index, (start, end, icao) in enumerate(spans):
        next_start = spans[index + 1][0] if index + 1 < len(spans) else 10**9
        between = [item for item in times if end - 2 <= item[0] < min(end + 80, next_start)]
        if between:
            clock = (between[0][2], between[0][3])
        else:
            before = [item for item in times if item[1] <= start and start - item[0] <= 500]
            clock = (before[-1][2], before[-1][3]) if before else None
        explicit = bool(between)
        if icao not in chosen:
            chosen[icao] = clock
            order.append(icao)
        elif explicit:
            chosen[icao] = clock
    return [(icao, chosen[icao]) for icao in order]


def _collapse_runs(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Склеивает «Калуга ... UUBC»: часы обычно стоят после последнего упоминания."""
    reduced: list[tuple[int, int, str]] = []
    for start, end, icao in spans:
        if reduced and reduced[-1][2] == icao and start - reduced[-1][1] <= 80:
            reduced[-1] = (start, end, icao)
            continue
        reduced.append((start, end, icao))
    return reduced


def _is_attachable(text: str) -> bool:
    if classify(text):
        return False
    if _NARRATIVE_RE.search(text):
        return False
    if find_spans(text):
        return True
    return _TIME_RE.search(_mask(text)) is not None


def _is_list_gap(text: str) -> bool:
    """Короткая строка в перечне аэропортов, даже если название ещё не в словаре."""
    if classify(text) or _NARRATIVE_RE.search(text):
        return False
    words = [word for word in text.split() if word]
    return 0 < len(words) <= 6


def resolve_clock(posted_at: datetime, hour: int, minute: int) -> datetime:
    post = posted_at.astimezone(MSK)
    best: datetime | None = None
    best_dist: float | None = None
    for delta in (-1, 0, 1):
        day = (post + timedelta(days=delta)).date()
        local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=MSK)
        dist = abs((local - post).total_seconds())
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = local
    assert best is not None
    return best.astimezone(UTC)


def parse_message(
    msg_id: int,
    text: str,
    reply_text: str,
    posted_at: datetime,
) -> tuple[list[Event], str | None]:
    paragraphs = split_paragraphs(text)
    norms = [norm_paragraph(p) for p in paragraphs]
    snippet = re.sub(r"\s+", " ", text).strip()[:240]
    events: list[Event] = []
    saw_action = False
    index = 0
    while index < len(norms):
        action = classify(norms[index])
        if not action:
            index += 1
            continue
        saw_action = True
        block = [index]
        nxt = index + 1
        while nxt < len(norms) and (_is_attachable(norms[nxt]) or _is_list_gap(norms[nxt])):
            if _is_attachable(norms[nxt]):
                block.append(nxt)
            nxt += 1
        pre: list[int] = []
        prev = index - 1
        while prev >= 0 and (_is_attachable(norms[prev]) or _is_list_gap(norms[prev])):
            if _is_attachable(norms[prev]):
                pre.append(prev)
            prev -= 1
        pre.reverse()
        joined = ". ".join(norms[i] for i in pre + block)
        found = airports_with_times(joined)
        if not found and reply_text:
            found = [(icao, None) for icao, _clock in airports_with_times(norm_paragraph(reply_text))]
            # reply fallback is only the airport set; clocks stay on the main post
            main_clocks = {icao: clock for icao, clock in airports_with_times(". ".join(norms[i] for i in block))}
            found = [(icao, main_clocks.get(icao)) for icao, _clock in found]
        for icao, clock in found:
            if clock is None:
                event_time = posted_at
                source = "posted"
            else:
                event_time = resolve_clock(posted_at, clock[0], clock[1])
                source = "clock"
            events.append(
                Event(
                    msg_id=msg_id,
                    icao=icao,
                    event_type=action,
                    event_time=event_time,
                    posted_at=posted_at,
                    time_source=source,
                    snippet=snippet,
                )
            )
        index = nxt
    if saw_action and not events:
        return [], "action_without_airport"
    return events, None


def build_intervals(events: list[Event], now: datetime) -> list[Interval]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.icao].append(event)
    gap = timedelta(hours=INFER_GAP_HOURS)
    intervals: list[Interval] = []
    for icao, group in grouped.items():
        group.sort(key=lambda item: (item.event_time, item.msg_id, item.event_type))
        current: Event | None = None
        for event in group:
            if current is None:
                if event.event_type != OPEN:
                    current = event
                continue
            if event.event_time <= current.event_time:
                continue
            elapsed = event.event_time - current.event_time
            if event.event_type == current.event_type:
                if elapsed > gap:
                    intervals.append(
                        _interval(current, current.event_time + gap, inferred=True, end_msg=None)
                    )
                    current = event
                continue
            if event.event_type != OPEN and elapsed > gap:
                intervals.append(
                    _interval(current, current.event_time + gap, inferred=True, end_msg=None)
                )
                current = event
                continue
            intervals.append(
                _interval(current, event.event_time, inferred=False, end_msg=event.msg_id)
            )
            current = None if event.event_type == OPEN else event
        if current is not None:
            if now - current.event_time > gap:
                intervals.append(
                    _interval(current, current.event_time + gap, inferred=True, end_msg=None)
                )
            else:
                intervals.append(_interval(current, None, inferred=False, end_msg=None))
    intervals.sort(key=lambda item: (item.icao, item.start, item.state))
    return intervals


def _interval(
    event: Event,
    end: datetime | None,
    *,
    inferred: bool,
    end_msg: int | None,
) -> Interval:
    return Interval(
        icao=event.icao,
        state=event.event_type,
        start=event.event_time,
        end=end,
        inferred_end=inferred,
        start_msg_id=event.msg_id,
        end_msg_id=end_msg,
    )


def rebuild_from_messages(messages: list[tuple[int, datetime, str, str]], now: datetime) -> tuple[list[Event], list[Interval], list[tuple[int, str, str]]]:
    events: list[Event] = []
    misses: list[tuple[int, str, str]] = []
    for msg_id, posted_at, text, reply in messages:
        try:
            parsed, reason = parse_message(msg_id, text, reply, posted_at)
        except Exception as exc:  # одно битое сообщение не валит всю ленту
            misses.append((msg_id, f"error:{exc.__class__.__name__}", re.sub(r"\s+", " ", text)[:240]))
            continue
        events.extend(parsed)
        if reason:
            misses.append((msg_id, reason, re.sub(r"\s+", " ", text)[:240]))
    return events, build_intervals(events, now), misses


def event_row(event: Event) -> tuple:
    return (
        event.msg_id,
        event.icao,
        event.event_type,
        iso(event.event_time),
        iso(event.posted_at),
        event.time_source,
        event.snippet,
    )


def interval_row(interval: Interval) -> tuple:
    return (
        interval.icao,
        interval.state,
        iso(interval.start),
        iso(interval.end) if interval.end is not None else None,
        1 if interval.inferred_end else 0,
        interval.start_msg_id,
        interval.end_msg_id,
    )


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_self_tests() -> None:
    posted = _dt("2024-10-01T04:50:00Z")

    def types(text: str, reply: str = "", when: datetime = posted) -> set[tuple[str, str]]:
        events, reason = parse_message(1, text, reply, when)
        assert reason is None, reason
        return {(event.icao, event.event_type) for event in events}

    assert types(
        "Аэропорт КАЛУГА (Грабцево)\n\nВВЕДЕНЫ временные ограничения на прием и выпуск воздушных судов.\n\n"
        "Ограничения необходимы для обеспечения безопасности полетов."
    ) == {("UUBC", CLOSED)}

    assert types(
        "Аэропорт ВНУКОВО принимает и отправляет рейсы по согласованию с соответствующими органами "
        "в связи с введением временных ограничений на использование воздушного пространства."
    ) == {("UUWW", RESTRICTED)}

    assert types(
        "Аэропорты\n\n— ГЕЛЕНДЖИК\n\n— СОЧИ\n\nСНЯТЫ ограничения на прием и выпуск воздушных судов.\n\n"
        "Ограничения вводили для обеспечения безопасности полетов."
    ) == {("URKG", OPEN), ("URSS", OPEN)}

    assert types(
        "Аэропорты ВНУКОВО, ДОМОДЕДОВО и ЖУКОВСКИЙ (Раменское) принимают и отправляют рейсы по согласованию."
    ) == {("UUWW", RESTRICTED), ("UUDD", RESTRICTED), ("UUBW", RESTRICTED)}

    clock_events, reason = parse_message(
        7,
        "Для обеспечения безопасности полетов временные ограничения на прием и выпуск ВС в 07:30 МСК "
        "введены в аэропортах:\n\nКазань (код ИКАО: UWKD);\n\nУльяновск (Баратаевка; UWLL).",
        "",
        posted,
    )
    assert reason is None
    by_icao = {event.icao: event for event in clock_events}
    assert set(by_icao) == {"UWKD", "UWLL"}
    assert by_icao["UWKD"].event_time == _dt("2024-10-01T04:30:00Z")
    assert by_icao["UWLL"].time_source == "clock"

    # Кириллическая «Е» в коде ИКАО.
    mixed, _reason = parse_message(
        8,
        "Временные ограничения на прием и выпуск воздушных судов с 08:00 МСК введены также в аэропорту "
        "Нижнекамск (Бегишево; код ИКАО: UWKЕ).",
        "",
        posted,
    )
    assert len(mixed) == 1 and mixed[0].icao == "UWKE"
    assert mixed[0].event_time == _dt("2024-10-01T05:00:00Z")

    split_posted = _dt("2024-10-01T21:30:00Z")
    split, _reason = parse_message(
        9,
        "Временные ограничения введены на работу аэропортов:\n\n"
        "Жуковский (код ИКАО: UUBW) – с 01:00 МСК;\n\n"
        "Калуга (Грабцево; UUBC) – с 00:45 МСК.\n\n"
        "Воздушные гавани временно не принимают и не отправляют авиарейсы.",
        "",
        split_posted,
    )
    split_map = {event.icao: event for event in split}
    assert set(split_map) == {"UUBW", "UUBC"}
    assert split_map["UUBW"].event_time == _dt("2024-10-01T22:00:00Z")
    assert split_map["UUBC"].event_time == _dt("2024-10-01T21:45:00Z")

    ignored, reason = parse_message(
        10,
        "Росавиация расследует инцидент c самолетом Cessna 182 в Подмосковье. Пилот не пострадал.",
        "",
        posted,
    )
    assert ignored == [] and reason is None

    regional, reason = parse_message(
        11,
        "ПУЛКОВО\n\nПСКОВ\n\nВВЕДЕНЫ временные ограничения на прием и выпуск воздушных судов.\n\n"
        "В этой связи в аэропорту Храброво возможны корректировки в расписании рейсов.",
        "",
        posted,
    )
    assert reason is None
    assert {(e.icao, e.event_type) for e in regional} == {("ULLI", CLOSED), ("ULOO", CLOSED)}

    resume, reason = parse_message(
        12,
        "Аэропорт Пулково возобновляет обслуживание рейсов.\n\n"
        "Их прием и выпуск осуществляется по согласованию с соответствующими органами.",
        "",
        posted,
    )
    assert reason is None
    assert {(e.icao, e.event_type) for e in resume} == {("ULLI", RESTRICTED)}

    quoted_open, reason = parse_message(
        13,
        "Аэропорты\n\nГЕЛЕНДЖИК\n\nСОЧИ\n\nСНЯТЫ ограничения на прием и выпуск воздушных судов.",
        "Аэропорт ГЕЛЕНДЖИК\n\nВВЕДЕНЫ дополнительные временные ограничения на прием и выпуск воздушных судов.",
        posted,
    )
    assert reason is None
    assert {(e.icao, e.event_type) for e in quoted_open} == {("URKG", OPEN), ("URSS", OPEN)}

    fallback, reason = parse_message(
        14,
        "ВВЕДЕНЫ временные ограничения на прием и выпуск воздушных судов.",
        "Аэропорт КАЛУГА (Грабцево)",
        posted,
    )
    assert reason is None
    assert [(e.icao, e.event_type) for e in fallback] == [("UUBC", CLOSED)]

    schedule, reason = parse_message(
        15,
        "Аэропорт ГЕЛЕНДЖИК\n\nВВЕДЕНЫ дополнительные временные ограничения на прием и выпуск воздушных судов.\n\n"
        "Согласно действующему NOTAM, аэропорт Геленджик принимает регулярные рейсы с 8:30 до 20:00.",
        "",
        _dt("2026-10-26T23:25:32Z"),
    )
    assert reason is None
    assert len(schedule) == 1
    assert schedule[0].event_type == CLOSED
    assert schedule[0].time_source == "posted"
    assert schedule[0].event_time == _dt("2026-10-26T23:25:32Z")

    empty, reason = parse_message(
        16,
        "ВВЕДЕНЫ временные ограничения на прием и выпуск воздушных судов.",
        "",
        posted,
    )
    assert empty == [] and reason == "action_without_airport"

    base = _dt("2024-10-01T10:00:00Z")
    later = _dt("2024-10-01T12:00:00Z")
    restr = _dt("2024-10-01T13:00:00Z")
    opened = _dt("2024-10-01T14:00:00Z")

    def ev(msg: int, kind: str, when: datetime) -> Event:
        return Event(msg, "UUWW", kind, when, when, "posted", "")

    intervals = build_intervals(
        [ev(1, CLOSED, base), ev(2, CLOSED, later), ev(3, OPEN, opened)],
        _dt("2024-10-02T00:00:00Z"),
    )
    assert len(intervals) == 1
    assert intervals[0].start == base and intervals[0].end == opened and not intervals[0].inferred_end

    intervals = build_intervals(
        [ev(1, CLOSED, base), ev(2, RESTRICTED, later), ev(3, OPEN, opened)],
        opened,
    )
    assert [(i.state, i.start, i.end) for i in intervals] == [
        (CLOSED, base, later),
        (RESTRICTED, later, opened),
    ]

    intervals = build_intervals([ev(1, CLOSED, base)], base + timedelta(hours=30))
    assert len(intervals) == 1 and intervals[0].inferred_end and intervals[0].end == base + timedelta(hours=24)

    intervals = build_intervals(
        [ev(1, CLOSED, base), ev(2, OPEN, base + timedelta(hours=40))],
        base + timedelta(hours=50),
    )
    assert intervals[0].end == base + timedelta(hours=40) and not intervals[0].inferred_end

    intervals = build_intervals(
        [ev(1, CLOSED, base), ev(2, CLOSED, base + timedelta(hours=30))],
        base + timedelta(hours=32),
    )
    assert len(intervals) == 2
    assert intervals[0].inferred_end and intervals[0].end == base + timedelta(hours=24)
    assert intervals[1].start == base + timedelta(hours=30) and intervals[1].end is None

    # «томск» не должен прилипать к «омск»
    tomsk, _reason = parse_message(17, "Аэропорт ТОМСК\n\nВВЕДЕНЫ временные ограничения на прием и выпуск.", "", posted)
    assert [e.icao for e in tomsk] == ["UNTT"]


if __name__ == "__main__":
    run_self_tests()
    print("parser ok")
