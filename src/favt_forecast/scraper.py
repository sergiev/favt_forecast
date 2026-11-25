"""Загрузка публичной ленты https://t.me/s/favt_info без API-ключей Telegram."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup, Tag

from favt_forecast.config import (
    CHANNEL,
    EMBED_URL,
    MAX_BACKFILL_PAGES,
    MAX_CATCHUP_PAGES,
    PAGE_URL,
    REQUEST_DELAY_SEC,
    REQUEST_TIMEOUT_SEC,
    USER_AGENT,
)
from favt_forecast.db import connect, get_state, max_msg_id, set_state, upsert_message
from favt_forecast.util import iso, now_utc, parse_dt

_VIEWS_RE = re.compile(r"([\d]+(?:[.,]\d+)?)\s*([KM])?", re.IGNORECASE)


@dataclass
class RawPost:
    msg_id: int
    posted_at: str
    text: str
    reply_text: str
    views: int | None

    @property
    def text_hash(self) -> str:
        payload = f"{self.posted_at}\n{self.text}\n{self.reply_text}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class Page:
    posts: list[RawPost]
    older_before: int | None

    @property
    def ids(self) -> list[int]:
        return [post.msg_id for post in self.posts]


def needs_embed(text: str) -> bool:
    stripped = text.rstrip()
    return "…" in text or stripped.endswith("...") or stripped.endswith("…")


def parse_views(value: str | None) -> int | None:
    if not value:
        return None
    match = _VIEWS_RE.search(value.replace(" ", ""))
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    suffix = (match.group(2) or "").upper()
    mult = {"K": 1_000, "M": 1_000_000}.get(suffix, 1)
    return int(number * mult)


def node_to_text(node: Tag | None) -> str:
    if node is None:
        return ""
    for br in node.find_all("br"):
        br.replace_with("\n")
    for emoji in node.find_all("i", class_="emoji"):
        emoji.replace_with(" ")
    text = node.get_text("", strip=False)
    text = text.replace("\xa0", " ").replace("\u200b", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_channel_html(html: str) -> Page:
    soup = BeautifulSoup(html, "html.parser")
    posts: list[RawPost] = []
    for node in soup.select("div.js-widget_message"):
        data_post = node.get("data-post") or ""
        match = re.search(rf"{re.escape(CHANNEL)}/(\d+)", str(data_post))
        if not match:
            continue
        time_node = node.select_one(".tgme_widget_message_date time")
        posted_raw = time_node.get("datetime") if time_node is not None else None
        if not posted_raw:
            continue
        posted_at = iso(parse_dt(str(posted_raw)))
        reply_node = node.select_one(".js-message_reply_text")
        main_node = node.select_one(".js-message_text")
        views_node = node.select_one(".tgme_widget_message_views")
        posts.append(
            RawPost(
                msg_id=int(match.group(1)),
                posted_at=posted_at,
                text=node_to_text(main_node),
                reply_text=node_to_text(reply_node),
                views=parse_views(views_node.get_text(" ", strip=True) if views_node else None),
            )
        )
    posts.sort(key=lambda post: post.msg_id)
    return Page(posts=posts, older_before=_older_cursor(soup, [post.msg_id for post in posts]))


def _older_cursor(soup: BeautifulSoup, ids: list[int]) -> int | None:
    if not ids:
        return None
    floor = min(ids)
    found: list[int] = []
    for link in soup.select("a.tme_messages_more"):
        href = str(link.get("href") or "")
        match = re.search(r"before=(\d+)", href)
        if not match:
            continue
        value = int(match.group(1))
        if value <= floor:
            found.append(value)
    if not found:
        return None
    return min(found)


class ChannelScraper:
    def __init__(self) -> None:
        self.client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"},
            timeout=REQUEST_TIMEOUT_SEC,
            follow_redirects=True,
        )
        self.embeds = 0

    def close(self) -> None:
        self.client.close()

    def _get(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.client.get(url)
                if response.status_code in {429, 500, 502, 503}:
                    time.sleep(min(30, 2 ** attempt))
                    continue
                response.raise_for_status()
                time.sleep(REQUEST_DELAY_SEC)
                return response.text
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"не удалось скачать {url}: {last_error}")

    def fetch_page(self, before: int | None = None) -> Page:
        url = PAGE_URL if before is None else f"{PAGE_URL}?before={before}"
        return parse_channel_html(self._get(url))

    def hydrate(self, post: RawPost) -> RawPost:
        if not needs_embed(post.text):
            return post
        html = self._get(EMBED_URL.format(channel=CHANNEL, msg_id=post.msg_id))
        self.embeds += 1
        page = parse_channel_html(html)
        full = next((item for item in page.posts if item.msg_id == post.msg_id), None)
        if full is None:
            return post
        text = full.text if len(full.text) >= len(post.text) else post.text
        reply = full.reply_text if len(full.reply_text) >= len(post.reply_text) else post.reply_text
        return RawPost(post.msg_id, post.posted_at or full.posted_at, text, reply, post.views or full.views)


def sync_channel(continue_backfill: bool = True, on_progress=None) -> dict:
    conn = connect()
    scraper = ChannelScraper()
    stats = {
        "new": 0,
        "updated": 0,
        "pages": 0,
        "embeds": 0,
        "backfill_complete": get_state(conn, "backfill_complete", "0") == "1",
    }

    def report(stage: str) -> None:
        stats["embeds"] = scraper.embeds
        if on_progress is not None:
            on_progress(stage, dict(stats))

    def consume(page: Page) -> None:
        fetched_at = iso(now_utc())
        for post in page.posts:
            post = scraper.hydrate(post)
            kind = upsert_message(
                conn,
                msg_id=post.msg_id,
                posted_at=post.posted_at,
                text=post.text,
                reply_text=post.reply_text,
                views=post.views,
                text_hash=post.text_hash,
                fetched_at=fetched_at,
            )
            if kind == "new":
                stats["new"] += 1
            elif kind == "updated":
                stats["updated"] += 1
        stats["pages"] += 1
        stats["embeds"] = scraper.embeds
        conn.commit()

    try:
        _catch_up(conn, scraper, consume, report)
        if continue_backfill and get_state(conn, "backfill_complete", "0") != "1":
            _backfill(conn, scraper, consume, report)
            _catch_up(conn, scraper, consume, report)
        stats["backfill_complete"] = get_state(conn, "backfill_complete", "0") == "1"
        set_state(conn, "last_sync_at", iso(now_utc()))
        conn.commit()
        report("done")
        return stats
    finally:
        scraper.close()
        conn.close()


def _catch_up(conn, scraper: ChannelScraper, consume, report) -> None:
    last_max = max_msg_id(conn)
    before: int | None = None
    seen: set[int] = set()
    capped = True
    for _ in range(MAX_CATCHUP_PAGES):
        page = scraper.fetch_page(before)
        report("catchup")
        if not page.posts:
            capped = False
            break
        consume(page)
        report("catchup")
        page_min = min(page.ids)
        if last_max and page_min <= last_max:
            capped = False
            break
        if last_max == 0:
            if page.older_before and not get_state(conn, "backfill_cursor"):
                set_state(conn, "backfill_cursor", str(page.older_before))
                conn.commit()
            capped = False
            break
        if page.older_before is None or page.older_before in seen or page.older_before == before:
            capped = False
            break
        seen.add(page.older_before)
        before = page.older_before
    if capped and last_max and before:
        # Дыра больше окна догонки: дочитываем её тем же обратным обходом.
        set_state(conn, "backfill_complete", "0")
        set_state(conn, "backfill_cursor", str(before))
        conn.commit()


def _backfill(conn, scraper: ChannelScraper, consume, report) -> None:
    cursor_raw = get_state(conn, "backfill_cursor")
    if not cursor_raw:
        page = scraper.fetch_page(None)
        report("backfill")
        if page.posts:
            consume(page)
        if page.older_before is None:
            set_state(conn, "backfill_complete", "1")
            conn.commit()
            return
        cursor_raw = str(page.older_before)
        set_state(conn, "backfill_cursor", cursor_raw)
        conn.commit()
    cursor = int(cursor_raw)
    seen = {cursor}
    for _ in range(MAX_BACKFILL_PAGES):
        page = scraper.fetch_page(cursor)
        report("backfill")
        if not page.posts:
            set_state(conn, "backfill_complete", "1")
            conn.commit()
            return
        page_max = max(page.ids)
        consume(page)
        report("backfill")
        older = page.older_before
        if older is None or older in seen or page_max >= cursor:
            set_state(conn, "backfill_complete", "1")
            set_state(conn, "backfill_cursor", "")
            conn.commit()
            return
        seen.add(older)
        cursor = older
        set_state(conn, "backfill_cursor", str(cursor))
        conn.commit()
    raise RuntimeError("бэкфилл превысил предохранитель по числу страниц")


def run_self_tests() -> None:
    html = """
    <div class="tgme_widget_message js-widget_message" data-post="favt_info/12">
      <a class="tgme_widget_message_date" href="https://t.me/favt_info/12">
        <time datetime="2024-01-02T03:04:05+00:00" class="time">03:04</time>
      </a>
      <a class="tgme_widget_message_reply" href="https://t.me/favt_info/11">
        <div class="tgme_widget_message_text js-message_reply_text">Аэропорт СОЧИ<br/>ВВЕДЕНЫ</div>
      </a>
      <div class="tgme_widget_message_text js-message_text">Аэропорт <b>КАЛУГА</b><br/><br/>СНЯТЫ ограничения</div>
      <span class="tgme_widget_message_views">1.2K</span>
    </div>
    <a class="tme_messages_more" href="/s/favt_info?before=12"></a>
    <a class="tme_messages_more" href="/s/favt_info?before=40"></a>
    """
    page = parse_channel_html(html)
    assert page.older_before == 12
    assert len(page.posts) == 1
    post = page.posts[0]
    assert post.msg_id == 12
    assert post.posted_at == "2024-01-02T03:04:05Z"
    assert "КАЛУГА" in post.text and "СНЯТЫ" in post.text
    assert "СОЧИ" not in post.text
    assert "СОЧИ" in post.reply_text and "ВВЕДЕНЫ" in post.reply_text
    assert post.views == 1200
    assert needs_embed("текст до…")
    assert not needs_embed("Аэропорт Калуга. Сняты ограничения.")
    assert parse_views("29.9K") == 29900
    assert parse_views("311") == 311


if __name__ == "__main__":
    run_self_tests()
    print("scraper ok")
