from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
DATA = PROJECT_DIR / "data"
DB_PATH = DATA / "favt.db"
MODEL_DIR = DATA / "models"

CHANNEL = "favt_info"
PAGE_URL = f"https://t.me/s/{CHANNEL}"
EMBED_URL = "https://t.me/{channel}/{msg_id}?embed=1"

HORIZON_HOURS = 336
INFER_GAP_HOURS = 24
MSK_OFFSET_HOURS = 3

REQUEST_DELAY_SEC = 0.28
REQUEST_TIMEOUT_SEC = 40.0
MAX_BACKFILL_PAGES = 2500
MAX_CATCHUP_PAGES = 250

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

NEAR_HORIZONS = (1, 2, 3, 4, 6, 8, 12, 18, 24, 30, 36)
FAR_HORIZONS = (12, 18, 24, 36, 48, 72, 96, 120, 168, 216, 288, 336)
BLEND_LO = 18.0
BLEND_HI = 36.0

MAX_TRAIN_ROWS = 280_000
MAX_ORIGINS = 1800


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
