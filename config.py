"""Configuration loader for MongoDB Atlas (and other env-driven settings).

Reads `.env` from the project root via python-dotenv and exposes the MongoDB
connection settings as module-level constants. If `MONGODB_URI` is not set,
`MONGODB_URI` will be `None` and `mongo_configured()` returns False — callers
should check this before attempting to connect so the app still runs without
cloud configured.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

MONGODB_URI: str | None = os.getenv("MONGODB_URI") or None
MONGODB_DB: str = os.getenv("MONGODB_DB", "cd_automation")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "samples")


def mongo_configured() -> bool:
    return bool(MONGODB_URI)


if not mongo_configured():
    logger.warning("MongoDB not configured — set MONGODB_URI in .env")
