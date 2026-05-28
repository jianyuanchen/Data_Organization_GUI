"""MongoDB Atlas push-only cloud layer.

Promotes confirmed local records to the configured Atlas collection. All
secrets are read via config.py (MONGODB_URI / MONGODB_DB / MONGODB_COLLECTION
from .env) -- never hardcoded. Every network op is guarded so a failure
(unconfigured, DNS down, auth, timeout) is returned as a summary instead of
bubbling up into the GUI.

This module is push-only by design (Phase 3). Retrieve-from-cloud will come
in a later phase.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable

import config

logger = logging.getLogger(__name__)

# 5s is short enough that a bad URI fails before the GUI feels frozen, long
# enough to forgive a single slow DNS hop or TLS handshake on Atlas.
_SERVER_SELECTION_TIMEOUT_MS = 5000


def get_client():
    """Return a MongoClient bound to config.MONGODB_URI.

    Raises RuntimeError if MONGODB_URI isn't set so callers can convert that
    into a single user-visible message instead of a pymongo traceback. The
    serverSelectionTimeoutMS keeps a bad/absent connection from hanging.
    """
    if not config.mongo_configured():
        raise RuntimeError(
            "MongoDB not configured -- set MONGODB_URI in .env")
    # Lazy import so the app still launches if pymongo isn't installed yet.
    from pymongo import MongoClient
    return MongoClient(
        config.MONGODB_URI,
        serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS,
        appname="cd-data-automation",
    )


def test_connection() -> tuple[bool, str]:
    """Ping the cluster. Returns (ok, short_message) for status display.

    Never raises; any pymongo or config error is captured into the message
    so the caller can render it directly next to a status indicator.
    """
    if not config.mongo_configured():
        return False, "MongoDB not configured (set MONGODB_URI in .env)"
    client = None
    try:
        client = get_client()
        client.admin.command("ping")
        return (True,
                f"Connected to {config.MONGODB_DB}.{config.MONGODB_COLLECTION}")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def _is_confirmed(rec: dict) -> bool:
    """Eligible-for-promotion gate. Both flags must agree.

    review_status='confirmed' alone (or verified=1 alone) means a partially-
    updated row -- treat it as not confirmed and skip with a log line.
    """
    return (rec.get("review_status") == "confirmed"
            and bool(rec.get("verified")))


def promote_records(records: Iterable[dict], log=print) -> dict:
    """Upsert each CONFIRMED local record into the configured Atlas collection.

    Records where review_status != 'confirmed' (or verified != 1) are skipped
    with a log line -- the caller may pass a raw selection that includes
    pending/rejected/needs_work/unparsed rows.

    Documents are keyed on record_id via replace_one(upsert=True), so
    re-promoting the same record updates the existing cloud doc instead of
    duplicating. Each doc carries: all the record's metadata fields, the
    embedded CSV spectra (wavelength + g + CD + UV as numeric lists), and
    provenance (added_by, verified_date, promoted_at set to now).

    Returns a summary dict:
        {
            "pushed":   int,              # newly inserted in Atlas
            "updated":  int,              # existing doc replaced (matched)
            "skipped":  int,              # not confirmed
            "failed":   int,              # CSV read / upsert / connect error
            "promoted": [(csv_path, promoted_at_iso), ...],
        }
    The `promoted` list is what the caller should feed back into the local
    DB (set promoted=1 + promoted_at) -- only rows that actually landed in
    Atlas appear in it.
    """
    summary: dict = {
        "pushed": 0, "updated": 0, "skipped": 0, "failed": 0,
        "promoted": [],
    }

    records = list(records)
    if not records:
        log("No records to promote.")
        return summary

    # 1) Eligibility filter. Non-confirmed rows never reach the network.
    confirmed = [r for r in records if _is_confirmed(r)]
    summary["skipped"] = len(records) - len(confirmed)
    if summary["skipped"]:
        log(f"  skipped {summary['skipped']} record(s): not confirmed.")
    if not confirmed:
        return summary

    # 2) Configured?
    if not config.mongo_configured():
        log("MongoDB not configured -- set MONGODB_URI in .env. Aborting.")
        summary["failed"] += len(confirmed)
        return summary

    # 3) Connect + ping up front so a bad URI fails ONCE here, not N times
    #    inside the upsert loop several seconds apart.
    client = None
    try:
        client = get_client()
        collection = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
        client.admin.command("ping")
    except Exception as e:
        log(f"MongoDB connection failed: {type(e).__name__}: {e}")
        summary["failed"] += len(confirmed)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        return summary

    # 4) Lazy import the CSV reader. plotting.py imports originpro at module
    #    top; we don't want this cloud module to need Origin just to read a
    #    CSV. The function itself is pure stdlib.
    try:
        from plotting import read_csv_columns
    except Exception as e:
        log(f"Could not load CSV reader: {type(e).__name__}: {e}")
        summary["failed"] += len(confirmed)
        try:
            client.close()
        except Exception:
            pass
        return summary

    promoted_at = datetime.now(timezone.utc).isoformat()

    for rec in confirmed:
        rid = rec.get("record_id")
        csv_path = rec.get("csv_path")
        if not rid:
            log(f"  skipping (no record_id): {csv_path}")
            summary["failed"] += 1
            continue
        try:
            cols = read_csv_columns(csv_path)
        except Exception as e:
            log(f"  CSV read failed for {csv_path}: "
                f"{type(e).__name__}: {e}")
            summary["failed"] += 1
            continue
        if not cols or len(cols) < 4:
            log(f"  CSV has no numeric data: {csv_path} -- skipped.")
            summary["failed"] += 1
            continue

        doc = dict(rec)
        doc["wavelength"] = list(cols[0])
        doc["g"] = list(cols[1])
        doc["cd"] = list(cols[2])
        doc["uv"] = list(cols[3])
        doc["promoted_at"] = promoted_at
        # Stable key for upsert + future retrieval.
        doc["record_id"] = rid

        try:
            result = collection.replace_one(
                {"record_id": rid}, doc, upsert=True)
        except Exception as e:
            log(f"  upsert failed for {csv_path}: "
                f"{type(e).__name__}: {e}")
            summary["failed"] += 1
            continue

        if result.upserted_id is not None:
            summary["pushed"] += 1
            log(f"  pushed: {csv_path}")
        else:
            # matched_count > 0 OR no-op replacement; either way the doc is
            # in Atlas under this record_id.
            summary["updated"] += 1
            log(f"  updated: {csv_path}")
        summary["promoted"].append((csv_path, promoted_at))

    try:
        client.close()
    except Exception:
        pass

    return summary
