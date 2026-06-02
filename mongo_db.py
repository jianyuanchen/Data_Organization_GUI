"""MongoDB Atlas cloud layer.

Promotes confirmed local records to the configured Atlas collection and
fetches them back for the read-only cloud browser. All secrets are read via
config.py (MONGODB_URI / MONGODB_DB / MONGODB_COLLECTION from .env) -- never
hardcoded. Every network op is guarded so a failure (unconfigured, DNS down,
auth, timeout) is returned as a summary/empty result instead of bubbling up
into the GUI.

Phase 3 added push (promote_records); Phase 3b adds the download counterpart
(fetch_records) for the cloud browser. Cloud records stay read-only -- the
only way to change one is to re-promote the corrected local record.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import Iterable

import config

logger = logging.getLogger(__name__)

# 5s is short enough that a bad URI fails before the GUI feels frozen, long
# enough to forgive a single slow DNS hop or TLS handshake on Atlas.
_SERVER_SELECTION_TIMEOUT_MS = 5000

# Cross-machine identity. record_id is a per-machine UUID, so it CANNOT key
# the cloud: the same experiment uploaded from two computers carries two
# record_ids. Identity is instead (filename_key, data_hash) -- a normalized
# filename plus a content hash of the spectral arrays -- which is identical on
# any machine for the same measurement. See compute_data_hash / _filename_key.
_HASH_DECIMALS = 6


def _filename_key(csv_path: str) -> str:
    """Path-INDEPENDENT identity key derived from the file's basename alone.

    The folder a file lives in must never affect cloud dedup, so the SAME
    experiment file in ANY directory on ANY machine has to collapse to one key:

      1. os.path.basename -- drop every directory/path component (separator-
         agnostic: backslashes are slashed first so Windows paths split too).
      2. drop the file extension.
      3. strip any leading "._" (macOS AppleDouble) / stray leading
         punctuation / whitespace.
      4. lowercase and strip surrounding whitespace.

    e.g. r"C:\\\\runs\\\\F8BT_500nm.csv", "/data/F8BT_500nm.CSV", and
    "._F8BT_500nm.csv" all yield "f8bt_500nm".
    """
    raw = (csv_path or "").replace("\\", "/")
    base = os.path.basename(raw)             # (1) drop folders
    base = os.path.splitext(base)[0]         # (2) drop extension
    base = re.sub(r"^[^0-9A-Za-z]+", "", base)  # (3) leading "._"/punct/space
    return base.strip().lower()              # (4) casefold + trim


def _doc_name_key(doc: dict) -> str:
    """Normalize an EXISTING cloud doc's stored name with the same rule.

    Legacy docs may carry a path-dependent or differently-normalized
    filename_key (older builds kept the extension and the directory could leak
    in). Re-deriving the key from the best available stored field at comparison
    time means both sides use the current rule, so old records still match.
    """
    src = (doc.get("filename") or doc.get("csv_path")
           or doc.get("filename_key") or "")
    return _filename_key(src)


def _fmt_value(x) -> str:
    """One spectral value as a fixed 6-dp string, with -0.0 collapsed to 0.0.

    A fixed decimal width plus negative-zero normalization make the hash input
    byte-identical for the same measurement on any machine -- tiny negatives
    that round to zero (e.g. -1e-9) and a literal -0.0 both serialize as
    "0.000000" so they never split into two cloud documents.
    """
    r = round(float(x), _HASH_DECIMALS)
    if r == 0.0:                       # True for 0.0 and -0.0 alike
        r = 0.0
    return f"{r:.{_HASH_DECIMALS}f}"


def compute_data_hash(wavelength, g, cd, uv) -> str:
    """Stable SHA-256 over the SPECTRAL ARRAYS ONLY -- no metadata, no filename.

    Each value is rounded to 6 dp on a throwaway serialization (the document
    still stores the full-precision arrays unchanged) and emitted in a fixed
    labeled order, so the same scan hashes identically across computers. 6 dp
    is generous enough to tell genuinely different scans apart while tolerating
    trivial floating-point / export noise that would otherwise look like a
    different measurement and create a false conflict.
    """
    parts = []
    for label, arr in (("wavelength", wavelength), ("g", g),
                       ("cd", cd), ("uv", uv)):
        parts.append(label + ":" + ",".join(_fmt_value(v) for v in (arr or [])))
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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


def fetch_records(query: dict | None = None, log=print) -> list[dict]:
    """Read records from the configured Atlas collection (read-only).

    `query` is an optional MongoDB filter dict (e.g. {"solvent": "CB"}). When
    None or empty, every document in the collection is returned. The signature
    accepts an arbitrary filter dict from the start so richer cascading
    filters (polymer / chirality / temperature) can be added later without
    changing callers -- this phase only passes simple equality criteria.

    Every failure mode (unconfigured, import missing, connect timeout, query
    error) is caught: a clear message is logged and an empty list is returned
    so the GUI never crashes on a cloud hiccup.

    Each returned dict carries the document's metadata fields plus the
    embedded spectra arrays (wavelength / cd / g / uv) exactly as stored by
    promote_records. The Mongo ObjectId (`_id`) is stringified so callers can
    treat the dict as plain JSON-ish data.
    """
    query = query or {}

    if not config.mongo_configured():
        log("MongoDB not configured -- set MONGODB_URI in .env.")
        return []

    client = None
    try:
        client = get_client()
        collection = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
        client.admin.command("ping")
        records = list(collection.find(query))
    except Exception as e:
        log(f"Cloud fetch failed: {type(e).__name__}: {e}")
        return []
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    # ObjectId isn't needed downstream and isn't plain-data; stringify it so
    # the record is safe to hand around the GUI.
    for rec in records:
        if "_id" in rec:
            rec["_id"] = str(rec["_id"])
    log(f"Fetched {len(records)} cloud record(s).")
    return records


def _is_confirmed(rec: dict) -> bool:
    """Eligible-for-promotion gate. Both flags must agree.

    review_status='confirmed' alone (or verified=1 alone) means a partially-
    updated row -- treat it as not confirmed and skip with a log line.
    """
    return (rec.get("review_status") == "confirmed"
            and bool(rec.get("verified")))


def promote_records(records: Iterable[dict], log=print) -> dict:
    """Insert each CONFIRMED local record into the Atlas collection, deduped.

    Records where review_status != 'confirmed' (or verified != 1) are skipped
    with a log line -- the caller may pass a raw selection that includes
    pending/rejected/needs_work/unparsed rows.

    Cross-machine dedup: identity is (filename_key, data_hash), NOT record_id
    (a per-machine UUID). For each confirmed record we read its spectra, hash
    them (compute_data_hash), normalize its filename (_filename_key), then look
    the collection up and branch:

      (a) filename_key AND data_hash both match an existing doc -> TRUE
          DUPLICATE. Nothing is written (existing _id untouched); counted as
          'already' and reported "Already in cloud, no change: <file>". This is
          also what a machine re-promoting its own record hits.
      (b) filename_key matches but data_hash differs -> CONFLICT. Nothing is
          written; a conflict detail is recorded for the caller to surface as
          an error dialog, and the upload is cancelled for that record.
      (c) data_hash matches but filename_key differs -> CONFLICT (same data,
          different name). Same cancel-and-report path as (b).
      (d) neither matches -> NEW. insert_one assigns a fresh Mongo _id.

    No path overwrites a differing document -- conflicts are cancelled, never
    merged. record_id is still stored as a trace of the promoting machine, but
    is no longer the identity key.

    Each doc carries: all the record's metadata fields, the embedded CSV
    spectra (wavelength + g + CD + UV as full-precision numeric lists),
    data_hash + filename_key + filename, and provenance (record_id, added_by,
    verified_date, promoted_at set to now).

    Returns a summary dict:
        {
            "pushed":    int,             # new docs inserted (case d)
            "already":   int,             # true duplicates, unchanged (case a)
            "conflicts": int,             # cancelled identity mismatches (b/c)
            "skipped":   int,             # not confirmed
            "failed":    int,             # CSV read / connect / insert error
            "promoted":  [(csv_path, promoted_at_iso), ...],
            "conflict_details": [(kind, filename, existing_id, message), ...],
        }
    `promoted` is what the caller feeds back into the local DB (set promoted=1
    + promoted_at) -- inserts AND true-duplicates appear there, since both mean
    "this record is in the cloud". `conflict_details` (kind is 'name' or
    'data'; existing_id is the stringified _id of the conflicting cloud doc) is
    what the caller pops as error dialogs.
    """
    summary: dict = {
        "pushed": 0, "already": 0, "conflicts": 0, "skipped": 0, "failed": 0,
        "promoted": [], "conflict_details": [],
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
    #    inside the loop several seconds apart.
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

    # 3b) Index the identity fields for general query hygiene (fetch/back-office
    #     filtering). Dedup itself matches in memory with normalization, so it
    #     doesn't rely on these. Idempotent and non-unique; failure is non-fatal.
    try:
        collection.create_index("filename_key")
        collection.create_index("data_hash")
    except Exception as e:
        log(f"  (cloud index note) {type(e).__name__}: {e}")

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

    # 5) Build an in-memory dedup index from the existing docs, normalizing
    #    each stored name with the CURRENT rule (_doc_name_key). Doing the
    #    match in memory -- rather than querying filename_key server-side --
    #    means legacy docs whose stored key was path-dependent or differently
    #    normalized still match correctly, because both sides use today's rule.
    #    Only light identifying fields are projected (no big spectra arrays).
    #    New inserts append to this list so duplicates within one batch are
    #    caught too.
    try:
        existing = list(collection.find(
            {}, {"filename_key": 1, "filename": 1, "csv_path": 1,
                 "data_hash": 1}))
    except Exception as e:
        log(f"Cloud read for dedup failed: {type(e).__name__}: {e}")
        summary["failed"] += len(confirmed)
        try:
            client.close()
        except Exception:
            pass
        return summary

    dedup_index = [
        {"key": _doc_name_key(d), "hash": d.get("data_hash"),
         "_id": d.get("_id"),
         "filename": (d.get("filename") or d.get("filename_key")
                      or "(unknown)")}
        for d in existing
    ]

    promoted_at = datetime.now(timezone.utc).isoformat()

    for rec in confirmed:
        csv_path = rec.get("csv_path")
        rid = rec.get("record_id")
        # fkey (lowercased) is the match key; filename keeps the original case
        # just for human-readable logs / dialog messages and the stored doc.
        fkey = _filename_key(csv_path)
        filename = ((csv_path or "").replace("\\", "/").rsplit("/", 1)[-1]
                    or "(unknown)")

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

        wavelength, g, cd, uv = cols[0], cols[1], cols[2], cols[3]
        data_hash = compute_data_hash(wavelength, g, cd, uv)

        # Path-independent dedup, matched in memory so both sides use today's
        # normalization: exact (a) first, then name-only (b), then hash-only
        # (c). fkey is already basename-only, so the same file in any folder
        # collapses to the same key as the stored doc's normalized name.
        exact = next((e for e in dedup_index
                      if e["key"] == fkey and e["hash"] == data_hash), None)
        by_name = (None if exact is not None else
                   next((e for e in dedup_index if e["key"] == fkey), None))
        by_hash = (None if (exact is not None or by_name is not None) else
                   next((e for e in dedup_index if e["hash"] == data_hash),
                        None))

        # (a) TRUE DUPLICATE -- same name AND same data already in cloud. Do not
        #     rewrite (existing _id untouched); record it as already-present.
        if exact is not None:
            summary["already"] += 1
            log(f"  Already in cloud, no change: {filename}")
            summary["promoted"].append((csv_path, promoted_at))
            continue

        # (b) CONFLICT -- same filename, different data. Cancel, do not write.
        #     Surface the existing doc's _id so the user can locate that exact
        #     record in Atlas to inspect/delete when resolving manually.
        if by_name is not None:
            existing_id = by_name.get("_id")
            msg = ("Filename matches an existing cloud record but the data "
                   f"differs. Existing record _id: {existing_id}. Please "
                   "double-check the filename -- same names should have the "
                   f"same data. Upload cancelled for {filename}.")
            log(f"  CONFLICT (name matches, data differs): {filename} "
                f"-- existing _id: {existing_id} -- upload cancelled.")
            summary["conflicts"] += 1
            summary["conflict_details"].append(
                ("name", filename, str(existing_id), msg))
            continue

        # (c) CONFLICT -- same data under a different filename. Cancel. Include
        #     the existing doc's _id alongside its filename for manual lookup.
        if by_hash is not None:
            existing_id = by_hash.get("_id")
            other = by_hash.get("filename") or "(unknown)"
            msg = ("This data already exists in the cloud under a different "
                   f"filename ({other}, _id: {existing_id}). Possible "
                   f"duplicate/naming inconsistency. Upload cancelled for "
                   f"{filename}.")
            log(f"  CONFLICT (data matches '{other}', _id: {existing_id}, "
                f"name differs): {filename} -- upload cancelled.")
            summary["conflicts"] += 1
            summary["conflict_details"].append(
                ("data", filename, str(existing_id), msg))
            continue

        # (d) NEW -- insert with a fresh Mongo _id. Full-precision arrays are
        #     stored as-is; only the hash used a rounded throwaway copy.
        doc = dict(rec)
        doc.pop("_id", None)               # never carry in / regenerate an _id
        doc["wavelength"] = list(wavelength)
        doc["g"] = list(g)
        doc["cd"] = list(cd)
        doc["uv"] = list(uv)
        doc["data_hash"] = data_hash
        doc["filename_key"] = fkey
        doc["filename"] = filename
        doc["record_id"] = rid             # trace of promoting machine, NOT key
        doc["promoted_at"] = promoted_at

        try:
            result = collection.insert_one(doc)
        except Exception as e:
            log(f"  insert failed for {csv_path}: "
                f"{type(e).__name__}: {e}")
            summary["failed"] += 1
            continue

        # Keep the in-memory index current so a second copy of this file later
        # in the SAME batch is detected as a true duplicate, not re-inserted.
        dedup_index.append({"key": fkey, "hash": data_hash,
                            "_id": result.inserted_id, "filename": filename})
        summary["pushed"] += 1
        log(f"  pushed (new): {filename}")
        summary["promoted"].append((csv_path, promoted_at))

    try:
        client.close()
    except Exception:
        pass

    return summary
