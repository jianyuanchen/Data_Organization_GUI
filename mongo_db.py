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

    `verified` is coerced via int(... or 0) == 1 rather than bool(...): a stray
    string "0" (which is truthy under bool()) must NOT read as verified. Real
    rows store an INTEGER 0/1, so this is exact for them and defensive for any
    stringy value that slips in.
    """
    return (rec.get("review_status") == "confirmed"
            and int(rec.get("verified") or 0) == 1)


# Metadata fields NOT compared when describing a filename conflict to the
# resolver: identity-derived fields, the bulk spectral arrays, and anything
# that legitimately varies per machine / per upload (path, ids, bookkeeping,
# workflow). What's left is the descriptive measurement metadata a human would
# actually want to reconcile (solvent, ratio, conc, peaks, ...).
_NON_COMPARABLE = {
    "_id",
    "wavelength", "g", "cd", "uv",            # bulk spectral arrays
    "data_hash", "filename", "filename_key",  # identity / derived
    "csv_path", "record_id", "batch_id",      # per-machine
    "promoted", "promoted_at", "edited",      # local bookkeeping
    "added_by", "verified", "verified_date",  # provenance / workflow
    "review_status", "parse_error", "flags",  # workflow / diagnostics
}


def _diff_fields(rec: dict, doc: dict) -> list:
    """Sorted names of metadata fields that differ between a local record and an
    existing cloud doc, ignoring the _NON_COMPARABLE bookkeeping/identity set.

    Only enriches a conflict descriptor so an in-app resolver can show 'these
    fields disagree' -- it never gates a write.
    """
    keys = (set(rec.keys()) | set(doc.keys())) - _NON_COMPARABLE
    return sorted(k for k in keys if rec.get(k) != doc.get(k))


def promote_records(records: Iterable[dict], log=print) -> dict:
    """Insert each CONFIRMED local record into the Atlas collection, deduped.

    Records where review_status != 'confirmed' (or verified != 1) are skipped
    with a log line -- the caller may pass a raw selection that includes
    pending/rejected/needs_work/unparsed rows.

    Identity is the normalized FILENAME ALONE (filename_key). data_hash is still
    computed and stored, but is now ADVISORY -- NOT part of identity. The lab
    uses a strict filename convention, so the filename is the trustworthy key.
    For each confirmed record we read its spectra, hash them (compute_data_hash),
    normalize its filename (_filename_key), look the collection up BY
    filename_key (a targeted, indexed query -- no full-collection scan), then
    branch:

      (a) a doc with this filename_key exists AND its data_hash matches -> TRUE
          DUPLICATE. Nothing is written (existing _id untouched); counted as
          'already' and reported "Already in cloud, no change: <file>". This is
          also what a machine re-promoting its own record hits.
      (b) a doc with this filename_key exists but its data_hash DIFFERS ->
          CONFLICT. Nothing is written; a rich conflict descriptor is collected
          and returned for an in-app resolver. Reject, never overwrite.
      (c) no doc with this filename_key -> NEW: insert_one assigns a fresh _id.
          If the identical data_hash already exists under a DIFFERENT filename,
          that is a different identity, so it is only LOGGED as an advisory and
          does NOT block the insert.

    Concurrency: a UNIQUE index on filename_key is the authority. If a parallel
    promote inserts the same filename_key between our lookup and our insert, the
    insert raises DuplicateKeyError; we re-read the now-existing doc and route it
    back through (a)/(b) rather than counting a generic failure.

    No path overwrites a differing document -- conflicts are cancelled, never
    merged. record_id is stored as a trace of the promoting machine, not a key.

    Each inserted doc carries: all the record's metadata fields, the embedded
    CSV spectra (wavelength + g + CD + UV as full-precision numeric lists),
    data_hash + filename_key + filename, and provenance (record_id, added_by,
    verified_date, promoted_at set to now).

    Returns a summary dict:
        {
            "pushed":    int,             # new docs inserted (case c)
            "already":   int,             # true duplicates, unchanged (case a)
            "skipped":   int,             # not confirmed
            "failed":    int,             # CSV read / connect / insert error
            "promoted":  [(csv_path, promoted_at_iso), ...],
            "conflicts": [ {              # case b -- for an in-app resolver
                "record":           dict,   # the local record as passed in
                "existing_id":      str,    # stringified _id of the cloud doc
                "filename_key":     str,
                "filename":         str,
                "differing_fields": [str],  # metadata fields that disagree
                "data_differs":     bool,   # spectral data/hash differs
            }, ... ],
            # Legacy mirror of `conflicts`, kept so the current main_window
            # (which still reads conflict_details) keeps surfacing conflict
            # dialogs until it migrates to the richer `conflicts` list.
            "conflict_details": [(kind, filename, existing_id, message), ...],
        }
    `promoted` is what the caller feeds back into the local DB (set promoted=1
    + promoted_at) -- inserts AND true-duplicates appear there, since both mean
    "this record is in the cloud".
    """
    summary: dict = {
        "pushed": 0, "already": 0, "skipped": 0, "failed": 0,
        "promoted": [], "conflicts": [], "conflict_details": [],
    }

    records = list(records)
    if not records:
        log("No records to promote.")
        return summary

    # 1) Eligibility filter. Non-confirmed rows never reach the network -- this
    #    runs BEFORE get_client() below, so if nothing is confirmed we return
    #    here without ever opening a connection.
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

    # 3a) DuplicateKeyError is the authority behind the unique filename_key index
    #     (see the insert path). pymongo imported fine above (get_client used
    #     it), so this import is safe; guarded anyway so an unexpected failure
    #     degrades to "no concurrent-insert recovery" rather than crashing.
    try:
        from pymongo.errors import DuplicateKeyError
    except Exception:
        DuplicateKeyError = None

    # 3b) Indexes. Identity is filename_key ALONE, so make THAT unique -- the DB
    #     then enforces one doc per filename even under concurrent promotes.
    #     data_hash stays a PLAIN (non-unique) index for the advisory/back-office
    #     lookups only. Idempotent; a failure here is non-fatal because the
    #     application-level check below still runs.
    try:
        collection.create_index("filename_key", unique=True)
    except Exception as e:
        log(f"  (cloud index note) filename_key unique: "
            f"{type(e).__name__}: {e}")
    try:
        collection.create_index("data_hash")
    except Exception as e:
        log(f"  (cloud index note) data_hash: {type(e).__name__}: {e}")

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

    # Metadata-only projection: never pull the ~800-point spectra arrays back
    # just to dedupe. Used for the filename_key lookup AND the advisory
    # data_hash lookup (pure exclusion is legal and keeps _id + metadata).
    _META_ONLY = {"wavelength": 0, "g": 0, "cd": 0, "uv": 0}

    def _find_by_fkey(fkey):
        """Existing doc for this filename_key (metadata only), or None."""
        return collection.find_one({"filename_key": fkey}, _META_ONLY)

    def _note_duplicate(csv_path, filename, *, concurrent=False):
        summary["already"] += 1
        suffix = " (concurrent insert)" if concurrent else ""
        log(f"  Already in cloud, no change{suffix}: {filename}")
        summary["promoted"].append((csv_path, promoted_at))

    def _record_conflict(rec, existing, fkey, filename, data_hash,
                         *, concurrent=False):
        """Collect a rich conflict descriptor (+ a legacy detail tuple) and log.
        Never writes -- the differing cloud doc is left exactly as-is."""
        existing_id = existing.get("_id")
        summary["conflicts"].append({
            "record": rec,
            "existing_id": str(existing_id),
            "filename_key": fkey,
            "filename": filename,
            "differing_fields": _diff_fields(rec, existing),
            "data_differs": existing.get("data_hash") != data_hash,
        })
        msg = ("Filename matches an existing cloud record but the data "
               f"differs. Existing record _id: {existing_id}. Please "
               "double-check the filename -- same names should have the "
               f"same data. Upload cancelled for {filename}.")
        summary["conflict_details"].append(
            ("name", filename, str(existing_id), msg))
        what = " (concurrent insert)" if concurrent else ""
        log(f"  CONFLICT{what} (filename matches, data differs): {filename} "
            f"-- existing _id: {existing_id} -- upload cancelled.")

    for rec in confirmed:
        csv_path = rec.get("csv_path")
        rid = rec.get("record_id")
        # fkey (lowercased) is the identity key; filename keeps the original
        # case just for human-readable logs / messages and the stored doc.
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

        # --- Dedup by filename_key ALONE, via a targeted indexed lookup ------
        try:
            existing = _find_by_fkey(fkey)
        except Exception as e:
            log(f"  cloud lookup failed for {filename}: "
                f"{type(e).__name__}: {e}")
            summary["failed"] += 1
            continue

        if existing is not None:
            if existing.get("data_hash") == data_hash:
                # (a) TRUE DUPLICATE -- same filename, same data.
                _note_duplicate(csv_path, filename)
            else:
                # (b) CONFLICT -- same filename, different data. Reject.
                _record_conflict(rec, existing, fkey, filename, data_hash)
            continue

        # (c) NEW. Advisory only: is this exact data already present under a
        #     DIFFERENT filename? Different filename == different identity, so
        #     we do NOT block -- just leave a breadcrumb. Best-effort: a failed
        #     advisory lookup must not stop the insert. (We already know no doc
        #     has this filename_key, so any data_hash hit is a different name.)
        try:
            other = collection.find_one({"data_hash": data_hash}, _META_ONLY)
        except Exception:
            other = None
        if other is not None:
            other_name = (other.get("filename") or other.get("filename_key")
                          or "(unknown)")
            log(f"  note: identical spectral data already present under "
                f"filename {other_name} (_id {other.get('_id')}) -- "
                f"inserting {filename} as new (different filename).")

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
            collection.insert_one(doc)
        except Exception as e:
            # A unique-index violation means a concurrent promote inserted this
            # filename_key first. Re-read it and route through (a)/(b) -- that
            # is NOT a generic failure. Any other error is a real failure.
            is_dup = (DuplicateKeyError is not None
                      and isinstance(e, DuplicateKeyError))
            if not is_dup:
                log(f"  insert failed for {csv_path}: "
                    f"{type(e).__name__}: {e}")
                summary["failed"] += 1
                continue
            try:
                raced = _find_by_fkey(fkey)
            except Exception as e2:
                log(f"  insert raced then re-read failed for {filename}: "
                    f"{type(e2).__name__}: {e2}")
                summary["failed"] += 1
                continue
            if raced is None:
                # Inserted-then-removed in the gap; redo it next time.
                log(f"  insert raced for {filename} but no doc on re-read "
                    f"-- counted failed.")
                summary["failed"] += 1
                continue
            if raced.get("data_hash") == data_hash:
                _note_duplicate(csv_path, filename, concurrent=True)
            else:
                _record_conflict(rec, raced, fkey, filename, data_hash,
                                 concurrent=True)
            continue

        summary["pushed"] += 1
        log(f"  pushed (new): {filename}")
        summary["promoted"].append((csv_path, promoted_at))

    try:
        client.close()
    except Exception:
        pass

    return summary
