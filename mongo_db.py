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


def fetch_record_by_id(existing_id: str, log=print) -> dict | None:
    """Fetch ONE cloud document by its _id, INCLUDING its embedded spectra
    arrays. Read-only; used by the conflict resolver's comparison view to load
    the cloud side on demand (only for the conflict actually being viewed).

    `existing_id` is the stringified ObjectId carried in a conflict descriptor.
    Returns the document dict (with `_id` stringified) on success, or None on any
    failure (unconfigured, malformed id, connect/timeout, query error, or not
    found). Never raises -- a clear line is logged and None returned so the GUI
    shows a "could not load" placeholder rather than crashing. pymongo + bson are
    lazy-imported; the 5s fast-fail timeout rides on get_client; MONGODB_URI is
    never logged.
    """
    if not config.mongo_configured():
        log("MongoDB not configured -- set MONGODB_URI in .env.")
        return None

    # Stringified _id -> ObjectId so we address the exact document. bson ships
    # with pymongo; lazy-import to keep the GUI launchable without it.
    try:
        from bson import ObjectId
        oid = ObjectId(str(existing_id))
    except Exception as e:
        log(f"Invalid cloud _id {existing_id!r}: {type(e).__name__}: {e}")
        return None

    client = None
    try:
        client = get_client()
        collection = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
        client.admin.command("ping")
        doc = collection.find_one({"_id": oid})
    except Exception as e:
        log(f"Cloud fetch-by-id failed: {type(e).__name__}: {e}")
        return None
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    if doc is None:
        log(f"No cloud record with _id {existing_id}.")
        return None
    doc["_id"] = str(doc["_id"])
    return doc


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


def _build_cloud_doc(record: dict, wavelength, g, cd, uv, data_hash: str,
                     fkey: str, filename: str, promoted_at: str) -> dict:
    """Build the canonical stored cloud document from a local record + spectra.

    Single source of truth for the doc shape so NEW inserts (promote_records)
    and sanctioned overwrites (replace_by_id) stay schema-consistent: a replaced
    doc is byte-for-byte the same shape as a freshly inserted one. Strips any
    incoming _id -- insert_one assigns a fresh one; replace_one preserves the
    matched doc's existing _id.
    """
    doc = dict(record)
    doc.pop("_id", None)
    doc["wavelength"] = list(wavelength)
    doc["g"] = list(g)
    doc["cd"] = list(cd)
    doc["uv"] = list(uv)
    doc["data_hash"] = data_hash
    doc["filename_key"] = fkey
    doc["filename"] = filename
    doc["record_id"] = record.get("record_id")  # trace of machine, NOT a key
    doc["promoted_at"] = promoted_at
    return doc


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
        }
    `promoted` is what the caller feeds back into the local DB (set promoted=1
    + promoted_at) -- inserts AND true-duplicates appear there, since both mean
    "this record is in the cloud". `conflicts` is consumed by the in-app
    resolver (main_window._resolve_conflicts), which is the only thing that can
    overwrite a cloud doc (via replace_by_id), so nobody edits Atlas by hand.
    """
    summary: dict = {
        "pushed": 0, "already": 0, "skipped": 0, "failed": 0,
        "promoted": [], "conflicts": [],
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
        """Collect a rich conflict descriptor for the in-app resolver and log.
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
        what = " (concurrent insert)" if concurrent else ""
        log(f"  CONFLICT{what} (filename matches, data differs): {filename} "
            f"-- existing _id: {existing_id} -- upload cancelled.")

    for rec in confirmed:
        csv_path = rec.get("csv_path")
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

        # Same builder replace_by_id uses, so inserts and sanctioned
        # overwrites produce identical doc shapes.
        doc = _build_cloud_doc(rec, wavelength, g, cd, uv, data_hash, fkey,
                               filename, promoted_at)

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


def replace_by_id(existing_id: str, record: dict, log=print) -> dict:
    """Overwrite ONE existing cloud doc -- addressed BY ITS _id -- with the
    local record's full document. The ONLY sanctioned cloud-overwrite path
    (the conflict resolver's "Replace Cloud with Local").

    Targets _id, NEVER filename_key, so there is no ambiguity about which doc is
    rewritten. Builds the stored document via _build_cloud_doc -- the same shape
    promote_records' NEW-insert branch uses -- so a replaced doc is
    schema-identical to a freshly inserted one (embedded arrays + data_hash +
    filename_key + filename + provenance).

    Never raises: every failure mode (unconfigured, bad _id, CSV read, connect,
    replace error, or no matching doc) is captured into the returned dict, so the
    GUI only ever sees a result -- never a pymongo traceback. MONGODB_URI is
    never logged. pymongo + bson are lazy-imported; get_client carries the 5s
    fast-fail timeout. Uses replace_one WITHOUT upsert: if the target doc was
    deleted since the conflict was detected we report it, never silently
    recreate it.

    Returns {"ok": bool, "promoted_at": str|None, "message": str}. On ok=True the
    caller mirrors promoted_at into local SQLite via db.mark_promoted.
    """
    result = {"ok": False, "promoted_at": None, "message": ""}

    if not config.mongo_configured():
        result["message"] = "MongoDB not configured -- set MONGODB_URI in .env."
        log(result["message"])
        return result

    csv_path = record.get("csv_path")
    filename = ((csv_path or "").replace("\\", "/").rsplit("/", 1)[-1]
                or "(unknown)")

    # Resolve the stringified _id back to an ObjectId so we target the exact
    # doc. bson ships with pymongo; lazy-import to keep the GUI launchable.
    try:
        from bson import ObjectId
        oid = ObjectId(str(existing_id))
    except Exception as e:
        result["message"] = (f"Invalid cloud _id {existing_id!r} -- nothing "
                             f"replaced: {type(e).__name__}: {e}")
        log(result["message"])
        return result

    # Read + hash the local spectra exactly like the NEW-insert branch.
    try:
        from plotting import read_csv_columns
    except Exception as e:
        result["message"] = (f"Could not load CSV reader: "
                             f"{type(e).__name__}: {e}")
        log(result["message"])
        return result
    try:
        cols = read_csv_columns(csv_path)
    except Exception as e:
        result["message"] = (f"CSV read failed for {csv_path}: "
                             f"{type(e).__name__}: {e}")
        log(result["message"])
        return result
    if not cols or len(cols) < 4:
        result["message"] = f"CSV has no numeric data: {csv_path} -- skipped."
        log(result["message"])
        return result

    wavelength, g, cd, uv = cols[0], cols[1], cols[2], cols[3]
    data_hash = compute_data_hash(wavelength, g, cd, uv)
    promoted_at = datetime.now(timezone.utc).isoformat()
    doc = _build_cloud_doc(record, wavelength, g, cd, uv, data_hash,
                           _filename_key(csv_path), filename, promoted_at)

    client = None
    try:
        client = get_client()
        collection = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
        client.admin.command("ping")
        # replace_one (no upsert): rewrite the WHOLE matched doc, preserving its
        # _id. matched_count==0 means the doc is gone -- report, don't recreate.
        res = collection.replace_one({"_id": oid}, doc)
        if res.matched_count == 0:
            result["message"] = (f"No cloud record with _id {existing_id} "
                                 f"(it may have been deleted) -- nothing "
                                 f"replaced for {filename}.")
            log(result["message"])
            return result
        result["ok"] = True
        result["promoted_at"] = promoted_at
        result["message"] = (f"Replaced cloud record _id {existing_id} with "
                             f"local {filename}.")
        log(result["message"])
        return result
    except Exception as e:
        result["message"] = (f"Cloud replace failed for {filename}: "
                             f"{type(e).__name__}: {e}")
        log(result["message"])
        return result
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


# ===========================================================================
# CLASSIFICATION TAGS  (additive, keyed by record_id -- never touch spectra)
# ===========================================================================
# The CD-shape review window (Phase B) writes two ADDITIVE fields back onto each
# cloud doc, addressed by record_id:
#   - computed_metrics    : a REFRESHABLE CACHE of the classifier's OBJECTIVE
#     metrics (UV/CD peaks + both ratios + gate booleans + borderline + the
#     thresholds snapshot). NO verdict. Re-synced whenever classifier.py's
#     PROVISIONAL thresholds or a record's manual_window change; the pass is
#     idempotent and re-runnable. (Renamed from the legacy `auto_classification`
#     block, which carried an auto label/ladder_type -- the metrics writers
#     $unset that old field so docs migrate on the next write.)
#   - human_classification: the DURABLE reviewer category (Ladder/Staircase/
#     Unsure + timestamp) -- now the ONLY category label and the single source
#     of truth for sorting and stats. Set on save, $unset on clear.
# Both are pure single-field updates -- the spectral arrays and every other field
# are left exactly as-is. Neither function ever raises; failures are logged and
# reported in the return value so the GUI never crashes on a cloud hiccup.
# record_id is the key per the persistence contract; one cloud doc per
# filename_key means it addresses a single document.


def sync_computed_metrics(updates, log=print) -> dict:
    """Additively cache the classifier's OBJECTIVE metrics onto cloud docs, keyed
    by record_id. Each update $sets `computed_metrics` AND $unsets the legacy
    `auto_classification` block (migration: a doc that still carries the old auto
    label loses it on the next write). Spectra and all other fields are untouched.
    Idempotent and re-runnable: safe to call after every measure pass (e.g. once
    classifier.py's thresholds change).

    `updates`: iterable of (record_id, metrics_doc) pairs. A pair with a falsy
    record_id is skipped (a doc with no record_id can't be addressed this way).

    Returns {"matched": int, "modified": int, "skipped": int, "failed": int}.
    Never raises -- a connect/query error logs a line and counts the batch as
    failed.
    """
    summary = {"matched": 0, "modified": 0, "skipped": 0, "failed": 0}

    ops_in = list(updates)
    ops = []
    for rid, metrics in ops_in:
        if not rid:
            summary["skipped"] += 1
            continue
        ops.append((rid, metrics))

    if not ops:
        if ops_in:
            log(f"  computed-metrics sync: {summary['skipped']} skipped "
                f"(no record_id); nothing to write.")
        return summary

    if not config.mongo_configured():
        log("MongoDB not configured -- cannot sync computed metrics.")
        summary["failed"] += len(ops)
        return summary

    client = None
    try:
        from pymongo import UpdateOne
        client = get_client()
        collection = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
        client.admin.command("ping")
        bulk = [UpdateOne({"record_id": rid},
                          {"$set": {"computed_metrics": metrics},
                           "$unset": {"auto_classification": ""}})
                for rid, metrics in ops]
        res = collection.bulk_write(bulk, ordered=False)
        summary["matched"] = res.matched_count
        summary["modified"] = res.modified_count
        tail = (f"; {summary['skipped']} skipped (no record_id)"
                if summary["skipped"] else "")
        log(f"  computed-metrics sync: matched {res.matched_count}, "
            f"modified {res.modified_count}{tail}.")
    except Exception as e:
        log(f"Computed-metrics sync failed: {type(e).__name__}: {e}")
        summary["failed"] += len(ops)
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    return summary


def set_human_classification(record_id, human_doc, log=print) -> dict:
    """Set OR CLEAR the durable human category on ONE cloud doc, keyed by
    record_id. human_classification (Ladder/Staircase/Unsure + timestamp) is now
    the SINGLE source of truth for sorting and stats in Phase B.

    `human_doc`: the category dict to $set, or None to CLEAR it ($unset) --
    clearing returns the record to unreviewed/grey. ONLY this one field is
    touched; computed_metrics, manual_window and the spectra are left intact.
    update_one without upsert: a missing record_id is reported, never created.

    Returns {"ok": bool, "message": str}. Never raises.
    """
    result = {"ok": False, "message": ""}

    if not record_id:
        result["message"] = "No record_id -- cannot save human classification."
        log(result["message"])
        return result
    if not config.mongo_configured():
        result["message"] = ("MongoDB not configured -- human classification "
                             "not saved.")
        log(result["message"])
        return result

    clearing = human_doc is None
    update = ({"$unset": {"human_classification": ""}} if clearing
              else {"$set": {"human_classification": human_doc}})

    client = None
    try:
        client = get_client()
        collection = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
        client.admin.command("ping")
        res = collection.update_one({"record_id": record_id}, update)
        if res.matched_count == 0:
            verb = "not cleared" if clearing else "not saved"
            result["message"] = (f"No cloud record with record_id "
                                 f"{record_id} -- classification {verb}.")
            log(result["message"])
            return result
        result["ok"] = True
        if clearing:
            result["message"] = (f"Cleared human classification for record_id "
                                 f"{record_id}.")
        else:
            result["message"] = (f"Saved human classification "
                                 f"'{human_doc.get('label')}' for record_id "
                                 f"{record_id}.")
        log(result["message"])
        return result
    except Exception as e:
        result["message"] = (f"Saving human classification failed: "
                             f"{type(e).__name__}: {e}")
        log(result["message"])
        return result
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def set_manual_window(record_id, manual_window, metrics_doc, log=print) -> dict:
    """Persist a per-record manual CD window AND its refreshed metrics cache
    TOGETHER on ONE cloud doc, keyed by record_id, in a single atomic update.

    computed_metrics is a cache derived from BOTH the provisional constants AND
    the window used, so the two must never be written out of sync. The caller
    re-measures under the new window and hands us both:

      - `manual_window`: a {"min_nm", "max_nm"} dict to SET, or None to CLEAR
        (revert the record to the data-driven window). Clearing $unsets the
        field so it disappears from the doc rather than lingering as null.
      - `metrics_doc`: the refreshed classifier.computed_metrics_doc(result),
        always $set so the stored cache matches the window just chosen.

    The legacy `auto_classification` block is always $unset here too, so a record
    migrates to computed_metrics the next time its window is touched.

    update_one without upsert: a missing record_id is reported, never created.
    Returns {"ok": bool, "message": str}. Never raises.
    """
    result = {"ok": False, "message": ""}

    if not record_id:
        result["message"] = "No record_id -- cannot save manual window."
        log(result["message"])
        return result
    if not config.mongo_configured():
        result["message"] = ("MongoDB not configured -- manual window not "
                             "saved.")
        log(result["message"])
        return result

    # Build ONE update so manual_window and the cache land together. The legacy
    # auto_classification block is $unset for migration.
    update: dict = {"$set": {"computed_metrics": metrics_doc},
                    "$unset": {"auto_classification": ""}}
    if manual_window is None:
        update["$unset"]["manual_window"] = ""
        action = "cleared (reverted to data-driven)"
    else:
        update["$set"]["manual_window"] = manual_window
        action = (f"set [{manual_window.get('min_nm')}-"
                  f"{manual_window.get('max_nm')}] nm")

    client = None
    try:
        client = get_client()
        collection = client[config.MONGODB_DB][config.MONGODB_COLLECTION]
        client.admin.command("ping")
        res = collection.update_one({"record_id": record_id}, update)
        if res.matched_count == 0:
            result["message"] = (f"No cloud record with record_id "
                                 f"{record_id} -- manual window not saved.")
            log(result["message"])
            return result
        result["ok"] = True
        result["message"] = (f"Manual window {action} for record_id "
                             f"{record_id} (auto cache refreshed).")
        log(result["message"])
        return result
    except Exception as e:
        result["message"] = (f"Saving manual window failed: "
                             f"{type(e).__name__}: {e}")
        log(result["message"])
        return result
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
