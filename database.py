"""
SQLite layer. Owns the schema, forward-only migrations, dedup pass, prune,
and the three upsert flavors. Source of truth for parsed scan metadata.
"""
from __future__ import annotations

import getpass
import json
import os
import sqlite3
import uuid
from dataclasses import asdict
from typing import Optional

from models import COLUMNS, Meta, canon_path


DB_PATH = "cd_metadata.db"


_TEXT_COLS = {"csv_path", "series", "p1_name", "p1_backbone", "p1_chirality",
              "p1_hand", "p2_name", "p2_backbone", "p2_chirality", "p2_hand",
              "config", "ratio", "solvent", "film_state",
              # additive block (manifest path only; regex rows carry NULL).
              # Supersedes the old `dopant` TEXT column.
              "additive1_name", "additive1_role", "additive1_unit",
              # forward-looking metadata
              "record_id", "flags", "verified_date", "added_by",
              # batch + verification state
              "batch_id", "review_status",
              # parse diagnostic (only set on review_status='unparsed' rows)
              "parse_error"}
_REAL_COLS = {"speed_mm_s", "peak_g", "peak_wl", "peak_cd", "peak_uv",
              # additive block numerics (supersede the old `dopant_conc` REAL)
              "additive1_conc", "additive1_min"}

# Manifest-ingest extras. Live on every scans row but are NOT part of
# COLUMNS / Meta -- same pattern as `edited` / `promoted`: managed only by
# ingest_manifest_record, invisible to the staging table and the upsert
# flavors, and therefore zero-impact on the regex path and everything
# downstream (verification window, promote-to-cloud).
#   arrays_json            embedded spectral arrays {wavelength,g,cd,uv}
#   data_hash              SHA-256 tripwire over the embedded arrays
#   source_folder          provenance string from the manifest row
#   manifest_version       manifest schema version that produced the row
#   manifest_generated_at  Cowork's ISO timestamp for the manifest
#   manifest_peak_g/_wl    the manifest's CHECKSUM peak values (provenance
#                          only -- peak_g/peak_wl hold the computed truth)
#   ingest_source          'manifest' (NULL on regex-path rows)
_MANIFEST_EXTRA_COLS = (
    ("arrays_json", "TEXT"),
    ("data_hash", "TEXT"),
    ("source_folder", "TEXT"),
    ("manifest_version", "TEXT"),
    ("manifest_generated_at", "TEXT"),
    ("manifest_peak_g", "REAL"),
    ("manifest_peak_wl", "REAL"),
    ("ingest_source", "TEXT"),
)


def _sqltype(col: str) -> str:
    if col in _TEXT_COLS:
        return "TEXT"
    if col in _REAL_COLS:
        return "REAL"
    return "INTEGER"


# Fields stored on every row but excluded from the filename-driven UPDATE path
# of upsert_preserving_edits. Once set on a row (either by INSERT or by user
# action), these survive re-ingestion of the same csv_path -- batch_id stays
# bound to the row's first-seen browse, review_status stays where the user
# left it, etc.
_USER_FIELDS = {"record_id", "flags", "verified", "verified_date", "added_by",
                "batch_id", "review_status"}


class DB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        # Persist across runs. Fresh DB: CREATE TABLE builds the full schema.
        # Existing DB: CREATE IF NOT EXISTS is a no-op, then ALTER TABLE ADD
        # COLUMN migrates forward -- each call is wrapped because SQLite raises
        # OperationalError when the column already exists, which is the normal
        # case on every subsequent startup.
        cols = ", ".join(f"{c} {_sqltype(c)}" for c in COLUMNS)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS scans ({cols}, "
            f"edited INTEGER NOT NULL DEFAULT 0, "
            f"PRIMARY KEY(csv_path))")
        for c in COLUMNS:
            try:
                self.conn.execute(
                    f"ALTER TABLE scans ADD COLUMN {c} {_sqltype(c)}")
            except sqlite3.OperationalError:
                pass
        try:
            self.conn.execute(
                "ALTER TABLE scans ADD COLUMN edited INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        # Cloud-promotion tracking. Mirrors the `edited` pattern: lives on
        # every row but is not part of COLUMNS / Meta -- managed only by
        # mark_promoted after a successful Mongo upsert. promoted=1 means
        # "this record has been copied to Atlas as of promoted_at".
        try:
            self.conn.execute(
                "ALTER TABLE scans ADD COLUMN promoted "
                "INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            self.conn.execute(
                "ALTER TABLE scans ADD COLUMN promoted_at TEXT")
        except sqlite3.OperationalError:
            pass
        # Manifest-ingest extras (see _MANIFEST_EXTRA_COLS). Same idempotent
        # ALTER TABLE migration pattern as edited/promoted above.
        for col, sqltype in _MANIFEST_EXTRA_COLS:
            try:
                self.conn.execute(
                    f"ALTER TABLE scans ADD COLUMN {col} {sqltype}")
            except sqlite3.OperationalError:
                pass
        self.conn.commit()

        # One-time (per-startup, but idempotent) migrations. Stash counts so
        # MainWindow can log them once log_box exists.
        try:
            self._dedup_count = self._dedupe_canonical_paths()
        except Exception:
            self._dedup_count = 0
        try:
            self._backfill_count = self._backfill_record_ids()
        except Exception:
            self._backfill_count = 0
        try:
            self._review_status_backfill = self._backfill_review_status()
        except Exception:
            self._review_status_backfill = 0

    def _dedupe_canonical_paths(self) -> int:
        """Collapse rows whose csv_path differs only in slash/case/normalization.

        For each group of rows sharing one canonical path:
          - keep the one with edited=1 if any (preserve manual corrections),
          - delete the rest,
          - UPDATE the survivor's csv_path to the canonical form.

        Idempotent: on a clean DB every group has a single row already in
        canonical form, so no DELETE / UPDATE runs.

        Returns the number of duplicate rows removed.
        """
        rows = self.conn.execute(
            "SELECT csv_path, edited FROM scans").fetchall()
        groups: dict[str, list[tuple[str, int]]] = {}
        for row in rows:
            key = canon_path(row["csv_path"])
            groups.setdefault(key, []).append((row["csv_path"], row["edited"]))

        removed = 0
        for canonical, members in groups.items():
            # Sort: edited=1 first so the manually-corrected row survives.
            members.sort(key=lambda m: 0 if m[1] else 1)
            survivor = members[0][0]
            for csv_path, _ in members[1:]:
                self.conn.execute(
                    "DELETE FROM scans WHERE csv_path=?", (csv_path,))
                removed += 1
            if survivor != canonical:
                try:
                    self.conn.execute(
                        "UPDATE scans SET csv_path=? WHERE csv_path=?",
                        (canonical, survivor))
                except sqlite3.IntegrityError:
                    # Should be unreachable -- every group owns its canonical
                    # key -- but if a collision sneaks in, drop the survivor
                    # rather than letting the loop crash.
                    self.conn.execute(
                        "DELETE FROM scans WHERE csv_path=?", (survivor,))
                    removed += 1
        self.conn.commit()
        return removed

    def _backfill_record_ids(self) -> int:
        """Give every legacy row a stable record_id.

        Rows ingested before the column existed land here with record_id NULL
        after ALTER TABLE. Assign a fresh uuid per row, once. Subsequent
        startups find nothing to do.
        """
        rows = self.conn.execute(
            "SELECT csv_path FROM scans "
            "WHERE record_id IS NULL OR record_id=''").fetchall()
        n = 0
        for row in rows:
            self.conn.execute(
                "UPDATE scans SET record_id=? WHERE csv_path=?",
                (str(uuid.uuid4()), row["csv_path"]))
            n += 1
        self.conn.commit()
        return n

    def _backfill_review_status(self) -> int:
        """Default existing rows to review_status='pending' after migration.

        ALTER TABLE ADD COLUMN lands a NULL on every existing row; treating
        NULL as 'pending' keeps queries / sidebar coloring uniform.
        Idempotent: subsequent startups find nothing to update.
        """
        rows = self.conn.execute(
            "SELECT csv_path FROM scans "
            "WHERE review_status IS NULL OR review_status=''").fetchall()
        n = 0
        for row in rows:
            self.conn.execute(
                "UPDATE scans SET review_status='pending' WHERE csv_path=?",
                (row["csv_path"],))
            n += 1
        self.conn.commit()
        return n

    def prune_missing(self) -> tuple[int, int]:
        """Delete rows whose csv_path file no longer exists on disk.

        Returns (total_pruned, pruned_that_were_edited). os.path.exists errors
        are treated as 'exists' so a transient FS hiccup never wipes the row.
        """
        rows = self.conn.execute(
            "SELECT csv_path, edited FROM scans").fetchall()
        total = 0
        edited = 0
        for row in rows:
            path = row["csv_path"]
            try:
                exists = os.path.exists(path)
            except Exception:
                exists = True
            if not exists:
                self.conn.execute(
                    "DELETE FROM scans WHERE csv_path=?", (path,))
                total += 1
                if row["edited"]:
                    edited += 1
        self.conn.commit()
        return total, edited

    def upsert(self, m: Meta):
        """Plain INSERT OR REPLACE. Clobbers the row (and resets the edited
        flag to 0). Kept for callers that explicitly want filename truth;
        on_browse uses upsert_preserving_edits instead.
        """
        d = asdict(m)
        placeholders = ", ".join("?" for _ in COLUMNS)
        self.conn.execute(
            f"INSERT OR REPLACE INTO scans ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders})", [d[c] for c in COLUMNS])
        self.conn.commit()

    def upsert_preserving_edits(self, m: Meta,
                                 batch_id: Optional[str] = None) -> str:
        """Three-way upsert that protects manual corrections:

            row missing               -> INSERT, edited=0    -> 'new'
            row exists, edited == 0   -> UPDATE from Meta    -> 'updated'
            row exists, edited == 1   -> leave content       -> 'preserved'

        On the UPDATE path, the user-metadata fields in _USER_FIELDS
        (record_id, flags, verified, verified_date, added_by, batch_id,
        review_status) are excluded from the filename-derived SET clause
        so they survive re-ingestion -- the filename only governs
        filename-derived data.

        `batch_id` is a re-tagging concern, intentionally handled
        separately from the preserve semantics: when passed, it is
        (re)applied on ALL THREE paths so a re-browse of a folder always
        re-groups its files under the current view. The user's content
        (edits, review_status, verification metadata, record_id) is still
        preserved on the 'preserved' path -- only batch_id moves.

        ADDITIVE LABELING (additive1_*): these are ordinary COLUMNS, so the
        whole-row 'preserved' path already protects them once a row is
        hand-edited. A human-confirmed additive role/unit becomes edited==1
        the moment it is changed in the staging table (update_cell) or the
        verification window (set_review's field_edits, which also flips
        edited=1), so a subsequent manifest re-import takes the 'preserved'
        branch and never reverts the labeling work. An UN-edited row, by
        contrast, legitimately re-tracks the manifest+registry (an undoped
        row that a corrected manifest dopes must update), which is exactly
        the 'updated' branch below.
        """
        cur = self.conn.execute(
            "SELECT edited FROM scans WHERE csv_path=?", (m.csv_path,))
        row = cur.fetchone()
        d = asdict(m)
        if row is None:
            if batch_id is not None:
                d["batch_id"] = batch_id
            placeholders = ", ".join("?" for _ in COLUMNS)
            self.conn.execute(
                f"INSERT INTO scans ({', '.join(COLUMNS)}) "
                f"VALUES ({placeholders})", [d[c] for c in COLUMNS])
            self.conn.commit()
            return "new"
        if row["edited"]:
            # Content preserved; only re-tag the batch_id if one was
            # passed. Nothing else is touched -- edited values,
            # review_status, verified, record_id all stay put.
            if batch_id is not None:
                self.conn.execute(
                    "UPDATE scans SET batch_id=? WHERE csv_path=?",
                    (batch_id, m.csv_path))
                self.conn.commit()
            return "preserved"
        # Un-edited existing row: refresh filename-derived fields AND
        # re-tag the batch_id when one is passed. The remaining user
        # fields (record_id / review_status / verified / verified_date /
        # added_by / flags) are still protected via _USER_FIELDS.
        non_pk = [c for c in COLUMNS
                  if c != "csv_path" and c not in _USER_FIELDS]
        sets = [f"{c}=?" for c in non_pk]
        vals = [d[c] for c in non_pk]
        if batch_id is not None:
            sets.append("batch_id=?")
            vals.append(batch_id)
        self.conn.execute(
            f"UPDATE scans SET {', '.join(sets)} WHERE csv_path=?",
            vals + [m.csv_path])
        self.conn.commit()
        return "updated"

    def upsert_unparsed(self, csv_path: str, parse_error: str,
                        batch_id: Optional[str] = None) -> str:
        """Record a placeholder row for a file whose filename could not be
        parsed. Sets review_status='unparsed', stores the parse error
        message, attaches the current batch_id, and gives the row a
        record_id like any other.

        Other columns are left NULL -- the staging table's display logic
        renders None as empty, so mostly-empty rows are fine. Unparsed
        rows are visible (gray tint, parse_error tooltip), excluded from
        plotting, and the verification window refuses Confirm on them.

        If a row already exists at this canonical path, only the
        unparsed-related fields (parse_error, batch_id, review_status)
        are touched. Everything else (record_id, manual edits, prior
        review state) is left intact -- the parse failure shouldn't
        clobber data the user may have entered.

        Returns 'new' on INSERT, 'updated' on UPDATE.
        """
        canonical = canon_path(csv_path)
        existing = self.conn.execute(
            "SELECT csv_path FROM scans WHERE csv_path=?",
            (canonical,)).fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO scans "
                "(csv_path, record_id, batch_id, parse_error, "
                " review_status, flags, verified, edited) "
                "VALUES (?, ?, ?, ?, 'unparsed', '', 0, 0)",
                (canonical, str(uuid.uuid4()), batch_id, parse_error))
            self.conn.commit()
            return "new"
        self.conn.execute(
            "UPDATE scans SET parse_error=?, batch_id=?, "
            "review_status='unparsed' WHERE csv_path=?",
            (parse_error, batch_id, canonical))
        self.conn.commit()
        return "updated"

    def update_cell(self, csv_path: str, column: str, value):
        if column not in COLUMNS or column == "csv_path":
            return
        # Defensive: keys in the DB are canonical, so the lookup must be too.
        csv_path = canon_path(csv_path)
        # Mark the row as manually edited so the next re-ingest preserves it.
        # COLUMNS doesn't contain 'edited', so this path is the only way the
        # flag gets set to 1 (other than a fresh INSERT, which sets 0).
        self.conn.execute(
            f"UPDATE scans SET {column}=?, edited=1 WHERE csv_path=?",
            (value, csv_path))
        self.conn.commit()

    def query(self, where: str = "", params: tuple = ()):
        sql = "SELECT * FROM scans"
        if where:
            sql += " WHERE " + where
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def distinct(self, column: str):
        return [r[0] for r in self.conn.execute(
            f"SELECT DISTINCT {column} FROM scans ORDER BY {column}").fetchall()
            if r[0] is not None]

    # ---- batch + verification helpers ------------------------------------
    def records_in_batch(self, batch_id: str) -> list[dict]:
        """All rows tagged with this batch, in insertion order (rowid)."""
        rows = self.conn.execute(
            "SELECT * FROM scans WHERE batch_id=? ORDER BY rowid",
            (batch_id,)).fetchall()
        return [dict(r) for r in rows]

    def records_by_paths(self, csv_paths) -> list[dict]:
        """Rows whose canonical csv_path is in csv_paths, in rowid order.

        Canonicalizes every input path so callers can pass raw paths from
        any source without worrying about case / slash differences. Empty
        list -> empty result (no SQL "IN ()" pitfall).
        """
        if not csv_paths:
            return []
        canonical = [canon_path(p) for p in csv_paths]
        placeholders = ",".join("?" for _ in canonical)
        rows = self.conn.execute(
            f"SELECT * FROM scans WHERE csv_path IN ({placeholders}) "
            f"ORDER BY rowid",
            tuple(canonical)).fetchall()
        return [dict(r) for r in rows]

    def delete_records(self, csv_paths) -> int:
        """Delete rows by canonical csv_path. LOCAL ONLY -- never touches the
        cloud, so anything already promoted to MongoDB stays safe in Atlas.

        Canonicalizes every input path (like records_by_paths) so callers can
        pass raw paths from the table. Returns the number of rows removed.
        Empty input -> 0, with no SQL "IN ()" pitfall.
        """
        if not csv_paths:
            return 0
        canonical = [canon_path(p) for p in csv_paths]
        placeholders = ",".join("?" for _ in canonical)
        cur = self.conn.execute(
            f"DELETE FROM scans WHERE csv_path IN ({placeholders})",
            tuple(canonical))
        self.conn.commit()
        return cur.rowcount

    def batches_summary(self) -> list[tuple]:
        """Return (batch_id, folder_name, row_count) per batch, most-recent
        first. `folder_name` is the basename of the dirname of any row in
        that batch -- since a browse points at one folder, this is a
        deterministic label for the dropdown.
        """
        rows = self.conn.execute(
            "SELECT batch_id, csv_path FROM scans "
            "WHERE batch_id IS NOT NULL").fetchall()
        agg: dict = {}
        for r in rows:
            bid = r["batch_id"]
            if bid not in agg:
                folder = os.path.basename(os.path.dirname(r["csv_path"])) \
                    or "(unknown)"
                agg[bid] = [folder, 0]
            agg[bid][1] += 1
        # batch_ids are ISO-timestamp prefixed, so plain string-sort DESC
        # gives most-recent-first.
        return [(bid, info[0], info[1])
                for bid, info in sorted(agg.items(), reverse=True)]

    def latest_batch_id(self) -> Optional[str]:
        """Highest-sorting batch_id in the DB, or None if no batches yet.

        Batch ids are ISO-timestamp-prefixed (see main_window.new_batch_id),
        so string DESC == most-recent-first.
        """
        row = self.conn.execute(
            "SELECT batch_id FROM scans WHERE batch_id IS NOT NULL "
            "ORDER BY batch_id DESC LIMIT 1").fetchone()
        return row["batch_id"] if row else None

    def mark_promoted(self, csv_path: str, promoted_at: str):
        """Flip a row's local promoted state after a successful Atlas upsert.

        The cloud upsert is the source of truth for whether a doc exists in
        Atlas; this method only records the local mirror so the UI can show
        a marker and the user can see when it was last pushed. Idempotent --
        calling twice just overwrites the timestamp.
        """
        csv_path = canon_path(csv_path)
        self.conn.execute(
            "UPDATE scans SET promoted=1, promoted_at=? WHERE csv_path=?",
            (promoted_at, csv_path))
        self.conn.commit()

    def set_review(self, csv_path: str, status: str, *,
                   verified: Optional[int] = None,
                   verified_date: Optional[str] = None,
                   added_by: Optional[str] = None,
                   field_edits: Optional[dict] = None):
        """Update review_status + optional verification metadata and field
        edits in one statement. `field_edits` map of column->new value goes
        through the same UPDATE; if non-empty, edited=1 is also set so the
        next re-ingest preserves the changes.
        """
        csv_path = canon_path(csv_path)
        sets = ["review_status=?"]
        vals: list = [status]
        if verified is not None:
            sets.append("verified=?")
            vals.append(verified)
        if verified_date is not None:
            sets.append("verified_date=?")
            vals.append(verified_date)
        if added_by is not None:
            sets.append("added_by=?")
            vals.append(added_by)
        if field_edits:
            for col, v in field_edits.items():
                if col in COLUMNS and col != "csv_path" and col not in _USER_FIELDS:
                    sets.append(f"{col}=?")
                    vals.append(v)
            sets.append("edited=1")
        self.conn.execute(
            f"UPDATE scans SET {', '.join(sets)} WHERE csv_path=?",
            vals + [csv_path])
        self.conn.commit()


# ---------------------------------------------------------------------------
# Manifest Confirm -> SQLite ingest (the ONLY place manifest-path CSVs are
# read). manifest.py validated the metadata and built the Meta; this function
# loads the spectral arrays, embeds them, computes the authoritative peak
# values, cross-checks the manifest's checksum peaks, and lands everything
# through the SAME staging writer the regex path uses
# (upsert_preserving_edits), so verification + promote flows are untouched.
# ---------------------------------------------------------------------------

# Mirrors plotting.ROW_LIMIT for the fallback reader below.
_SPECTRA_ROW_LIMIT = 801


def _read_spectra(path: str) -> list[list[float]]:
    """[wavelength, g, cd, uv] float lists for a data CSV.

    Prefers plotting.read_csv_columns (the canonical reader, shared with the
    verification window and promote). plotting imports originpro at module
    level, so on a machine without Origin we fall back to an equivalent
    inline reader -- same parsing rules (tab->comma, numeric rows only,
    ROW_LIMIT cap), so the embedded arrays and data_hash are identical
    either way. Raises ValueError when no numeric rows are found.
    """
    try:
        from plotting import read_csv_columns
        cols = read_csv_columns(path)
    except ImportError:
        rows = []
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                parts = line.replace("\t", ",").split(",")
                try:
                    rows.append([float(parts[i]) for i in range(4)])
                except (ValueError, IndexError):
                    continue
                if len(rows) == _SPECTRA_ROW_LIMIT:
                    break
        cols = [list(col) for col in zip(*rows)] if rows else []
    if not cols or len(cols) < 4 or not cols[0]:
        raise ValueError("no numeric 4-column data rows found")
    return cols


def _peak_from_arrays(wavelength, g) -> tuple[float, float]:
    """(peak_g, peak_wl) computed from the loaded arrays: the signed g value
    of largest magnitude and the wavelength it occurs at -- the same quantity
    the filename convention's gval=/nm tokens record. Non-finite points are
    skipped; raises ValueError if nothing usable remains."""
    best_i = None
    best_mag = -1.0
    n = min(len(wavelength), len(g))
    for i in range(n):
        v = g[i]
        # NaN fails both comparisons; +/-inf is rejected explicitly.
        mag = abs(v)
        if mag != mag or mag == float("inf"):
            continue
        if mag > best_mag:
            best_mag = mag
            best_i = i
    if best_i is None:
        raise ValueError("g array has no finite values")
    return float(g[best_i]), float(wavelength[best_i])


def ingest_manifest_record(db: DB, meta, *, manifest_row: dict,
                           carried_flags=(), batch_id: Optional[str] = None,
                           log=print) -> dict:
    """Stage one confirmed manifest row: arrays + metadata as one record.

    `meta` is manifest.build_metadata's Meta; `manifest_row` is the validated
    parsed dict (for the checksum peaks + provenance strings);
    `carried_flags` are needs_review reasons the human chose to confirm
    anyway -- they ride along in `flags` so the verification window shows
    why the row is amber.

    Steps (the ingest contract):
      a. load spectral arrays from meta.csv_path  (the ONLY CSV read)
      b. embed them in the staging row (arrays_json)
      c. record_id was minted by the Meta dataclass -- stable from here on
      d. compute peak g / peak wavelength FROM the arrays
      e. cross-check the manifest's checksum peaks within tolerance;
         mismatch flags the record (never blocks)
      f. the COMPUTED peaks land in peak_g/peak_wl (authoritative); the
         manifest's stay in manifest_peak_g/_wl as provenance
      g. data_hash over the embedded arrays (re-read tripwire; same hash
         the cloud layer uses, so promote dedup behavior is unchanged)
      h. provenance: added_by, verified=0, source_folder, manifest_version

    Returns {"ok", "result", "reasons", "peak_g", "peak_wl"}; ok=False (with
    "error") when the CSV could not be read -- nothing is staged then.
    """
    # a. Load arrays. A failure here means there is no record to stage:
    # the manifest path embeds arrays atomically with the metadata.
    try:
        wavelength, g, cd, uv = _read_spectra(meta.csv_path)
    except Exception as e:
        return {"ok": False,
                "error": f"could not read spectra: {type(e).__name__}: {e}"}

    reasons = list(carried_flags)

    # d. Computed peaks are the source of truth.
    try:
        peak_g, peak_wl = _peak_from_arrays(wavelength, g)
    except ValueError as e:
        return {"ok": False, "error": f"could not compute peaks: {e}"}

    # e. Derived-field cross-check against the manifest's checksum values.
    # config is lazy-imported (it pulls dotenv at module level, and database
    # must stay importable without it).
    try:
        from config import (MANIFEST_PEAK_G_ABS_TOL, MANIFEST_PEAK_G_REL_TOL,
                            MANIFEST_PEAK_WL_TOL_NM)
    except Exception:
        MANIFEST_PEAK_WL_TOL_NM, MANIFEST_PEAK_G_REL_TOL, \
            MANIFEST_PEAK_G_ABS_TOL = 3.0, 0.05, 1e-3
    m_g = manifest_row.get("peak_gval")
    m_wl = manifest_row.get("peak_wl_nm")
    if m_wl is not None and abs(float(m_wl) - peak_wl) > MANIFEST_PEAK_WL_TOL_NM:
        reasons.append(f"derived-field mismatch: manifest peak_wl_nm {m_wl} "
                       f"vs computed {peak_wl:g}")
    if m_g is not None:
        tol = max(MANIFEST_PEAK_G_REL_TOL * abs(peak_g),
                  MANIFEST_PEAK_G_ABS_TOL)
        if abs(float(m_g) - peak_g) > tol:
            reasons.append(f"derived-field mismatch: manifest peak_gval {m_g} "
                           f"vs computed {peak_g:g}")

    # f. + h. Authoritative peaks + provenance onto the Meta before staging.
    meta.peak_g = peak_g
    meta.peak_wl = peak_wl
    meta.flags = json.dumps(reasons) if reasons else ""
    meta.review_status = "needs_review" if reasons else "pending"
    meta.verified = 0
    meta.verified_date = None
    try:
        meta.added_by = getpass.getuser()
    except Exception:
        meta.added_by = "manifest-import"

    # Same staging writer as the regex path: new rows INSERT with everything
    # above; existing rows keep their user fields (review_status, flags,
    # record_id, ...) exactly as a re-browse would.
    result = db.upsert_preserving_edits(meta, batch_id=batch_id)
    if result != "new" and reasons:
        log(f"  note: {os.path.basename(meta.csv_path)} re-ingested with "
            f"flags (existing review state preserved): {'; '.join(reasons)}")

    # g. Tripwire hash over the embedded arrays. Reuse the cloud layer's
    # implementation so the local hash always equals the one promote will
    # compute; if mongo_db can't import (e.g. dotenv missing) we store NULL
    # rather than inventing a second, divergent hash.
    data_hash = None
    try:
        from mongo_db import compute_data_hash
        data_hash = compute_data_hash(wavelength, g, cd, uv)
    except Exception as e:
        log(f"  (data_hash unavailable: {type(e).__name__}: {e})")

    # b. Embed arrays + manifest provenance. These extras live outside
    # COLUMNS, so the upsert above never touches them -- written every
    # ingest (the file is their source of truth even on preserved rows).
    arrays_json = json.dumps({"wavelength": wavelength, "g": g,
                              "cd": cd, "uv": uv})
    db.conn.execute(
        "UPDATE scans SET arrays_json=?, data_hash=?, source_folder=?, "
        "manifest_version=?, manifest_generated_at=?, manifest_peak_g=?, "
        "manifest_peak_wl=?, ingest_source='manifest' WHERE csv_path=?",
        (arrays_json, data_hash, manifest_row.get("source_folder"),
         manifest_row.get("manifest_version"),
         manifest_row.get("generated_at"), m_g, m_wl, meta.csv_path))
    db.conn.commit()

    return {"ok": True, "result": result, "reasons": reasons,
            "peak_g": peak_g, "peak_wl": peak_wl}
