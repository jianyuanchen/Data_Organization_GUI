"""
SQLite layer. Owns the schema, forward-only migrations, dedup pass, prune,
and the three upsert flavors. Source of truth for parsed scan metadata.
"""
from __future__ import annotations

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
              # forward-looking metadata
              "record_id", "flags", "verified_date", "added_by",
              # batch + verification state
              "batch_id", "review_status",
              # parse diagnostic (only set on review_status='unparsed' rows)
              "parse_error"}
_REAL_COLS = {"speed_mm_s", "peak_g", "peak_wl", "peak_cd", "peak_uv"}


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
