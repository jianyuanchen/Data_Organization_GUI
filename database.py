"""
SQLite layer. Owns the schema, forward-only migrations, dedup pass, prune,
and the three upsert flavors. Source of truth for parsed scan metadata.
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import asdict

from models import COLUMNS, Meta, canon_path


DB_PATH = "cd_metadata.db"


_TEXT_COLS = {"csv_path", "series", "p1_name", "p1_backbone", "p1_chirality",
              "p1_hand", "p2_name", "p2_backbone", "p2_chirality", "p2_hand",
              "config", "ratio", "solvent", "film_state",
              # forward-looking metadata
              "record_id", "flags", "verified_date", "added_by"}
_REAL_COLS = {"speed_mm_s", "peak_g"}


def _sqltype(col: str) -> str:
    if col in _TEXT_COLS:
        return "TEXT"
    if col in _REAL_COLS:
        return "REAL"
    return "INTEGER"


# Fields stored on every row but excluded from the filename-driven UPDATE path
# of upsert_preserving_edits. Once set on a row (either by INSERT or by user
# action), these survive re-ingestion of the same csv_path.
_USER_FIELDS = {"record_id", "flags", "verified", "verified_date", "added_by"}


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

    def upsert_preserving_edits(self, m: Meta) -> str:
        """Three-way upsert that protects manual corrections:

            row missing               -> INSERT, edited=0    -> 'new'
            row exists, edited == 0   -> UPDATE from Meta    -> 'updated'
            row exists, edited == 1   -> leave untouched     -> 'preserved'

        On the UPDATE path, the user-metadata fields in _USER_FIELDS
        (record_id, flags, verified, verified_date, added_by) are excluded
        from the SET clause so they survive re-ingestion -- the filename
        only governs filename-derived data.
        """
        cur = self.conn.execute(
            "SELECT edited FROM scans WHERE csv_path=?", (m.csv_path,))
        row = cur.fetchone()
        d = asdict(m)
        if row is None:
            placeholders = ", ".join("?" for _ in COLUMNS)
            self.conn.execute(
                f"INSERT INTO scans ({', '.join(COLUMNS)}) "
                f"VALUES ({placeholders})", [d[c] for c in COLUMNS])
            self.conn.commit()
            return "new"
        if row["edited"]:
            return "preserved"
        non_pk = [c for c in COLUMNS
                  if c != "csv_path" and c not in _USER_FIELDS]
        set_clause = ", ".join(f"{c}=?" for c in non_pk)
        self.conn.execute(
            f"UPDATE scans SET {set_clause} WHERE csv_path=?",
            [d[c] for c in non_pk] + [m.csv_path])
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
