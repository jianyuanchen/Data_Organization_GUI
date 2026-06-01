"""
Pure data layer: Meta dataclass, polymer classifier, canonical-path helper, and
the column lists that drive both the SQLite schema and the staging table.

No file I/O, no DB, no Qt. Safe to import from anywhere without side effects.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_ANNEAL_TIME = 10          # minutes; not stored in filename
SOLVENTS = ["CB", "DCB", "Tol"]   # controlled vocabulary


def canon_path(p: str) -> str:
    """Canonical spelling used as the scans.csv_path primary key.

    Collapses '/' vs '\\', '.' / '..' segments, relative-vs-absolute, and
    case differences (via normcase on Windows) into ONE string. Two paths
    that name the same file always produce the same key, so one file can
    only ever produce one row in `scans`.

    Applied at every entry point that lands a path into the DB or looks one
    up. The parser itself still works from the ORIGINAL path's stem so
    regex matches like 'AP'/'AN' aren't broken by the Windows lowercasing.
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


@dataclass
class Meta:
    csv_path: str
    series: str
    p1_name: str
    p1_backbone: str
    p1_chirality: str            # achiral | main-chain | side-chain | unknown
    p1_hand: Optional[str]       # R | S | None
    p1_pct: Optional[int]        # side-chain chiral %
    p2_name: str                 # "None" for single component
    p2_backbone: Optional[str]
    p2_chirality: Optional[str]
    p2_hand: Optional[str]
    p2_pct: Optional[int]
    n_components: int
    config: str                  # 1-comp | achiral+achiral | chiral+achiral | chiral+chiral
    ratio: str
    conc: int                    # mg/mL
    solvent: str
    film_state: str              # AP | AN
    speed_mm_s: float
    anneal_temp: Optional[int]
    anneal_time: Optional[int]   # default tag, not from filename
    peak_g: float                # peak g-value, parsed from 'gval=' token
    peak_wl: float               # peak wavelength for the g-value peak (nm)
    # Computed peaks from the verification window's Find Max/Min flow.
    # NOT filename-derived; default None and only filled in when the
    # reviewer applies a computed value. Apply sets edited=1, so the
    # value is preserved across re-browse via the existing preserved
    # path -- no need to add these to _USER_FIELDS.
    peak_cd: Optional[float] = None
    peak_cd_wl: Optional[int] = None
    peak_uv: Optional[float] = None
    peak_uv_wl: Optional[int] = None
    # ---- forward-looking metadata (stored, not yet used by any logic) -------
    # Stable per-record id. Independent of csv_path so it survives file moves
    # and is the durable hook for future MongoDB / vector-DB integrations.
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Parse-uncertainty / review flags. Default empty; will later carry e.g.
    # a JSON list of fields a tool was unsure about.
    flags: str = ""
    # Human-verification status. 0 = unverified, 1 = verified.
    verified: int = 0
    # ISO-8601 timestamp string of when `verified` flipped to 1. Nullable.
    verified_date: Optional[str] = None
    # Provenance for a future shared lab DB. Nullable.
    added_by: Optional[str] = None
    # Shared identifier for the rows inserted by a single browse action.
    # Set at INSERT time (see database.upsert_preserving_edits); rows
    # already in the DB keep their original batch_id across re-ingestion.
    batch_id: Optional[str] = None
    # Per-record verification state: pending | confirmed | rejected |
    # needs_work | unparsed. Driven by the verification window for the
    # first four; 'unparsed' is set by upsert_unparsed when a filename
    # fails parse_filename and the file is recorded as a placeholder
    # worklist entry instead of silently dropped.
    review_status: str = "pending"
    # Diagnostic message from parse_filename, set only on unparsed rows
    # (review_status='unparsed') so the user can see WHY parsing failed
    # without re-running the parser. Always None on parsed rows.
    parse_error: Optional[str] = None


def classify_polymer(token: str):
    """Return (backbone, chirality, hand, pct) for a polymer token."""
    if token == "None":
        return (None, None, None, None)
    # main-chain chiral, e.g. R-F8BT / S-F8BT
    m = re.match(r"^([RS])-(.+)$", token)
    if m:
        return (m.group(2), "main-chain", m.group(1), None)
    # side-chain chiral, e.g. C-PFBT100 / C-PFBT50
    m = re.match(r"^C-([A-Za-z0-9]+?)(\d+)$", token)
    if m:
        return (m.group(1), "side-chain", None, int(m.group(2)))
    # bare achiral, e.g. F8BT
    return (token, "achiral", None, None)


# Column list driving the SQLite schema and (filtered) the staging table.
COLUMNS = list(Meta.__annotations__.keys())

# Fields that are stored + round-tripped but never shown as editable columns
# in the staging table. Kept here so the table layout stays in sync with the
# data model.
HIDDEN_COLUMNS = ("record_id", "flags", "verified", "verified_date", "added_by",
                  "batch_id", "review_status", "parse_error")
VISIBLE_COLUMNS = [c for c in COLUMNS if c not in HIDDEN_COLUMNS]


# Row-tint hex codes per review_status. Shared between the verification
# window's sidebar and the main staging table so the two views stay in
# visual sync. Kept as plain strings here (no Qt dep); UI modules convert
# to QColor at their boundary.
#
# `needs_work` is intentionally peach (#ffe5b4) rather than the Bootstrap
# warning yellow used in some other UIs -- the main staging table's
# pending-edit cell tint is #fff3cd, and we need the two visuals to stay
# distinguishable. "" for pending means "no tint" in the main table; the
# verification sidebar falls back to a very light gray for pending so the
# row still reads as a real list entry.
REVIEW_STATUS_COLORS = {
    "pending":    "",          # no tint in the main staging table
    "confirmed":  "#d4edda",   # light green
    "rejected":   "#f8d7da",   # light pink / red
    "needs_work": "#ffe5b4",   # peach -- distinct from #fff3cd pending-edit yellow
    "unparsed":   "#d3d3d3",   # light gray -- "needs renaming, not in review flow"
}
