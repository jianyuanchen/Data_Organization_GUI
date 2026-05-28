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
    peak_wl: int                 # peak wavelength (nm)
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
HIDDEN_COLUMNS = ("record_id", "flags", "verified", "verified_date", "added_by")
VISIBLE_COLUMNS = [c for c in COLUMNS if c not in HIDDEN_COLUMNS]
