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


# ---------------------------------------------------------------------------
# Manifest ingest schema -- SINGLE SOURCE OF TRUTH for manifest.csv columns.
#
# Both sides of the contract cite this table: the Cowork prompt that GENERATES
# a manifest is written against it, and manifest.validate_row enforces it on
# the way in. Column order here is the canonical manifest column order.
#
# Keys per entry:
#   name      manifest.csv header
#   required  True / False. Conditionally-required columns are required=False
#             with the condition spelled out in `condition`.
#   type      "str" | "float"
#   allowed   enum of legal values, or None for free-form
#   maps_to   the models.Meta / cloud field this column feeds
#   notes     semantics, defaults, and the chirality rule
#
# CHIRALITY RULE (poly1_chir / poly2_chir) -- interpreted BY POLYMER IDENTITY:
#   - side-chain-chiral family (name starts with "C-", e.g. "C-PFBT"):
#       *_chir is a side-chain percentage ("100", "50") -> Meta.pN_pct
#   - main-chain-chiral family (bare backbone, e.g. "F8BT"):
#       *_chir is a handedness letter ("R"/"S")          -> Meta.pN_hand
#   - blank -> achiral
# manifest.build_metadata applies this mapping to build the index-aligned
# materials structure -- the raw chir string is never stored as-is.
# ---------------------------------------------------------------------------

# Solvent vocabulary lives in models.SOLVENTS; import here so the schema and
# the parser-side controlled vocabulary can never drift apart. models has no
# dependencies, so this import is always safe.
from models import SOLVENTS as _SOLVENTS

MANIFEST_COLUMNS = [
    {"name": "series", "required": True, "type": "str", "allowed": None,
     "maps_to": "Meta.series", "notes": "experiment series label, e.g. 'R1'"},
    {"name": "poly1", "required": True, "type": "str", "allowed": None,
     "maps_to": "Meta.p1_name (token rebuilt with poly1_chir)",
     "notes": "polymer name WITHOUT chirality suffix, e.g. 'C-PFBT', 'F8BT'"},
    {"name": "poly1_chir", "required": False, "type": "str", "allowed": None,
     "maps_to": "Meta.p1_pct or Meta.p1_hand (see CHIRALITY RULE)",
     "notes": "side-chain % for C- polymers, R/S for main-chain; blank = achiral"},
    {"name": "poly2", "required": False, "type": "str", "allowed": None,
     "maps_to": "Meta.p2_name",
     "notes": "blank or 'None' = single-component film"},
    {"name": "poly2_chir", "required": False, "type": "str", "allowed": None,
     "maps_to": "Meta.p2_pct or Meta.p2_hand (see CHIRALITY RULE)",
     "notes": "same rule as poly1_chir"},
    {"name": "ratio", "required": True, "type": "str", "allowed": None,
     "maps_to": "Meta.ratio",
     "notes": "'100' single-component, 'A:B' blend (stored as 'AxB' to match "
              "the filename convention); blend parts should sum to 100"},
    {"name": "conc_mg_ml", "required": True, "type": "float", "allowed": None,
     "maps_to": "Meta.conc", "notes": "solution concentration in mg/mL"},
    {"name": "solvent", "required": True, "type": "str",
     "allowed": list(_SOLVENTS),
     "maps_to": "Meta.solvent", "notes": "controlled vocabulary"},
    {"name": "speed_mm_s", "required": True, "type": "float", "allowed": None,
     "maps_to": "Meta.speed_mm_s", "notes": "print speed in mm/s"},
    {"name": "state", "required": True, "type": "str", "allowed": ["AP", "AN"],
     "maps_to": "Meta.film_state", "notes": "AP = as printed, AN = annealed"},
    {"name": "anneal_T_C", "required": False, "type": "float", "allowed": None,
     "condition": "REQUIRED iff state == 'AN'; must be blank when state == 'AP'",
     "maps_to": "Meta.anneal_temp", "notes": "anneal temperature in deg C"},
    {"name": "anneal_min", "required": False, "type": "float", "allowed": None,
     "maps_to": "Meta.anneal_time",
     "notes": "anneal time in minutes; blank defaults to 10 (AN films only)"},
    {"name": "dopant", "required": False, "type": "str", "allowed": None,
     "maps_to": "Meta.dopant", "notes": "blank = undoped"},
    {"name": "dopant_conc_mg_ml", "required": False, "type": "float",
     "allowed": None,
     "condition": "REQUIRED iff dopant is present",
     "maps_to": "Meta.dopant_conc", "notes": "dopant concentration in mg/mL"},
    {"name": "peak_gval", "required": False, "type": "float", "allowed": None,
     "maps_to": "scans.manifest_peak_g (provenance only)",
     "notes": "CHECKSUM ONLY -- cross-checked against the value computed from "
              "the spectra at ingest; the COMPUTED value is authoritative"},
    {"name": "peak_wl_nm", "required": False, "type": "float", "allowed": None,
     "maps_to": "scans.manifest_peak_wl (provenance only)",
     "notes": "CHECKSUM ONLY -- same rule as peak_gval"},
    {"name": "filename", "required": True, "type": "str", "allowed": None,
     "maps_to": "Meta.csv_path (resolved against source_folder)",
     "notes": "JOIN KEY ONLY -- keeps its original bench name; never parsed "
              "for metadata"},
    {"name": "source_folder", "required": True, "type": "str", "allowed": None,
     "maps_to": "scans.source_folder (provenance)",
     "notes": "folder the data file came from; also cross-checked against row "
              "values when the folder name encodes a condition"},
    {"name": "generated_at", "required": True, "type": "str", "allowed": None,
     "maps_to": "scans.manifest_generated_at (provenance)",
     "notes": "ISO timestamp, set by Cowork when the manifest was generated"},
    {"name": "manifest_version", "required": True, "type": "str",
     "allowed": None,
     "maps_to": "scans.manifest_version (provenance)",
     "notes": "schema version string; rows newer than manifest.MANIFEST_VERSION "
              "are rejected"},
]

# Derived-field cross-check tolerances (ingest step e). The manifest's
# peak_gval / peak_wl_nm are checksums; if present they must agree with the
# values computed from the loaded arrays within these bounds, else the record
# stages flagged as needs_review.
MANIFEST_PEAK_WL_TOL_NM = 3.0     # |manifest - computed| wavelength tolerance
MANIFEST_PEAK_G_REL_TOL = 0.05    # relative tolerance on the g-value...
MANIFEST_PEAK_G_ABS_TOL = 1e-3    # ...with an absolute floor near zero


# ---------------------------------------------------------------------------
# Folder-scan exclusions -- the SINGLE SOURCE OF TRUTH shared between manifest
# GENERATION (Cowork decides what NOT to put in a manifest) and import-time
# ORPHAN detection (import_window decides what NOT to flag as a CSV that's
# missing from the manifest). Pointing both sides at these constants means a
# file class that is intentionally absent from the manifest can never surface
# as a spurious orphan.
#
# A folder around one printed sample holds more than the processed spectral
# files: the raw per-orientation JASCO captures, their binary .jws sources,
# and macOS AppleDouble sidecars. None of those belong in the manifest, so
# none should be flagged as an orphan either.
# ---------------------------------------------------------------------------

# Raw orientation scans: the four (rotation, flip) JASCO captures that are
# combined into one processed spectrum. Matched against the filename STEM
# (extension stripped), so "<sample>_0_0-1.csv" and friends are recognized.
ORIENTATION_SCAN_SUFFIXES = ("_0_0-1", "_0_180-1", "_90_0-1", "_90_180-1")

# Markers that positively identify a PROCESSED spectral file by its name (the
# computed g-value token, in either the filename-convention 'gval=' form or
# the raw-export 'g-fac=' form). A folder CSV carrying one of these but absent
# from the manifest is a genuine miss worth flagging; one that doesn't is
# treated as incidental (notes, summaries, raw exports) and left alone.
PROCESSED_FILE_MARKERS = ("gval=", "g-fac=")
