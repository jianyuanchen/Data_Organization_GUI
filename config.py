"""Configuration loader for MongoDB Atlas (and other env-driven settings).

Reads `.env` from the project root via python-dotenv and exposes the MongoDB
connection settings as module-level constants. If `MONGODB_URI` is not set,
`MONGODB_URI` will be `None` and `mongo_configured()` returns False — callers
should check this before attempting to connect so the app still runs without
cloud configured.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_ENV_PATH = _HERE / ".env"
load_dotenv(_ENV_PATH)

MONGODB_URI: str | None = os.getenv("MONGODB_URI") or None
MONGODB_DB: str = os.getenv("MONGODB_DB", "cd_automation")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "samples")


def mongo_configured() -> bool:
    return bool(MONGODB_URI)


if not mongo_configured():
    logger.warning("MongoDB not configured — set MONGODB_URI in .env")


# ---------------------------------------------------------------------------
# Additive registry -- the queryable vocabulary behind the additive block.
#
# Films carry 0..MAX_ADDITIVES additives. Each additive's UNIT travels with its
# value (mg_ml, vol_pct, ...) and its ROLE (dopant, solvent_additive, ...) is a
# first-class field, so a vol% solvent additive can never land silently-wrong in
# what used to be a mg/mL `dopant_conc` column.
#
# The registry maps a CANONICAL additive name to its default role + unit + the
# raw aliases that should resolve to it. It is loaded from `additive_registry.json`
# (next to this file) at import so the "Add to registry" UI action can APPEND a
# new entry to JSON -- never rewrite this .py source -- and reload_additive_registry()
# picks it up for the next validation pass. resolve_additive() is the single
# name-resolver that replaces every hardcoded "MG -> Magic Green" special-case.
# ---------------------------------------------------------------------------

# Constrained enums. additive1_role / additive1_unit are validated against these
# the way `solvent` is validated against models.SOLVENTS -- a typo'd role or a
# bogus unit is caught at ingest instead of silently corrupting a query later.
ADDITIVE_ROLES = ["dopant", "solvent_additive", "plasticizer",
                  "nucleating_agent", "unknown"]
ADDITIVE_UNITS = ["mg_ml", "vol_pct", "mol_pct", "wt_pct"]

# Films are 0-1 additive today. To extend: bump this AND copy-paste the
# additive1_* block in MANIFEST_COLUMNS / models.Meta as additive2_* (the
# rest of the pipeline -- validator, cloud collapse -- already loops to
# MAX_ADDITIVES, so no other code changes).
MAX_ADDITIVES = 1

_ADDITIVE_REGISTRY_PATH = _HERE / "additive_registry.json"


def _load_additive_registry() -> dict:
    """Read additive_registry.json into a dict. A missing/corrupt file yields an
    empty registry (logged) rather than crashing import -- the app still runs,
    every additive just reads as UNKNOWN until the file is restored."""
    try:
        with open(_ADDITIVE_REGISTRY_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("additive_registry.json is not a JSON object")
        return data
    except FileNotFoundError:
        logger.warning("additive_registry.json not found at %s -- additives "
                       "will read as unknown", _ADDITIVE_REGISTRY_PATH)
        return {}
    except Exception as e:
        logger.warning("could not load additive_registry.json (%s: %s) -- "
                       "additives will read as unknown", type(e).__name__, e)
        return {}


ADDITIVE_REGISTRY: dict = _load_additive_registry()


def reload_additive_registry() -> dict:
    """Re-read additive_registry.json into the module-global ADDITIVE_REGISTRY.

    Reassigns the module global so resolve_additive() (defined here, so it reads
    this module's namespace at call time) immediately sees new entries -- the
    "Add to registry" action calls this after appending, and the next validation
    pass resolves the just-added additive. Returns the fresh registry.
    """
    global ADDITIVE_REGISTRY
    ADDITIVE_REGISTRY = _load_additive_registry()
    return ADDITIVE_REGISTRY


def resolve_additive(name: str):
    """Resolve a raw additive name to (canonical_name, registry_entry|None).

    Matches case-insensitively against registry KEYS and each entry's ALIASES,
    so "mg", "MG" and "magic green" all resolve to ("Magic Green", {...}).
    Returns (cleaned_name, None) for a non-blank name that matches nothing
    (an UNKNOWN additive) and ("", None) for a blank name. This is the ONE
    place name->canonical resolution happens.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return "", None
    low = cleaned.lower()
    for canonical, entry in ADDITIVE_REGISTRY.items():
        if low == canonical.lower():
            return canonical, entry
        for alias in (entry.get("aliases") or []):
            if low == str(alias).strip().lower():
                return canonical, entry
    return cleaned, None


def add_additive_to_registry(name: str, role: str, default_unit: str,
                             aliases=None) -> dict:
    """APPEND (or update) one additive entry in additive_registry.json, then
    reload ADDITIVE_REGISTRY.

    Writes JSON only -- never this .py source. The write is atomic (temp file in
    the same directory + os.replace) so an interrupted save never leaves a
    half-written, unparseable registry. `name` is stored as the canonical key;
    `aliases` records any raw token(s) seen so a future manifest using them
    resolves automatically. Returns the stored entry. Raises on bad
    role/unit/name so the caller can surface the problem instead of writing
    garbage into the vocabulary.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("additive name is blank")
    if role not in ADDITIVE_ROLES:
        raise ValueError(f"role '{role}' not in {ADDITIVE_ROLES}")
    if default_unit not in ADDITIVE_UNITS:
        raise ValueError(f"unit '{default_unit}' not in {ADDITIVE_UNITS}")

    # Start from the on-disk file (not the in-memory copy) so a concurrent edit
    # to the JSON isn't clobbered by a stale registry.
    data = _load_additive_registry()
    seen = {str(a).strip() for a in (data.get(name, {}).get("aliases") or [])}
    for a in (aliases or []):
        a = str(a).strip()
        if a and a.lower() != name.lower():
            seen.add(a)
    entry = {"role": role, "default_unit": default_unit,
             "aliases": sorted(seen)}
    data[name] = entry

    fd, tmp = tempfile.mkstemp(prefix=".additive-registry-", suffix=".json",
                               dir=str(_HERE))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, _ADDITIVE_REGISTRY_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    reload_additive_registry()
    return entry


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
    # ---- Additive block (registry-driven; replaces the old dopant columns) --
    # The unit travels WITH the value and the role is a queryable field, so a
    # vol% solvent additive can't masquerade as a mg/mL dopant. additive1_name
    # resolves via config.resolve_additive (registry keys + aliases); role/unit
    # auto-fill from the registry default when blank. Blank name == no additive
    # (all additive1_* cells must then be blank).
    #
    # To add a SECOND additive later: bump config.MAX_ADDITIVES and copy-paste
    # this 5-column block as additive2_name / additive2_role / additive2_conc /
    # additive2_unit / additive2_min (same maps_to with the index bumped). No
    # other code changes -- the validator and the cloud-doc collapse already
    # loop to MAX_ADDITIVES.
    {"name": "additive1_name", "required": False, "type": "str",
     "allowed": None, "maps_to": "Meta.additive1_name",
     "notes": "blank = no additive; resolves via additive registry "
              "(keys + aliases, e.g. 'MG' -> 'Magic Green')"},
    {"name": "additive1_role", "required": False, "type": "str",
     "allowed": ADDITIVE_ROLES, "maps_to": "Meta.additive1_role",
     "notes": "auto-filled from the registry default when blank; "
              "controlled vocabulary"},
    {"name": "additive1_conc", "required": False, "type": "float",
     "allowed": None,
     "condition": "expected when additive1_name is present (flagged if blank)",
     "maps_to": "Meta.additive1_conc",
     "notes": "additive concentration; UNIT is additive1_unit (not assumed)"},
    {"name": "additive1_unit", "required": False, "type": "str",
     "allowed": ADDITIVE_UNITS, "maps_to": "Meta.additive1_unit",
     "notes": "unit of additive1_conc; auto-filled from registry default when "
              "blank; controlled vocabulary"},
    {"name": "additive1_min", "required": False, "type": "float",
     "allowed": None, "maps_to": "Meta.additive1_min",
     "notes": "additive exposure/doping time in minutes (blank = unspecified)"},
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
