"""
Manifest ingest front-end: read + validate an AI-generated manifest.csv and
build Meta objects from its rows. Peer of parser.py -- the regex filename
parser stays as the legacy/fallback path; this module is the second front-end
that feeds the SAME SQLite staging records.

Scope guard: NO spectral-array loading happens here. Arrays are read exactly
once, at the Confirm -> SQLite step (database.ingest_manifest_record). This
module is metadata-only: csv.DictReader in, validated row dicts and Meta
objects out.

The column contract lives in config.MANIFEST_COLUMNS (single source of truth,
shared with the Cowork prompt that generates manifests). The one subtle rule
is chirality: poly1_chir / poly2_chir are interpreted BY POLYMER IDENTITY --
a side-chain percentage for "C-" polymers, an R/S handedness for bare
backbones -- and folded into the same polymer tokens the filename convention
uses, so classify_polymer produces identical structured fields on both paths.
"""
from __future__ import annotations

import csv
import os
import re
import tempfile
from typing import Optional

from config import MANIFEST_COLUMNS
from models import (
    DEFAULT_ANNEAL_TIME,
    Meta,
    SOLVENTS,
    canon_path,
    classify_polymer,
)
from parser import _derive_config

# Highest manifest schema version this build understands. Rows whose
# manifest_version is NEWER are rejected ("error") rather than half-parsed;
# older versions are accepted (forward-only schema, like the SQLite layer).
MANIFEST_VERSION = "1.0"

_REQUIRED = [c["name"] for c in MANIFEST_COLUMNS if c["required"]]
_FLOAT_FIELDS = {c["name"] for c in MANIFEST_COLUMNS if c["type"] == "float"}
# Column-level enums, declared once in config. allowed:None columns (free-form,
# or polymorphic like poly1_chir/poly2_chir whose R/S vs % rule lives in
# _polymer_token) are skipped, so the generic enum step never touches them.
_ALLOWED = {c["name"]: c["allowed"]
            for c in MANIFEST_COLUMNS if c.get("allowed")}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> list[dict]:
    """Read manifest.csv into a list of raw row dicts.

    csv.DictReader keyed by the header row; keys and values are stripped
    (utf-8-sig eats a BOM from Excel exports). Rows that are entirely blank
    are dropped. No validation here -- that's validate_row, per row, so one
    bad line never blocks the rest of the manifest.
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        for raw in csv.DictReader(f):
            row = {(k or "").strip(): (v or "").strip()
                   for k, v in raw.items() if k is not None}
            if any(v for v in row.values()):
                rows.append(row)
    return rows


def save_manifest(path: str, rows: list[dict]) -> None:
    """Atomically write `rows` (raw manifest dicts) back to `path` as CSV.

    Column order follows config.MANIFEST_COLUMNS exactly, with the header row
    preserved. Any extra keys present in the data (e.g. a column a newer
    generator added) are appended AFTER the canonical columns so a hand-edit
    never silently drops data.

    ATOMIC: the rows are written to a temp file in the SAME directory, flushed
    + fsync'd, then os.replace()'d over the original. os.replace is atomic on
    a single filesystem, so an interruption leaves the original manifest fully
    intact -- never a half-written file. The temp file is removed on any error.

    REGENERATION TRADEOFF (intentional, no guard yet): once a manifest is
    hand-edited and saved here, re-running Cowork over the same folder will
    OVERWRITE these edits -- the manifest is regenerated from scratch. We are
    deliberately NOT adding a `manually_edited` guard column at this stage;
    that protection is deferred.
    """
    canonical = [c["name"] for c in MANIFEST_COLUMNS]
    seen = set(canonical)
    extras: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extras.append(key)
    fieldnames = canonical + extras

    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".manifest-", suffix=".csv",
                               dir=directory)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {k: ("" if row.get(k) is None else row.get(k))
                     for k in fieldnames})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Never leave a stray temp file if the write or replace failed.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# per-row validation
# ---------------------------------------------------------------------------

def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _none_token(v) -> bool:
    """Blank or the literal 'None' -- both mean 'no second component'."""
    return _blank(v) or str(v).strip().lower() == "none"


def _num(v):
    """float(v), collapsed to int when integral so values land in SQLite the
    same way the regex path stores them (conc/anneal_temp are INTEGER columns).
    Raises ValueError on malformed input."""
    f = float(str(v).strip())
    return int(f) if f == int(f) else f


def _check_enum(field: str, value: str, errors: list, warnings: list) -> str:
    """Validate a NON-BLANK value against the column's config `allowed` list,
    case-insensitively. Returns the canonical cased member:

      - exact member            -> returned unchanged
      - case-variant of a member-> AUTO-CORRECTED: warn + return the cased
                                    member (mirrors the old solvent behavior)
      - matches nothing         -> error appended, original value returned

    Blank values and allowed:None fields are the caller's concern (the
    _REQUIRED loop owns required-ness); this never errors on them.
    """
    allowed = _ALLOWED.get(field)
    if not allowed or not value:
        return value
    if value in allowed:
        return value
    canon = {a.lower(): a for a in allowed}.get(value.lower())
    if canon is not None:
        warnings.append(f"{field} '{value}' normalized to '{canon}'")
        return canon
    errors.append(
        f"unknown {field} '{value}' (expected one of {', '.join(allowed)})")
    return value


def _version_tuple(v: str) -> tuple:
    parts = re.findall(r"\d+", v or "")
    return tuple(int(p) for p in parts) if parts else (0,)


def _polymer_token(name: str, chir: str, label: str,
                   errors: list, warnings: list) -> Optional[str]:
    """Apply the CHIRALITY RULE and rebuild the filename-style polymer token.

    The token is what classify_polymer expects, so both ingest paths produce
    byte-identical structured polymer fields:
        C-PFBT + '100' -> 'C-PFBT100'   (side-chain pct)
        F8BT   + 'R'   -> 'R-F8BT'      (main-chain handedness)
        F8BT   + ''    -> 'F8BT'        (achiral)
    Returns None when the combination is unbuildable (an `errors` entry is
    appended); suspicious-but-buildable combinations append to `warnings`.
    """
    name = (name or "").strip()
    chir = (chir or "").strip()
    if not name:
        return None

    # Token already fully encoded in the name (Cowork shouldn't do this, but
    # don't double-apply the rule if it does).
    if re.match(r"^[RS]-", name):
        if chir:
            warnings.append(
                f"{label}: name '{name}' already embeds handedness; "
                f"chir column '{chir}' ignored")
        return name
    if re.match(r"^C-[A-Za-z0-9]*?\d+$", name):
        if chir:
            warnings.append(
                f"{label}: name '{name}' already embeds a side-chain %; "
                f"chir column '{chir}' ignored")
        else:
            warnings.append(
                f"{label}: side-chain % embedded in name '{name}' instead of "
                f"the chir column")
        return name

    if name.startswith("C-"):
        # Side-chain-chiral family: chir is a percentage.
        if not chir:
            warnings.append(
                f"{label}: blank chirality on usually-chiral polymer "
                f"'{name}' (treated as achiral)")
            return name
        try:
            pct = int(float(chir))
        except ValueError:
            errors.append(
                f"{label}: malformed side-chain % '{chir}' for '{name}'")
            return None
        if not (0 < pct <= 100):
            warnings.append(
                f"{label}: side-chain % {pct} outside (0, 100] for '{name}'")
        return f"{name}{pct}"

    # Main-chain family (bare backbone): chir is a handedness letter.
    if not chir:
        return name                       # achiral, e.g. plain F8BT
    if chir.upper() in ("R", "S"):
        return f"{chir.upper()}-{name}"
    errors.append(
        f"{label}: chirality '{chir}' invalid for main-chain polymer "
        f"'{name}' (expected R/S)")
    return None


# Folder-name condition tokens for the metadata-level cross-checks. DCB must
# be tried before CB so '20DCB' never half-matches as '0DCB'... i.e. as CB.
_FOLDER_CONC_SOLV = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:[p.]\d+)?)(DCB|CB|Tol)(?![A-Za-z])", re.I)
_FOLDER_RATIO = re.compile(r"\((\d+)\s*[-:x]\s*(\d+)\)", re.I)
_FOLDER_TEMP = re.compile(r"(?<![A-Za-z0-9])T(\d{2,3})(?![0-9])")
_SOLVENT_CASE = {s.lower(): s for s in SOLVENTS}


def _folder_cross_checks(parsed: dict) -> list[str]:
    """Compare conditions ENCODED IN the source_folder string against the row.

    Only fields the folder actually encodes are checked, and a field is only
    flagged when NO token in the folder agrees with the row (folders holding
    sibling condition sets legitimately mention several). Conflicts are
    surfaced for the human -- never auto-corrected.
    """
    folder = parsed.get("source_folder") or ""
    out: list[str] = []

    cs = _FOLDER_CONC_SOLV.findall(folder)
    if cs:
        solvents = {_SOLVENT_CASE.get(s.lower(), s) for _, s in cs}
        concs = {float(c.replace("p", ".")) for c, _ in cs}
        if parsed["solvent"] not in solvents:
            out.append(
                f"folder/row conflict: solvent (folder says "
                f"{'/'.join(sorted(solvents))}, row says {parsed['solvent']})")
        if float(parsed["conc"]) not in concs:
            out.append(
                f"folder/row conflict: conc_mg_ml (folder says "
                f"{'/'.join(str(c) for c in sorted(concs))}, "
                f"row says {parsed['conc']})")

    ratios = _FOLDER_RATIO.findall(folder)
    if ratios:
        folder_ratios = {f"{a}x{b}" for a, b in ratios}
        if parsed["ratio"] not in folder_ratios:
            out.append(
                f"folder/row conflict: ratio (folder says "
                f"{'/'.join(sorted(folder_ratios))}, row says "
                f"{parsed['ratio']})")

    temps = _FOLDER_TEMP.findall(folder)
    if temps and parsed.get("anneal_temp") is not None:
        if float(parsed["anneal_temp"]) not in {float(t) for t in temps}:
            out.append(
                f"folder/row conflict: anneal_T_C (folder says "
                f"{'/'.join(sorted(temps))}, row says "
                f"{parsed['anneal_temp']})")
    return out


def validate_row(row: dict) -> tuple[str, list[str], Optional[dict]]:
    """Validate one raw manifest row. Returns (status, reasons, parsed).

        status: "ok" | "needs_review" | "error"
        reasons: every error and warning message, errors first
        parsed:  normalized/typed dict ready for build_metadata, or None
                 when status == "error" (unrecoverable)

    "error" = unbuildable: missing required field, unknown solvent/state,
    malformed number, AN with no anneal_T_C, dopant with no dopant_conc,
    manifest_version newer than supported.
    "needs_review" = buildable but suspicious: folder/row conflict, blank
    chirality on a usually-chiral polymer, ratio/poly2 inconsistency, etc.
    File presence is NOT checked here (import_window owns that), and no
    arrays are read.
    """
    errors: list[str] = []
    warnings: list[str] = []
    get = lambda k: (row.get(k) or "").strip()

    # Schema version gate first -- a newer schema may have changed any other
    # column's semantics, so nothing else is trustworthy.
    ver = get("manifest_version")
    if ver and _version_tuple(ver) > _version_tuple(MANIFEST_VERSION):
        return ("error",
                [f"manifest_version '{ver}' is newer than supported "
                 f"'{MANIFEST_VERSION}' -- update the app or regenerate the "
                 f"manifest"],
                None)

    for name in _REQUIRED:
        if _blank(row.get(name)):
            errors.append(f"missing required field: {name}")

    # Typed fields. Parse failures are unrecoverable for required numbers;
    # optional numbers only error when non-blank and malformed.
    nums: dict = {}
    for name in _FLOAT_FIELDS:
        v = row.get(name)
        if _blank(v):
            nums[name] = None
            continue
        try:
            nums[name] = _num(v)
        except ValueError:
            errors.append(f"malformed number: {name}='{str(v).strip()}'")
            nums[name] = None

    # Column-level enums are validated declaratively from config `allowed`
    # (see _check_enum / _ALLOWED). solvent auto-corrects case; state keeps its
    # .upper() coercion BEFORE the check, so parsed["state"] and the annealing
    # conditionals below both see the uppercased value (an already-canonical
    # value matches exactly, so no spurious normalization warning fires).
    solvent = _check_enum("solvent", get("solvent"), errors, warnings)

    state = get("state").upper()
    _check_enum("state", state, errors, warnings)

    # Conditional requirements around annealing.
    anneal_t = nums.get("anneal_T_C")
    anneal_min = nums.get("anneal_min")
    if state == "AN":
        # Only complain about blankness -- a malformed anneal_T_C already
        # produced its own "malformed number" error above.
        if anneal_t is None and _blank(row.get("anneal_T_C")):
            errors.append("state is AN but anneal_T_C is blank")
        if anneal_min is None:
            anneal_min = DEFAULT_ANNEAL_TIME
    elif state == "AP":
        if anneal_t is not None:
            errors.append(
                f"state is AP but anneal_T_C='{anneal_t}' (must be blank)")
        if anneal_min is not None:
            warnings.append(
                f"state is AP but anneal_min='{anneal_min}' (ignored)")
        anneal_min = None

    # Dopant pairing.
    dopant = get("dopant") if not _none_token(row.get("dopant")) else ""
    dopant_conc = nums.get("dopant_conc_mg_ml")
    if dopant and dopant_conc is None:
        errors.append("dopant present but dopant_conc_mg_ml is blank")
    if not dopant and dopant_conc is not None:
        warnings.append(
            f"dopant_conc_mg_ml='{dopant_conc}' given without a dopant")

    # Polymers + chirality rule.
    p1_token = _polymer_token(get("poly1"), get("poly1_chir"), "poly1",
                              errors, warnings)
    two_comp = not _none_token(row.get("poly2"))
    p2_token = (_polymer_token(get("poly2"), get("poly2_chir"), "poly2",
                               errors, warnings)
                if two_comp else "None")
    if not two_comp and get("poly2_chir"):
        warnings.append(
            f"poly2_chir='{get('poly2_chir')}' given without a poly2")

    # Ratio: '100' single-component, 'A:B' blend (normalized to the filename
    # convention's 'AxB' so staging values match the regex path).
    ratio_raw = get("ratio")
    ratio = None
    if ratio_raw:
        m = re.match(r"^(\d+)\s*[:xX/-]\s*(\d+)$", ratio_raw)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            ratio = f"{a}x{b}"
            if a + b != 100:
                warnings.append(
                    f"blend ratio {a}:{b} does not sum to 100")
            if not two_comp:
                warnings.append(
                    f"blend ratio '{ratio_raw}' on a single-component row")
        elif re.match(r"^\d+$", ratio_raw):
            ratio = ratio_raw
            if ratio != "100":
                warnings.append(
                    f"single-value ratio '{ratio_raw}' (expected '100')")
            if two_comp:
                warnings.append(
                    f"ratio '{ratio_raw}' on a two-component row "
                    f"(expected 'A:B')")
        else:
            errors.append(f"malformed ratio '{ratio_raw}'")

    if errors:
        return ("error", errors + warnings, None)

    parsed = {
        "series": get("series"),
        "p1_token": p1_token,
        "p2_token": p2_token,
        "ratio": ratio,
        "conc": nums["conc_mg_ml"],
        "solvent": solvent,
        "speed": nums["speed_mm_s"],
        "state": state,
        "anneal_temp": anneal_t,
        "anneal_min": anneal_min,
        "dopant": dopant or None,
        "dopant_conc": dopant_conc,
        "peak_gval": nums.get("peak_gval"),
        "peak_wl_nm": nums.get("peak_wl_nm"),
        "filename": get("filename"),
        "source_folder": get("source_folder"),
        "generated_at": get("generated_at"),
        "manifest_version": ver,
    }
    warnings.extend(_folder_cross_checks(parsed))

    return ("needs_review" if warnings else "ok", warnings, parsed)


# ---------------------------------------------------------------------------
# building models objects
# ---------------------------------------------------------------------------

def materials(parsed: dict) -> list[dict]:
    """Index-aligned materials array with the chirality rule applied and the
    ratio split alongside: [{polymer, ratio_pct, sidechain_pct | handedness},
    ...]. Future cloud-document shape; build_metadata derives the SAME facts
    into Meta's flat p1_/p2_ fields for SQLite staging.
    """
    out = []
    tokens = [parsed["p1_token"]]
    if parsed["p2_token"] != "None":
        tokens.append(parsed["p2_token"])
    splits = ([int(p) for p in parsed["ratio"].split("x")]
              if "x" in parsed["ratio"] else [100])
    for i, token in enumerate(tokens):
        backbone, chirality, hand, pct = classify_polymer(token)
        mat = {"polymer": token,
               "backbone": backbone,
               "chirality": chirality,
               "ratio_pct": splits[i] if i < len(splits) else None}
        if chirality == "side-chain":
            mat["sidechain_pct"] = pct
        elif chirality == "main-chain":
            mat["handedness"] = hand
        out.append(mat)
    return out


def build_metadata(parsed: dict, csv_path: Optional[str] = None) -> Meta:
    """Build the staging Meta from a validated row (validate_row's `parsed`).

    Goes through the SAME classify_polymer / _derive_config helpers the regex
    path uses, so the structured fields are identical for equivalent inputs.
    peak_g / peak_wl carry the manifest's CHECKSUM values for now -- the
    ingest step replaces them with values computed from the spectra (the
    spectrum is the source of truth) and keeps these as provenance only.

    csv_path defaults to source_folder/filename; import_window passes the
    path it actually resolved (e.g. against a dropped folder).
    """
    if csv_path is None:
        csv_path = os.path.join(parsed["source_folder"], parsed["filename"])
    p1, p2 = parsed["p1_token"], parsed["p2_token"]
    p1b, p1c, p1h, p1p = classify_polymer(p1)
    p2b, p2c, p2h, p2p = classify_polymer(p2)
    n = 1 if p2 == "None" else 2

    return Meta(
        csv_path=canon_path(csv_path), series=parsed["series"],
        p1_name=p1, p1_backbone=p1b, p1_chirality=p1c, p1_hand=p1h, p1_pct=p1p,
        p2_name=p2, p2_backbone=p2b, p2_chirality=p2c, p2_hand=p2h, p2_pct=p2p,
        n_components=n, config=_derive_config(p1c, p2c, n),
        ratio=parsed["ratio"], conc=parsed["conc"], solvent=parsed["solvent"],
        film_state=parsed["state"], speed_mm_s=parsed["speed"],
        anneal_temp=parsed["anneal_temp"],
        anneal_time=(parsed["anneal_min"] if parsed["state"] == "AN" else None),
        peak_g=parsed["peak_gval"], peak_wl=parsed["peak_wl_nm"],
        dopant=parsed["dopant"], dopant_conc=parsed["dopant_conc"],
    )
