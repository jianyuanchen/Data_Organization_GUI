"""
Filename -> Meta parser. Strict positional decoding of the CSV filename
convention (see the module docstring of main_window.py for examples).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from models import (
    DEFAULT_ANNEAL_TIME,
    Meta,
    canon_path,
    classify_polymer,
)


def _derive_config(p1_chir, p2_chir, n_components):
    if n_components == 1:
        return "1-comp"
    def bucket(c):
        return "achiral" if c == "achiral" else "chiral"
    parts = sorted([bucket(p1_chir), bucket(p2_chir)])  # canonical order
    return f"{parts[0]}+{parts[1]}"


def parse_filename(path: str) -> Meta:
    """Parse a CSV path into Meta. Raises ValueError on malformed names.

    Convention:
        Series _ Poly1 _ Poly2 _ Ratio _ ConcSolvent _ Speed _ State
            [_ Temp if AN] _ Gval _ Wavelength

    The last two tokens are always Gval ('gval=0p047') then Wavelength ('500nm').
    Temp ('T###') appears only when State == 'AN'.
    """
    stem = Path(path).stem
    f = stem.split("_")
    if len(f) < 9:
        raise ValueError(f"Too few fields ({len(f)}) in '{stem}'")

    # Pull the last two tokens (Gval, Wavelength) off the end.
    wl_tok = f.pop()           # e.g. "500nm"
    g_tok  = f.pop()           # e.g. "gval=0p047"
    if not wl_tok.endswith("nm"):
        raise ValueError(f"Bad wavelength token '{wl_tok}'")
    # Accept integer ('500nm') and 'p'-decimal ('523p5nm' -> 523.5) forms,
    # matching how speed (v0p005) and g-value (gval=0p47) encode decimals.
    try:
        peak_wl = float(wl_tok[:-2].replace("p", "."))
    except ValueError:
        raise ValueError(f"Bad wavelength token '{wl_tok}'")
    if not g_tok.startswith("gval="):
        raise ValueError(f"Bad g-value token '{g_tok}'")
    try:
        peak_g = float(g_tok[len("gval="):].replace("p", "."))
    except ValueError:
        raise ValueError(f"Bad g-value token '{g_tok}'")

    # Remaining 7 tokens (AP) or 8 tokens (AN with T###).
    if len(f) not in (7, 8):
        raise ValueError(
            f"Unexpected field count in '{stem}' (got {len(f) + 2} total)")

    series, p1, p2, ratio, conc_solv, speed, state = f[0:7]
    temp_tok = f[7] if len(f) > 7 else None

    # conc + solvent, e.g. 20CB
    m = re.match(r"^(\d+)([A-Za-z]+)$", conc_solv)
    if not m:
        raise ValueError(f"Bad conc/solvent token '{conc_solv}'")
    conc, solvent = int(m.group(1)), m.group(2)

    # speed: v0p005 -> 0.005
    if not speed.startswith("v"):
        raise ValueError(f"Bad speed token '{speed}'")
    try:
        speed_val = float(speed[1:].replace("p", "."))
    except ValueError:
        raise ValueError(f"Bad speed token '{speed}'")

    if state not in ("AP", "AN"):
        raise ValueError(f"Bad film state '{state}' (expected AP/AN)")

    anneal_temp: Optional[int] = None
    if state == "AN":
        if temp_tok is None:
            raise ValueError("Annealed film missing T### token")
        if not temp_tok.startswith("T"):
            raise ValueError(f"Bad temp token '{temp_tok}'")
        try:
            anneal_temp = int(temp_tok[1:])
        except ValueError:
            raise ValueError(f"Bad temp token '{temp_tok}'")
    else:  # AP
        if temp_tok is not None:
            raise ValueError(
                f"AP film should have no temp token, got '{temp_tok}'")

    p1b, p1c, p1h, p1p = classify_polymer(p1)
    p2b, p2c, p2h, p2p = classify_polymer(p2)
    n = 1 if p2 == "None" else 2

    return Meta(
        csv_path=canon_path(path), series=series,
        p1_name=p1, p1_backbone=p1b, p1_chirality=p1c, p1_hand=p1h, p1_pct=p1p,
        p2_name=p2, p2_backbone=p2b, p2_chirality=p2c, p2_hand=p2h, p2_pct=p2p,
        n_components=n, config=_derive_config(p1c, p2c, n),
        ratio=ratio, conc=conc, solvent=solvent, film_state=state,
        speed_mm_s=speed_val, anneal_temp=anneal_temp,
        anneal_time=(DEFAULT_ANNEAL_TIME if state == "AN" else None),
        peak_g=peak_g, peak_wl=peak_wl,
    )
