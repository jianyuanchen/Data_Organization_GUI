"""
CD-shape classifier engine: UV-gated bisignate ladder detector.

PURE ANALYSIS. Operates on a record's spectral arrays (wavelength + CD +
UV-Vis) and returns a structured, fully-auditable classification result. It
NEVER reads or writes a file, never touches the DB, never mutates the input
arrays, and never modifies a cloud document -- the label is derived/computed on
demand. This is Phase A; the visual two-column review UI is Phase B.

Determination is NUMERIC, computed straight from the raw CSV arrays. The Phase B
plots are AUDIT-ONLY -- they visualize the same numbers and never feed the
decision. The data is used RAW: no smoothing, no resampling, no value edits. The
only reshaping is a stable sort by wavelength ASCENDING, because the raw files
run 700 -> 300 nm and the rest of the procedure assumes index order tracks
wavelength order. The same sort permutation is applied to all three arrays so
they stay aligned.

Domain context: a *ladder*-type CD signature is a genuine bisignate couplet that
sits under an evolved two-band UV envelope. The decision is a chain of ordered
gates (UV bands -> data-driven window -> CD couplet -> handedness -> evolution):

  * FLAGGED is reserved for data that cannot be evaluated at all -- empty/short
    arrays, NaN/inf, a flat UV with no dynamic range, or a required search
    window with no samples. A *failed gate is STAIRCASE, never FLAGGED.*
  * A single UV band, an absent CD couplet, or an under-evolved couplet are all
    real classifications -> STAIRCASE.
  * Both evolution gates passing on a genuine couplet -> LADDER (S or R by
    handedness).

The CD analysis window is data-driven by default, but a per-record manual_window
override (set visually in Phase B and persisted on the cloud doc) REPLACES it
when present; UV detection and the UV ratio gate are unaffected. The window
actually used is recorded on every result (window_used / window_source) so a
stored classification always carries the window it was computed under.

A separate "borderline" flag marks results whose UV ratio or lobe ratio sits
within +/-BORDERLINE_BAND of its threshold so a human knows to eyeball them; it
does NOT change the hard label.

All thresholds / windows below are PROVISIONAL: Jeff will tune the prominence
ratio `k` (UV_PROMINENCE_RATIO) and the right-edge fraction `f`
(BAND2_RIGHT_FRACTION) VISUALLY in Phase B. Because the persisted
auto_classification is a CACHE derived from these provisional constants, the
classify+write pass is kept idempotent / safely re-runnable after any retune.

Dependencies: numpy + scipy.signal.find_peaks (prominence-based peak detection
on the raw curve -- prominence, not absolute height, is what separates real
bands from ripple without smoothing). No Qt, no originpro. The cloud layer
(mongo_db) is imported lazily inside the runner so importing this module never
opens a connection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy.signal import find_peaks


# ===========================================================================
# TUNABLE CONSTANTS  (PROVISIONAL -- Jeff tunes k and f visually in Phase B)
# ===========================================================================
UV_PEAK_SEARCH_MIN = 300.0      # nm, region to look for the two UV bands
UV_PEAK_SEARCH_MAX = 600.0      # nm

# k: peak prominence threshold as a FRACTION of the record's own UV dynamic
# range (uv.max() - uv.min()) -> auto-scales per record so it is robust to
# magnitude (e.g. one record peaks ~3.0 while others sit ~0.2). Prominence (NOT
# absolute height) is what separates real bands from ripple without smoothing.
UV_PROMINENCE_RATIO = 0.05

# f: right edge = where band 2 has decayed to f of its peak-above-baseline.
BAND2_RIGHT_FRACTION = 0.15

# Far-red flat tail used to estimate the UV baseline.
BASELINE_TAIL_MIN = 650.0       # nm
BASELINE_TAIL_MAX = 700.0       # nm

UV_RATIO_THRESHOLD = 0.75       # peak2/peak1 evolution gate (baseline-subtracted)
LOBE_RATIO_THRESHOLD = 0.50     # smaller-lobe/larger-lobe evolution gate

# +/- band around each threshold -> flag for review (does NOT change the label).
BORDERLINE_BAND = 0.05

# Nominal CD analysis window. This is the FALLBACK SEED for the Phase B window
# slider when a record has NO manual_window -- it does NOT drive classification
# on its own: a record with no manual_window is classified under the DATA-DRIVEN
# window (inter-band valley -> band-2 decay). A per-record manual_window, when
# set, overrides the data-driven window. (Provisional.)
WINDOW_DEFAULT_MIN = 450.0      # nm
WINDOW_DEFAULT_MAX = 500.0      # nm

# ---- sanity floor (NOT a tuning knob) -------------------------------------
# Below this many finite, aligned samples a record is treated as "short" ->
# FLAGGED. find_peaks needs >= 3 points just to see one interior maximum; we
# keep a little headroom so a window restriction still leaves something usable.
MIN_USABLE_SAMPLES = 5


# ---- canonical labels ------------------------------------------------------
LABEL_LADDER_S = "Ladder-S"
LABEL_LADDER_R = "Ladder-R"
LABEL_STAIRCASE = "Staircase"
LABEL_FLAGGED = "Flagged"
LABELS = (LABEL_LADDER_S, LABEL_LADDER_R, LABEL_STAIRCASE, LABEL_FLAGGED)


def is_ladder(label: Optional[str]) -> bool:
    """True for either handedness of ladder (the LADDER column in Phase B)."""
    return label in (LABEL_LADDER_S, LABEL_LADDER_R)


def label_family(label: Optional[str]) -> str:
    """Collapse a hard label to a coarse family ('ladder'/'staircase'/'flagged')
    so a human 'ladder'/'staircase' override can be compared against either
    handedness of the auto label."""
    if is_ladder(label):
        return "ladder"
    if label == LABEL_STAIRCASE:
        return "staircase"
    return "flagged"


# ===========================================================================
# RESULT TYPES
# ===========================================================================
@dataclass
class Point:
    """A marked point on a curve: (wavelength, value). Either may be None when
    the point was not computable for this record."""
    wl: Optional[float] = None
    value: Optional[float] = None


@dataclass
class Gates:
    """Per-gate pass/fail booleans -- everything Phase B needs to show WHERE the
    chain stopped. A later gate is only meaningful once the earlier ones pass."""
    uv_band_detected: bool = False   # step 2: >= 2 prominent UV peaks
    cd_couplet: bool = False         # step 4: genuine bisignate CD couplet
    uv_ratio_pass: bool = False      # step 6: uv_two_peak_ratio >= threshold
    lobe_ratio_pass: bool = False    # step 6: lobe_ratio >= threshold


@dataclass
class ClassificationResult:
    label: str                                   # one of LABELS
    ladder_type: Optional[str] = None            # "S" | "R" | None
    bisignate: bool = False                       # genuine CD couplet present
    borderline: bool = False
    borderline_flags: list = field(default_factory=list)   # metric names in band
    audit_reason: str = ""                        # human-readable "why this label"

    # --- UV evidence ---
    uv_baseline: Optional[float] = None
    uv_peak1: Point = field(default_factory=Point)   # shorter-lambda band
    uv_peak2: Point = field(default_factory=Point)   # longer-lambda band
    interband_valley_lambda: Optional[float] = None  # = window LEFT edge (auto)
    window_left: Optional[float] = None
    window_right: Optional[float] = None
    window_source: Optional[str] = None              # "manual" | "data-driven"
    uv_two_peak_ratio: Optional[float] = None        # (p2-base)/(p1-base)

    # --- CD evidence (inside the data-driven window) ---
    cd_pos_lobe: Point = field(default_factory=Point)
    cd_neg_lobe: Point = field(default_factory=Point)
    lobe_ratio: Optional[float] = None               # min(|+|,|-|)/max(|+|,|-|)
    crossover_wavelength: Optional[float] = None      # CD zero-crossing (interp)

    gates: Gates = field(default_factory=Gates)
    reasons: list = field(default_factory=list)       # ordered audit notes

    @property
    def window(self) -> tuple:
        """The data-driven CD window as (left, right)."""
        return (self.window_left, self.window_right)

    def to_dict(self) -> dict:
        """Flat dict (nested Point/Gates expanded) for logging / JSON / audit."""
        return asdict(self)


# ===========================================================================
# ARRAY HELPERS  (numpy, NaN/inf-safe, never mutate the caller's data)
# ===========================================================================
def _to_float_array(seq) -> np.ndarray:
    """Best-effort conversion of an arbitrary sequence to a 1-D float array.

    Returns an empty array on any failure (None, ragged, non-numeric) so the
    caller can treat "no usable data" uniformly rather than catching errors.
    """
    if seq is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(seq, dtype=float)
    except (ValueError, TypeError):
        return np.array([], dtype=float)
    return arr.ravel()


def _prepare(wavelength, cd, uv):
    """Align (wavelength, cd, uv), drop any sample non-finite in ANY of the
    three, and STABLE-sort the survivors by wavelength ASCENDING.

    The three signals are evaluated on one shared, aligned grid (the CD couplet
    is read inside a UV-derived window), so cleaning is JOINT -- a NaN in one
    channel drops that sample from all three to keep them parallel. The caller's
    arrays are never mutated (masking + argsort both copy).

    Returns (wl, cd, uv) float arrays sorted by wavelength, or None when fewer
    than MIN_USABLE_SAMPLES finite aligned samples remain (-> caller FLAGS it).
    """
    wl = _to_float_array(wavelength)
    cdv = _to_float_array(cd)
    uvv = _to_float_array(uv)
    n = min(wl.size, cdv.size, uvv.size)
    if n < MIN_USABLE_SAMPLES:
        return None
    wl, cdv, uvv = wl[:n], cdv[:n], uvv[:n]
    mask = np.isfinite(wl) & np.isfinite(cdv) & np.isfinite(uvv)
    wl, cdv, uvv = wl[mask], cdv[mask], uvv[mask]
    if wl.size < MIN_USABLE_SAMPLES:
        return None
    order = np.argsort(wl, kind="stable")   # 700->300 raw files -> ascending
    return wl[order], cdv[order], uvv[order]


def _zero_crossing_wl(wl: np.ndarray, cd: np.ndarray,
                      lo: int, hi: int) -> Optional[float]:
    """Wavelength of the first genuine CD zero-crossing between indices lo..hi
    (inclusive), by LINEAR INTERPOLATION between the two bracketing samples.

    "Genuine" = a true sign change (cd[k] * cd[k+1] < 0), or a sample sitting
    exactly on zero. Reported at data resolution -- no smoothing is used to find
    it. Returns None when CD keeps one sign across the whole span.
    """
    for k in range(lo, hi):
        a, b = float(cd[k]), float(cd[k + 1])
        if a == 0.0:
            return float(wl[k])
        if a * b < 0.0:
            # fraction of the [k, k+1] step at which the straight line hits 0
            t = a / (a - b)
            return float(wl[k] + t * (wl[k + 1] - wl[k]))
    if float(cd[hi]) == 0.0:
        return float(wl[hi])
    return None


# ===========================================================================
# CLASSIFICATION CORE
# ===========================================================================
def _flagged(reason: str, **partial) -> ClassificationResult:
    """Build a FLAGGED result (un-computable data) carrying whatever partial
    evidence was gathered before the procedure had to stop."""
    return ClassificationResult(
        label=LABEL_FLAGGED, audit_reason=reason, reasons=[reason], **partial)


def _near(value: Optional[float], threshold: float) -> bool:
    """True if `value` sits within +/- BORDERLINE_BAND of `threshold`."""
    if value is None:
        return False
    return (threshold - BORDERLINE_BAND) <= value <= (threshold + BORDERLINE_BAND)


def _resolve_manual_window(manual_window) -> Optional[tuple]:
    """Normalize a per-record manual window override to (lo, hi) floats, or None
    when absent / malformed (-> caller uses the data-driven window instead).

    Accepts a {"min_nm", "max_nm"} dict (as persisted on the cloud doc) or a
    plain (lo, hi) pair. A non-numeric, non-finite, or non-increasing window is
    treated as "no override" rather than an error, so a bad stored value can
    never break classification -- it just falls back to the data-driven window.
    """
    if manual_window is None:
        return None
    if isinstance(manual_window, dict):
        lo, hi = manual_window.get("min_nm"), manual_window.get("max_nm")
    else:
        try:
            lo, hi = manual_window
        except (TypeError, ValueError):
            return None
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        return None
    return (lo, hi)


def classify_arrays(wavelength, cd, uv,
                    manual_window=None) -> ClassificationResult:
    """Classify ONE record from its raw spectral arrays. The primary entry point.

    `wavelength`, `cd`, `uv` are parallel numeric sequences (lists or numpy
    arrays). `g` is not needed for the decision. The procedure follows the
    ordered gates documented at the top of the module; each early exit records
    WHICH gate stopped it. Any un-computable input yields a FLAGGED result with
    a reason rather than raising.

    `manual_window` is an optional per-record override of the CD analysis window
    ({"min_nm", "max_nm"} dict or (lo, hi) pair). When present and valid it
    REPLACES the data-driven window (steps 4-7 read the CD couplet inside it);
    UV band detection and the UV peak2/peak1 ratio gate are UNAFFECTED. When
    absent/malformed the window is data-driven exactly as before.
    """
    prepared = _prepare(wavelength, cd, uv)
    if prepared is None:
        return _flagged("empty/short or non-finite arrays "
                        "(fewer than %d usable samples)" % MIN_USABLE_SAMPLES)
    wl, cdv, uvv = prepared

    # --- step 1: UV baseline from the far-red flat tail ---------------------
    tail = (wl >= BASELINE_TAIL_MIN) & (wl <= BASELINE_TAIL_MAX)
    if not tail.any():
        return _flagged(
            "no UV samples in baseline tail "
            f"[{BASELINE_TAIL_MIN:.0f}-{BASELINE_TAIL_MAX:.0f} nm]")
    baseline = float(np.mean(uvv[tail]))

    # Per-record UV dynamic range that scales the prominence threshold (k).
    uv_span = float(uvv.max() - uvv.min())
    if uv_span <= 0.0:
        return _flagged("UV signal is flat (no dynamic range)",
                        uv_baseline=baseline)

    # --- step 2: UV band detection (the "two peaks in one band" criterion) --
    win = (wl >= UV_PEAK_SEARCH_MIN) & (wl <= UV_PEAK_SEARCH_MAX)
    if not win.any():
        return _flagged(
            "no UV samples in search window "
            f"[{UV_PEAK_SEARCH_MIN:.0f}-{UV_PEAK_SEARCH_MAX:.0f} nm]",
            uv_baseline=baseline)
    wl_w, uv_w = wl[win], uvv[win]

    prominence = UV_PROMINENCE_RATIO * uv_span
    peak_idx, props = find_peaks(uv_w, prominence=prominence)

    if peak_idx.size < 2:
        # NOT bisignate: a single (or absent) UV band -> staircase, not flagged.
        # Still report the most prominent / largest absorbance as peak1 for audit.
        if peak_idx.size == 1:
            i1 = int(peak_idx[0])
        else:
            i1 = int(np.argmax(uv_w))
        return ClassificationResult(
            label=LABEL_STAIRCASE,
            audit_reason="single UV band",
            reasons=["single UV band"],
            uv_baseline=baseline,
            uv_peak1=Point(float(wl_w[i1]), float(uv_w[i1])),
            gates=Gates(uv_band_detected=False),
        )

    # Two MOST prominent bands, then ordered shorter-lambda -> longer-lambda.
    # (Index order already tracks wavelength because wl is sorted ascending.)
    proms = np.asarray(props["prominences"], dtype=float)
    top2 = peak_idx[np.argsort(proms)[::-1][:2]]
    i1, i2 = sorted(int(p) for p in top2)        # i1 = shorter lambda, i2 = longer
    peak1 = Point(float(wl_w[i1]), float(uv_w[i1]))
    peak2 = Point(float(wl_w[i2]), float(uv_w[i2]))

    # Baseline-subtracted band heights -- used by both the data-driven RIGHT
    # edge and the step-6 UV ratio gate, so compute once regardless of window.
    peak1_height = float(uv_w[i1]) - baseline
    peak2_height = float(uv_w[i2]) - baseline

    # --- step 3: resolve the CD analysis window [LEFT, RIGHT] ---------------
    # A per-record manual_window OVERRIDES the data-driven derivation; with none
    # the window is DATA-DRIVEN (inter-band valley LEFT, band-2 decay RIGHT) as
    # before. Either way the CD couplet (steps 4-7) is read inside [left, right].
    mw = _resolve_manual_window(manual_window)
    if mw is not None:
        left, right = mw
        interband_valley = None              # no data-driven valley in this mode
        window_source = "manual"
    else:
        # LEFT = inter-band valley: lambda of the UV minimum between the two
        # bands. This is what excludes band-1's CD feature (e.g. a ~345 nm dip).
        seg = uv_w[i1:i2 + 1]
        valley_i = i1 + int(np.argmin(seg))
        left = float(wl_w[valley_i])
        interband_valley = left
        # RIGHT = walking from peak2 toward LONGER lambda (in the full sorted
        # array, since the decay can run past UV_PEAK_SEARCH_MAX), the first
        # wavelength where (uv - baseline) has fallen to f of band-2's height.
        right_threshold = BAND2_RIGHT_FRACTION * peak2_height
        start = int(np.searchsorted(wl, peak2.wl, side="right"))
        right = None
        for k in range(start, wl.size):
            if (float(uvv[k]) - baseline) <= right_threshold:
                right = float(wl[k])
                break
        if right is None:
            right = float(wl[-1])            # never decayed -> clamp to far edge
        window_source = "data-driven"

    # --- CD samples inside the window (a missing window is un-computable) ----
    inwin = (wl >= left) & (wl <= right)
    common = dict(
        uv_baseline=baseline,
        uv_peak1=peak1, uv_peak2=peak2,
        interband_valley_lambda=interband_valley,
        window_left=left, window_right=right,
        window_source=window_source,
    )
    if not inwin.any():
        return _flagged(
            f"no CD samples in {window_source} window "
            f"[{left:.0f}-{right:.0f} nm]", **common)
    wl_in, cd_in = wl[inwin], cdv[inwin]

    # --- step 4: CD couplet inside the window -------------------------------
    pos_i = int(np.argmax(cd_in))
    neg_i = int(np.argmin(cd_in))
    pos_lobe = Point(float(wl_in[pos_i]), float(cd_in[pos_i]))
    neg_lobe = Point(float(wl_in[neg_i]), float(cd_in[neg_i]))
    common["cd_pos_lobe"] = pos_lobe
    common["cd_neg_lobe"] = neg_lobe

    opposite = (pos_lobe.value > 0.0) and (neg_lobe.value < 0.0)
    lo, hi = sorted((pos_i, neg_i))
    crossover = _zero_crossing_wl(wl_in, cd_in, lo, hi)
    couplet = opposite and (crossover is not None)
    common["crossover_wavelength"] = crossover

    # --- step 6 metrics (computed regardless so Phase B can show them) ------
    # peak1_height / peak2_height were computed above (baseline-subtracted).
    if peak1_height > 0.0:
        uv_two_peak_ratio = peak2_height / peak1_height
    else:
        uv_two_peak_ratio = None        # band-1 at/below baseline -> undefined
    common["uv_two_peak_ratio"] = uv_two_peak_ratio

    big = max(abs(pos_lobe.value), abs(neg_lobe.value))
    lobe_ratio = (min(abs(pos_lobe.value), abs(neg_lobe.value)) / big
                  if big > 0.0 else None)
    common["lobe_ratio"] = lobe_ratio

    if not couplet:
        reason = "no CD couplet"
        return ClassificationResult(
            label=LABEL_STAIRCASE, audit_reason=reason, reasons=[reason],
            gates=Gates(uv_band_detected=True, cd_couplet=False),
            **common)

    # --- step 5: handedness (only ASSIGNS R vs S; never disqualifies) -------
    # Scanning LONG -> SHORT lambda: positive lobe first (longer lambda) -> S;
    # negative lobe first (longer lambda) -> R. (R is theoretical/unseen.)
    ladder_type = "S" if pos_lobe.wl > neg_lobe.wl else "R"

    # --- step 6: evolution gates (BOTH must pass to confirm a ladder) -------
    reasons: list = []
    # Baseline-subtracted heights, consistent with the right-edge decay calc.
    uv_ratio_pass = (uv_two_peak_ratio is not None
                     and uv_two_peak_ratio >= UV_RATIO_THRESHOLD)
    if uv_two_peak_ratio is None:
        reasons.append("UV peak1 at/below baseline; peak2/peak1 undefined")
    elif not uv_ratio_pass:
        reasons.append("UV peak2/peak1 below 0.75")

    lobe_ratio_pass = (lobe_ratio is not None
                       and lobe_ratio >= LOBE_RATIO_THRESHOLD)
    if not lobe_ratio_pass:
        reasons.append("couplet under-evolved (<50%)")

    gates = Gates(uv_band_detected=True, cd_couplet=True,
                  uv_ratio_pass=uv_ratio_pass, lobe_ratio_pass=lobe_ratio_pass)

    # --- step 9: borderline (does NOT change the hard label) ----------------
    borderline_flags: list = []
    if _near(uv_two_peak_ratio, UV_RATIO_THRESHOLD):
        borderline_flags.append("uv_two_peak_ratio")
    if _near(lobe_ratio, LOBE_RATIO_THRESHOLD):
        borderline_flags.append("lobe_ratio")

    # --- step 7: result -----------------------------------------------------
    if uv_ratio_pass and lobe_ratio_pass:
        label = LABEL_LADDER_S if ladder_type == "S" else LABEL_LADDER_R
        reasons.insert(0, f"bisignate couplet confirmed; "
                          f"both evolution gates passed ({ladder_type}-ladder)")
    else:
        label = LABEL_STAIRCASE
        # couplet is real but not yet evolved -> staircase this round.
        reasons.insert(0, "genuine couplet but under-evolved")

    return ClassificationResult(
        label=label,
        ladder_type=ladder_type,
        bisignate=True,
        borderline=bool(borderline_flags),
        borderline_flags=borderline_flags,
        audit_reason="; ".join(reasons),
        gates=gates,
        reasons=reasons,
        **common,
    )


def classify_record(rec: dict) -> ClassificationResult:
    """Classify a cloud document dict using its embedded spectra.

    Reads the `wavelength` / `cd` / `uv` arrays stored by
    mongo_db.promote_records, plus an optional per-record `manual_window`
    override ({"min_nm", "max_nm"}) that, when present, replaces the data-driven
    CD analysis window. Read-only -- the document is never modified.
    """
    return classify_arrays(
        rec.get("wavelength"), rec.get("cd"), rec.get("uv"),
        manual_window=rec.get("manual_window"))


# ===========================================================================
# CLOUD-CACHE DOC HELPERS  (additive tags written back to each cloud doc)
# ===========================================================================
def thresholds_snapshot() -> dict:
    """The PROVISIONAL tuning constants as a plain dict, sourced straight from
    the module-level constants above (never duplicate the numbers elsewhere --
    read them from here). Embedded in every auto_classification cache doc so a
    later pass can detect that k/f or any threshold changed and the cache is
    stale, and re-run the (idempotent) classify+write pass.
    """
    return {
        "uv_peak_search_nm": [UV_PEAK_SEARCH_MIN, UV_PEAK_SEARCH_MAX],
        "uv_prominence_ratio": UV_PROMINENCE_RATIO,
        "band2_right_fraction": BAND2_RIGHT_FRACTION,
        "baseline_tail_nm": [BASELINE_TAIL_MIN, BASELINE_TAIL_MAX],
        "uv_ratio_threshold": UV_RATIO_THRESHOLD,
        "lobe_ratio_threshold": LOBE_RATIO_THRESHOLD,
        "borderline_band": BORDERLINE_BAND,
    }


def _pt(p: Point) -> list:
    """A Point as a plain [wl, value] list for the cache doc."""
    return [p.wl, p.value]


def auto_classification_doc(result: ClassificationResult) -> dict:
    """Compact, queryable cache document for a cloud doc's `auto_classification`
    field: the hard label + handedness + borderline flag + the key metrics
    (both ratios, peak / lobe / crossover locations) + per-gate booleans + the
    thresholds snapshot that produced them + a UTC timestamp.

    This is a REFRESHABLE CACHE derived from PROVISIONAL thresholds AND the
    window the record was classified under: re-run the (idempotent) classify+
    write pass whenever k/f or the constants above change, OR whenever a
    record's manual_window changes. `window_used` records the actual window so a
    stored result always carries the window it was computed under. Contains only
    derived scalars -- NEVER the spectral arrays.
    """
    return {
        "label": result.label,
        "ladder_type": result.ladder_type,
        "bisignate": result.bisignate,
        "borderline": result.borderline,
        "borderline_flags": list(result.borderline_flags),
        "audit_reason": result.audit_reason,
        "uv_baseline": result.uv_baseline,
        "uv_peak1": _pt(result.uv_peak1),
        "uv_peak2": _pt(result.uv_peak2),
        "interband_valley_lambda": result.interband_valley_lambda,
        "window_used": [result.window_left, result.window_right],
        "window_source": result.window_source,
        "uv_two_peak_ratio": result.uv_two_peak_ratio,
        "cd_pos_lobe": _pt(result.cd_pos_lobe),
        "cd_neg_lobe": _pt(result.cd_neg_lobe),
        "lobe_ratio": result.lobe_ratio,
        "crossover_wavelength": result.crossover_wavelength,
        "gates": asdict(result.gates),
        "reasons": list(result.reasons),
        "thresholds": thresholds_snapshot(),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# ===========================================================================
# CLOUD RUNNER + CONSOLE SUMMARY  (Phase A inspection path)
# ===========================================================================
def _f(x: Optional[float], nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _wl(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0f}nm"


def _record_label(rec: dict, index: int) -> str:
    """Compact human identifier for a per-record log line. Prefers the stored
    filename; falls back to series + polymer names.
    """
    ident = rec.get("filename") or rec.get("filename_key")
    if not ident:
        series = rec.get("series") or "?"
        p1 = rec.get("p1_name") or "?"
        p2 = rec.get("p2_name")
        poly = p1 if (not p2 or p2 == "None") else f"{p1}+{p2}"
        ident = f"{series} {poly}"
    return f"{index + 1:>3}. {ident}"


def _format_metrics(res: ClassificationResult) -> str:
    g = res.gates
    uv = (f"UV[band={'Y' if g.uv_band_detected else 'N'} "
          f"p1={_f(res.uv_peak1.value)}@{_wl(res.uv_peak1.wl)} "
          f"p2={_f(res.uv_peak2.value)}@{_wl(res.uv_peak2.wl)} "
          f"ratio={_f(res.uv_two_peak_ratio, 2)} "
          f"pass={'Y' if g.uv_ratio_pass else 'N'}]")
    cd = (f"CD[couplet={'Y' if g.cd_couplet else 'N'} "
          f"+={_f(res.cd_pos_lobe.value)}@{_wl(res.cd_pos_lobe.wl)} "
          f"-={_f(res.cd_neg_lobe.value)}@{_wl(res.cd_neg_lobe.wl)} "
          f"lobe={_f(res.lobe_ratio, 2)} "
          f"pass={'Y' if g.lobe_ratio_pass else 'N'} "
          f"x0={_wl(res.crossover_wavelength)}]")
    src = res.window_source[:4] if res.window_source else "n/a"
    win = f"win[{src}]=[{_wl(res.window_left)},{_wl(res.window_right)}]"
    return f"{uv}  {cd}  {win}"


def _borderline_text(res: ClassificationResult) -> str:
    if not res.borderline:
        return "-"
    return "BORDERLINE(" + ",".join(res.borderline_flags) + ")"


def _is_verified(rec: dict) -> bool:
    try:
        return int(rec.get("verified") or 0) == 1
    except (TypeError, ValueError):
        return False


def print_tally(results: list, log=print) -> dict:
    """Print and return the summary tally over a list of (rec, result) pairs."""
    total = len(results)
    n_ladder_s = sum(1 for _, r in results if r.label == LABEL_LADDER_S)
    n_ladder_r = sum(1 for _, r in results if r.label == LABEL_LADDER_R)
    n_ladder = n_ladder_s + n_ladder_r
    n_stair = sum(1 for _, r in results if r.label == LABEL_STAIRCASE)
    n_flag = sum(1 for _, r in results if r.label == LABEL_FLAGGED)
    n_border = sum(1 for _, r in results if r.borderline)
    classified = n_ladder + n_stair

    log("")
    log("=" * 64)
    log("CLASSIFICATION SUMMARY")
    log(f"  total records          : {total}")
    log(f"  ladder (S / R)         : {n_ladder}  ({n_ladder_s} S / {n_ladder_r} R)")
    log(f"  staircase              : {n_stair}")
    log(f"  flagged                : {n_flag}")
    log(f"  borderline             : {n_border}  (marked for human review)")
    if total:
        log(f"  ladder % (of all)      : {100.0 * n_ladder / total:.1f}%")
    if classified:
        log(f"  ladder % (of classified): "
            f"{100.0 * n_ladder / classified:.1f}%")
    log("=" * 64)

    return {
        "total": total, "ladder": n_ladder, "ladder_s": n_ladder_s,
        "ladder_r": n_ladder_r, "staircase": n_stair, "flagged": n_flag,
        "borderline": n_border,
        "ladder_pct": (100.0 * n_ladder / total) if total else 0.0,
    }


def run_cloud_classification(query: dict | None = None, log=print,
                             verified_only: bool = True) -> list:
    """Classify the CLOUD (verified) records and print each + a summary tally.

    Fetches read-only via mongo_db.fetch_records (lazy-imported so importing
    this module never opens a connection). Cloud documents are confirmed +
    verified by construction (the promote gate), so `verified_only` is a
    defensive belt-and-suspenders filter on top. Every record is classified
    purely from its embedded arrays; nothing is read from disk and NOTHING is
    modified -- local or cloud.

    Returns the list of (record, ClassificationResult) pairs so a caller (or
    Phase B's UI) can reuse the computed results without re-running.
    """
    try:
        from mongo_db import fetch_records
    except Exception as e:                       # pragma: no cover - import guard
        log(f"Cannot import cloud layer: {type(e).__name__}: {e}")
        return []

    records = fetch_records(query or {}, log=log)

    if verified_only:
        kept = [r for r in records if _is_verified(r)]
        if len(kept) != len(records):
            log(f"Filtered to {len(kept)} verified record(s) "
                f"({len(records) - len(kept)} unverified skipped).")
        records = kept

    if not records:
        log("No cloud records to classify.")
        return []

    log(f"\nClassifying {len(records)} cloud record(s)...\n")
    results: list = []
    for i, rec in enumerate(records):
        res = classify_record(rec)
        results.append((rec, res))
        log(f"{_record_label(rec, i)}  {res.label.upper():<10} "
            f"{_borderline_text(res):<24} {_format_metrics(res)}")
        if res.audit_reason:
            log(f"        reason: {res.audit_reason}")

    print_tally(results, log=log)
    return results


if __name__ == "__main__":
    # Direct run: classify whatever is in the configured Atlas collection and
    # print the per-record lines + tally. Degrades to a clear message (no
    # crash) when the cloud is unconfigured/unreachable.
    run_cloud_classification()
