"""
CD-shape classifier engine: UV-gated bisignate ladder detector.

PURE ANALYSIS. Operates on a record's spectral arrays (wavelength + CD +
UV-Vis) and returns a structured, fully-auditable METRICS result. It NEVER reads
or writes a file, never touches the DB, never mutates the input arrays, and never
modifies a cloud document -- the metrics are computed on demand. This is Phase A;
the visual worklist + review UI is Phase B.

This module is a MEASURER, not a judge: it computes the objective numbers a human
needs, but emits NO Ladder/Staircase verdict. The final category is assigned by a
human in Phase B (human_classification) and is the SINGLE source of truth for
sorting and stats. The metrics are computed RAW (no smoothing, no resampling, no
value edits); the only reshaping is a stable sort by wavelength ASCENDING, because
the raw files run 700 -> 300 nm and the rest of the procedure assumes index order
tracks wavelength order. The same sort permutation is applied to all three arrays
so they stay aligned.

The metrics are produced through an ordered chain of gates
(UV bands -> CD window -> CD couplet -> evolution ratios):

  * UV bands: are there two prominent UV peaks? (peak1/peak2 + locations)
  * uv_peak_ratio: RAW peak2/peak1 (longer-lambda over shorter-lambda).
  * CD couplet: a genuine bisignate couplet inside the window (pos/neg lobe).
  * cd_peak_ratio: smaller-lobe/larger-lobe.
  * two gate pass/fail booleans (uv_ratio_pass / lobe_ratio_pass) reported as
    OBJECTIVE evidence -- they do NOT collapse into a verdict here.

Un-computable data (empty/short arrays, NaN/inf, a flat UV with no dynamic range,
or a window with no samples) yields a result with empty metrics and a note rather
than raising; the human still sees the record (grey/unreviewed in Phase B).

The CD analysis window is the per-record manual_window when one is set (chosen
visually in Phase B and persisted on the cloud doc), else the global default
[WINDOW_DEFAULT_MIN, WINDOW_DEFAULT_MAX]. There is NO algorithmic re-derivation:
the window the CD couplet is read inside is exactly one of those two, and it is
recorded on every result (window_used / window_source = "manual" | "default") so
a stored metrics result always carries the window it was computed under.

A "borderline" flag marks results whose UV ratio or lobe ratio sits within
+/-BORDERLINE_BAND of its threshold so a human knows to eyeball them.

All thresholds / windows below are PROVISIONAL: Jeff will tune the prominence
ratio `k` (UV_PROMINENCE_RATIO) and the default window VISUALLY in Phase B.
Because the persisted computed_metrics block is a CACHE derived from these
provisional constants AND the window used, the measure+write pass is kept
idempotent / safely re-runnable after any retune.

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
# TUNABLE CONSTANTS  (PROVISIONAL -- Jeff tunes k visually in Phase B)
# ===========================================================================
UV_PEAK_SEARCH_MIN = 300.0      # nm, region to look for the two UV bands
UV_PEAK_SEARCH_MAX = 600.0      # nm

# k: peak prominence threshold as a FRACTION of the record's own UV dynamic
# range (uv.max() - uv.min()) -> auto-scales per record so it is robust to
# magnitude (e.g. one record peaks ~3.0 while others sit ~0.2). Prominence (NOT
# absolute height) is what separates real bands from ripple without smoothing.
UV_PROMINENCE_RATIO = 0.05

# Far-red flat tail used to estimate the UV baseline (DISPLAY-ONLY now -- it no
# longer feeds the ratio gate or any window edge).
BASELINE_TAIL_MIN = 650.0       # nm
BASELINE_TAIL_MAX = 700.0       # nm

UV_RATIO_THRESHOLD = 0.75       # raw peak2/peak1 ratio gate (no baseline)
LOBE_RATIO_THRESHOLD = 0.50     # smaller-lobe/larger-lobe evolution gate

# +/- band around each threshold -> flag for review (does NOT change the label).
BORDERLINE_BAND = 0.05

# The CD analysis window when a record has NO manual_window: the SINGLE source of
# truth for the default window. The CD couplet IS measured inside it -- there is
# NO algorithmic re-derivation. A per-record manual_window, when set, replaces
# it. (Provisional.)
WINDOW_DEFAULT_MIN = 450.0      # nm
WINDOW_DEFAULT_MAX = 500.0      # nm

# ---- sanity floor (NOT a tuning knob) -------------------------------------
# Below this many finite, aligned samples a record is treated as "short" -> its
# metrics are un-computable. find_peaks needs >= 3 points just to see one interior
# maximum; we keep a little headroom so a window restriction still leaves
# something usable.
MIN_USABLE_SAMPLES = 5


# ---- human category vocabulary ---------------------------------------------
# The classifier emits NO verdict. The ONLY category label is the human's review
# decision (human_classification), one of these three. Phase B sorts and tallies
# on these; classifier.py just owns the vocabulary so the UI and any cloud query
# share one spelling.
HUMAN_LADDER = "ladder"
HUMAN_STAIRCASE = "staircase"
HUMAN_UNSURE = "unsure"
HUMAN_LABELS = (HUMAN_LADDER, HUMAN_STAIRCASE, HUMAN_UNSURE)


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
    """Objective MEASUREMENTS for one record -- NO verdict. The classifier is a
    measurer; the Ladder/Staircase category is assigned by a human in Phase B.
    Every field is a computed metric, a gate boolean, or a note explaining why a
    metric is absent (un-computable data)."""
    computable: bool = True                       # False = un-computable / bad data
    bisignate: bool = False                       # genuine CD couplet present
    borderline: bool = False
    borderline_flags: list = field(default_factory=list)   # metric names in band
    notes: str = ""                               # objective measurement notes

    # --- UV evidence ---
    uv_baseline: Optional[float] = None              # display-only (far-red tail)
    uv_peak1: Point = field(default_factory=Point)   # shorter-lambda band
    uv_peak2: Point = field(default_factory=Point)   # longer-lambda band
    window_left: Optional[float] = None
    window_right: Optional[float] = None
    window_source: Optional[str] = None              # "manual" | "default"
    uv_two_peak_ratio: Optional[float] = None        # p2/p1 (raw, no baseline)

    # --- CD evidence (inside the window) ---
    cd_pos_lobe: Point = field(default_factory=Point)
    cd_neg_lobe: Point = field(default_factory=Point)
    lobe_ratio: Optional[float] = None               # min(|+|,|-|)/max(|+|,|-|)
    crossover_wavelength: Optional[float] = None      # CD zero-crossing (interp)

    gates: Gates = field(default_factory=Gates)

    @property
    def window(self) -> tuple:
        """The CD window as (left, right)."""
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
def _uncomputable(reason: str, **partial) -> ClassificationResult:
    """Build a result for UN-COMPUTABLE data (empty/short/non-finite arrays, flat
    UV, or a window with no samples): empty metrics + a note, no verdict, and
    computable=False so Phase B can file it under FLAGGED / bad data. The human
    still sees the record (grey/unreviewed) and decides in Phase B."""
    return ClassificationResult(computable=False, notes=reason, **partial)


def _near(value: Optional[float], threshold: float) -> bool:
    """True if `value` sits within +/- BORDERLINE_BAND of `threshold`."""
    if value is None:
        return False
    return (threshold - BORDERLINE_BAND) <= value <= (threshold + BORDERLINE_BAND)


def _resolve_manual_window(manual_window) -> Optional[tuple]:
    """Normalize a per-record manual window override to (lo, hi) floats, or None
    when absent / malformed (-> caller uses the global default window instead).

    Accepts a {"min_nm", "max_nm"} dict (as persisted on the cloud doc) or a
    plain (lo, hi) pair. A non-numeric, non-finite, or non-increasing window is
    treated as "no override" rather than an error, so a bad stored value can
    never break classification -- it just falls back to the default window.
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
    """Measure ONE record from its raw spectral arrays. The primary entry point.

    `wavelength`, `cd`, `uv` are parallel numeric sequences (lists or numpy
    arrays). `g` is not needed. Returns a ClassificationResult of OBJECTIVE
    metrics (UV bands, raw peak ratio, CD couplet, lobe ratio, the two gate
    booleans, crossover, window used) -- NO Ladder/Staircase verdict. Any
    un-computable input yields a result with empty metrics + a note rather than
    raising.

    `manual_window` is an optional per-record override of the CD analysis window
    ({"min_nm", "max_nm"} dict or (lo, hi) pair). When present and valid the CD
    couplet is read inside it; otherwise it is read inside the global default
    window [WINDOW_DEFAULT_MIN, WINDOW_DEFAULT_MAX]. UV band detection and the UV
    peak2/peak1 ratio gate are UNAFFECTED by the window.
    """
    prepared = _prepare(wavelength, cd, uv)
    if prepared is None:
        return _uncomputable("empty/short or non-finite arrays "
                             "(fewer than %d usable samples)"
                             % MIN_USABLE_SAMPLES)
    wl, cdv, uvv = prepared

    # --- step 1: UV baseline from the far-red flat tail ---------------------
    tail = (wl >= BASELINE_TAIL_MIN) & (wl <= BASELINE_TAIL_MAX)
    if not tail.any():
        return _uncomputable(
            "no UV samples in baseline tail "
            f"[{BASELINE_TAIL_MIN:.0f}-{BASELINE_TAIL_MAX:.0f} nm]")
    baseline = float(np.mean(uvv[tail]))

    # Per-record UV dynamic range that scales the prominence threshold (k).
    uv_span = float(uvv.max() - uvv.min())
    if uv_span <= 0.0:
        return _uncomputable("UV signal is flat (no dynamic range)",
                             uv_baseline=baseline)

    # --- step 2: UV band detection (the "two peaks in one band" criterion) --
    win = (wl >= UV_PEAK_SEARCH_MIN) & (wl <= UV_PEAK_SEARCH_MAX)
    if not win.any():
        return _uncomputable(
            "no UV samples in search window "
            f"[{UV_PEAK_SEARCH_MIN:.0f}-{UV_PEAK_SEARCH_MAX:.0f} nm]",
            uv_baseline=baseline)
    wl_w, uv_w = wl[win], uvv[win]

    prominence = UV_PROMINENCE_RATIO * uv_span
    peak_idx, props = find_peaks(uv_w, prominence=prominence)

    if peak_idx.size < 2:
        # Only a single (or no) prominent UV band -> not bisignate. A MEASUREMENT,
        # not a verdict: report the largest absorbance as peak1 for the human to
        # eyeball; no CD window/couplet is computed.
        if peak_idx.size == 1:
            i1 = int(peak_idx[0])
        else:
            i1 = int(np.argmax(uv_w))
        return ClassificationResult(
            notes="single UV band",
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

    # --- step 3: resolve the CD analysis window [LEFT, RIGHT] ---------------
    # SINGLE source of truth: the per-record manual_window when set, else the
    # global default [WINDOW_DEFAULT_MIN, WINDOW_DEFAULT_MAX]. There is NO
    # algorithmic re-derivation -- the CD couplet (step 4 on) is read inside
    # exactly this [left, right], and it is the window stored on the result.
    mw = _resolve_manual_window(manual_window)
    if mw is not None:
        left, right = mw
        window_source = "manual"
    else:
        left, right = WINDOW_DEFAULT_MIN, WINDOW_DEFAULT_MAX
        window_source = "default"

    # --- CD samples inside the window (a missing window is un-computable) ----
    inwin = (wl >= left) & (wl <= right)
    common = dict(
        uv_baseline=baseline,
        uv_peak1=peak1, uv_peak2=peak2,
        window_left=left, window_right=right,
        window_source=window_source,
    )
    if not inwin.any():
        return _uncomputable(
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
    # RAW peak2/peak1 ratio (longer-lambda over shorter-lambda), NOT baseline-
    # subtracted. uv_baseline is kept for display only and does not feed this.
    p1_raw, p2_raw = peak1.value, peak2.value
    if p1_raw > 0.0:
        uv_two_peak_ratio = p2_raw / p1_raw
    else:
        uv_two_peak_ratio = None        # band-1 at/below zero -> undefined
    common["uv_two_peak_ratio"] = uv_two_peak_ratio

    big = max(abs(pos_lobe.value), abs(neg_lobe.value))
    lobe_ratio = (min(abs(pos_lobe.value), abs(neg_lobe.value)) / big
                  if big > 0.0 else None)
    common["lobe_ratio"] = lobe_ratio

    if not couplet:
        return ClassificationResult(
            notes="no CD couplet", bisignate=False,
            gates=Gates(uv_band_detected=True, cd_couplet=False),
            **common)

    # --- evolution gates: reported as OBJECTIVE evidence, NOT a verdict ------
    # Both ratios are measured against their thresholds; the booleans are shown
    # to the human, who makes the Ladder/Staircase call. The classifier does NOT
    # collapse them into a label here, and computes no handedness.
    notes: list = []
    # RAW peak2/peak1 ratio (no baseline subtraction).
    uv_ratio_pass = (uv_two_peak_ratio is not None
                     and uv_two_peak_ratio >= UV_RATIO_THRESHOLD)
    if uv_two_peak_ratio is None:
        notes.append("UV peak1 at/below zero; peak2/peak1 undefined")
    elif not uv_ratio_pass:
        notes.append("UV peak2/peak1 below threshold")

    lobe_ratio_pass = (lobe_ratio is not None
                       and lobe_ratio >= LOBE_RATIO_THRESHOLD)
    if not lobe_ratio_pass:
        notes.append("lobe ratio below threshold")

    gates = Gates(uv_band_detected=True, cd_couplet=True,
                  uv_ratio_pass=uv_ratio_pass, lobe_ratio_pass=lobe_ratio_pass)

    # --- borderline: a metric sits within +/-BORDERLINE_BAND of its threshold -
    borderline_flags: list = []
    if _near(uv_two_peak_ratio, UV_RATIO_THRESHOLD):
        borderline_flags.append("uv_two_peak_ratio")
    if _near(lobe_ratio, LOBE_RATIO_THRESHOLD):
        borderline_flags.append("lobe_ratio")

    return ClassificationResult(
        bisignate=True,
        borderline=bool(borderline_flags),
        borderline_flags=borderline_flags,
        notes="; ".join(notes),
        gates=gates,
        **common,
    )


def classify_record(rec: dict) -> ClassificationResult:
    """Classify a cloud document dict using its embedded spectra.

    Reads the `wavelength` / `cd` / `uv` arrays stored by
    mongo_db.promote_records, plus an optional per-record `manual_window`
    override ({"min_nm", "max_nm"}) that, when present, replaces the global
    default CD analysis window. Read-only -- the document is never modified.
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
    read them from here). Embedded in every computed_metrics cache doc so a later
    pass can detect that k or any threshold changed and the cache is stale, and
    re-run the (idempotent) measure+write pass.
    """
    return {
        "uv_peak_search_nm": [UV_PEAK_SEARCH_MIN, UV_PEAK_SEARCH_MAX],
        "uv_prominence_ratio": UV_PROMINENCE_RATIO,
        "default_window_nm": [WINDOW_DEFAULT_MIN, WINDOW_DEFAULT_MAX],
        "baseline_tail_nm": [BASELINE_TAIL_MIN, BASELINE_TAIL_MAX],
        "uv_ratio_threshold": UV_RATIO_THRESHOLD,
        "lobe_ratio_threshold": LOBE_RATIO_THRESHOLD,
        "borderline_band": BORDERLINE_BAND,
    }


def _pt(p: Point) -> list:
    """A Point as a plain [wl, value] list for the cache doc."""
    return [p.wl, p.value]


def computed_metrics_doc(result: ClassificationResult) -> dict:
    """Compact, queryable cache document for a cloud doc's `computed_metrics`
    field: the OBJECTIVE metrics only (both ratios, peak / lobe / crossover
    locations, the bisignate + borderline flags, the per-gate booleans) + the
    thresholds snapshot that produced them + a UTC timestamp. NO verdict and NO
    handedness -- the human's human_classification is the only category label.

    This is a REFRESHABLE CACHE derived from PROVISIONAL thresholds AND the
    window the record was measured under: re-run the (idempotent) measure+write
    pass whenever k/f or the constants above change, OR whenever a record's
    manual_window changes. `window_used` records the actual window so a stored
    result always carries the window it was computed under. Contains only derived
    scalars -- NEVER the spectral arrays.
    """
    return {
        "bisignate": result.bisignate,
        "borderline": result.borderline,
        "borderline_flags": list(result.borderline_flags),
        "notes": result.notes,
        "uv_baseline": result.uv_baseline,
        "uv_peak1": _pt(result.uv_peak1),
        "uv_peak2": _pt(result.uv_peak2),
        "window_used": [result.window_left, result.window_right],
        "window_source": result.window_source,
        "uv_two_peak_ratio": result.uv_two_peak_ratio,
        "cd_pos_lobe": _pt(result.cd_pos_lobe),
        "cd_neg_lobe": _pt(result.cd_neg_lobe),
        "lobe_ratio": result.lobe_ratio,
        "crossover_wavelength": result.crossover_wavelength,
        "gates": asdict(result.gates),
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
    """Print and return a summary tally of OBJECTIVE measurements over a list of
    (rec, result) pairs. NO verdict counts -- the classifier no longer labels; a
    human assigns Ladder/Staircase in Phase B."""
    total = len(results)
    n_two_band = sum(1 for _, r in results if r.gates.uv_band_detected)
    n_couplet = sum(1 for _, r in results if r.gates.cd_couplet)
    n_both_gates = sum(1 for _, r in results
                       if r.gates.uv_ratio_pass and r.gates.lobe_ratio_pass)
    n_border = sum(1 for _, r in results if r.borderline)

    log("")
    log("=" * 64)
    log("MEASUREMENT SUMMARY  (objective metrics -- no verdict)")
    log(f"  total records           : {total}")
    log(f"  two UV bands detected    : {n_two_band}")
    log(f"  genuine CD couplet       : {n_couplet}")
    log(f"  both evolution gates ok  : {n_both_gates}")
    log(f"  borderline               : {n_border}  (marked for human review)")
    log("=" * 64)

    return {
        "total": total, "two_band": n_two_band, "couplet": n_couplet,
        "both_gates": n_both_gates, "borderline": n_border,
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

    log(f"\nMeasuring {len(records)} cloud record(s)...\n")
    results: list = []
    for i, rec in enumerate(records):
        res = classify_record(rec)
        results.append((rec, res))
        log(f"{_record_label(rec, i)}  "
            f"{_borderline_text(res):<24} {_format_metrics(res)}")
        if res.notes:
            log(f"        notes: {res.notes}")

    print_tally(results, log=log)
    return results


if __name__ == "__main__":
    # Direct run: classify whatever is in the configured Atlas collection and
    # print the per-record lines + tally. Degrades to a clear message (no
    # crash) when the cloud is unconfigured/unreachable.
    run_cloud_classification()
