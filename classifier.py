"""
CD-shape classifier engine: ladder vs staircase vs flagged.

PURE ANALYSIS. Operates on a record's spectral arrays (wavelength + CD +
UV-Vis) and returns a structured, fully-auditable classification result. It
NEVER reads or writes a file, never touches the DB, never mutates the input
arrays, and never modifies a cloud document -- the label is derived/computed
on demand. This is Phase A; the visual two-column review UI is Phase B.

Domain context: a *ladder*-type CD signature indicates a specific polymer
backbone conformation. Two STRICT criteria must BOTH pass for "ladder":

  Criterion 1 -- a genuine bisignate CD couplet in 450-500 nm whose
                 |positive| / |negative| peak ratio is >= 0.50.
  Criterion 2 -- two UV-Vis absorption peaks in 300-550 nm whose
                 peak2 / peak1 (longer-wl / shorter-wl) ratio is >= 0.75.

A record is "ladder" when BOTH criteria pass, otherwise "staircase". A
ladder-type ALWAYS exhibits the bisignate couplet, so the ABSENCE of an
opposite-sign couplet in 450-500 nm definitionally means NOT ladder ->
staircase (criterion 1 simply fails). "flagged" is reserved for records whose
spectra cannot be evaluated at all -- empty/short/NaN/inf arrays, or no samples
in a required search window. A separate "borderline" flag marks results sitting
within +/-0.05 of either threshold so a human knows to eyeball them; it does
NOT change the hard label.

All thresholds / windows are PROVISIONAL (trend-based) and kept as named
constants below so they're trivial to tune later.

Dependencies: numpy only. No scipy (a small prominence-based peak finder is
implemented here), no Qt, no originpro. The cloud layer (mongo_db) is imported
lazily inside the runner so importing this module never opens a connection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


# ===========================================================================
# TUNABLE CONSTANTS  (provisional, trend-based -- adjust here only)
# ===========================================================================
# Criterion 1 -- CD couplet.
CD_WINDOW_NM = (450.0, 500.0)        # search window for the bisignate couplet
CRIT1_RATIO_THRESHOLD = 0.50         # |pos peak| / |neg peak| must be >= this

# Criterion 2 -- UV-Vis two-peak ratio.
UV_WINDOW_NM = (300.0, 550.0)        # search window for the two absorption peaks
CRIT2_RATIO_THRESHOLD = 0.75         # peak2 / peak1 must be >= this
# A local maximum counts as a real UV peak only if its prominence is at least
# this fraction of the in-window absorbance span (max - min). Filters noise
# bumps without needing scipy. Provisional.
UV_PEAK_PROMINENCE_FRAC = 0.05

# Borderline band: a ratio within +/- this of EITHER threshold flags the
# result for human review (does not change the hard label).
BORDERLINE_DELTA = 0.05

LABELS = ("ladder", "staircase", "flagged")


# ===========================================================================
# RESULT TYPES
# ===========================================================================
@dataclass
class Criterion1:
    """CD-couplet evidence (search window CD_WINDOW_NM)."""
    pos_peak_value: Optional[float]   # CD maximum (expected > 0 for a couplet)
    pos_peak_wl: Optional[float]
    neg_peak_value: Optional[float]   # CD minimum (expected < 0 for a couplet)
    neg_peak_wl: Optional[float]
    couplet_ratio: Optional[float]    # |pos| / |neg|; may exceed 1 and still pass
    passed: bool                      # couplet_ratio >= CRIT1_RATIO_THRESHOLD
    opposite_signs: bool              # pos > 0 AND neg < 0 (a genuine couplet)
    has_data: bool = True             # CD samples existed in the window; False
                                      # is the only criterion-1 reason to flag


@dataclass
class Criterion2:
    """UV-Vis two-peak evidence (search window UV_WINDOW_NM)."""
    peak1_value: Optional[float]      # absorbance at the shorter-wl peak
    peak1_wl: Optional[float]
    peak2_value: Optional[float]      # absorbance at the longer-wl peak
    peak2_wl: Optional[float]
    ratio: Optional[float]            # peak2 / peak1
    passed: bool                      # ratio >= CRIT2_RATIO_THRESHOLD
    single_peak: bool                 # only one (or zero) prominent UV peak found
    has_data: bool = True             # UV samples existed in the window; False
                                      # is the only criterion-2 reason to flag


@dataclass
class ClassificationResult:
    label: str                        # "ladder" | "staircase" | "flagged"
    borderline: bool                  # True if any criterion sits in its band
    borderline_criteria: list = field(default_factory=list)  # e.g. ["criterion2"]
    criterion1: Optional[Criterion1] = None
    criterion2: Optional[Criterion2] = None
    reasons: list = field(default_factory=list)   # human-readable audit notes

    def to_dict(self) -> dict:
        """Flat dict (nested criteria expanded) for logging / JSON / audit."""
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


def _aligned_clean(wavelength, y) -> tuple[np.ndarray, np.ndarray]:
    """Pair wavelength with one signal, trimmed to a common length and with
    any non-finite (NaN/inf) sample in EITHER array dropped pairwise.

    A copy is produced via the masking, so the caller's arrays are untouched.
    Returns (wl, y) float arrays (possibly empty).
    """
    wl = _to_float_array(wavelength)
    yy = _to_float_array(y)
    n = min(wl.size, yy.size)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    wl, yy = wl[:n], yy[:n]
    mask = np.isfinite(wl) & np.isfinite(yy)
    return wl[mask], yy[mask]


def _window(wl: np.ndarray, y: np.ndarray,
            lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (wl, y) restricted to lo <= wl <= hi."""
    m = (wl >= lo) & (wl <= hi)
    return wl[m], y[m]


# ---- self-contained prominence-based peak finder (no scipy) ----------------
def _local_maxima_indices(y: np.ndarray) -> list:
    """Indices of interior local maxima, plateau-aware.

    A plateau (run of equal values) that rises before it and falls after it
    counts as one peak, reported at the plateau's midpoint. Boundary samples
    are never peaks (we need a neighbor on each side to judge a rise/fall).
    """
    n = y.size
    peaks: list = []
    i = 1
    while i < n - 1:
        if y[i] > y[i - 1]:
            j = i
            while j < n - 1 and y[j + 1] == y[i]:
                j += 1
            if j < n - 1 and y[j + 1] < y[i]:
                peaks.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return peaks


def _prominence(y: np.ndarray, p: int) -> float:
    """Topographic prominence of the peak at index `p` (scipy-equivalent).

    Scan left until a sample higher than the peak (or the border), tracking the
    minimum seen -> left base. Same to the right -> right base. The peak's base
    is the HIGHER of the two; prominence = peak height - base.
    """
    height = float(y[p])
    n = y.size

    i = p - 1
    left_min = height
    while i >= 0 and y[i] <= height:
        if y[i] < left_min:
            left_min = float(y[i])
        i -= 1

    j = p + 1
    right_min = height
    while j < n and y[j] <= height:
        if y[j] < right_min:
            right_min = float(y[j])
        j += 1

    return height - max(left_min, right_min)


def _prominent_peaks(y: np.ndarray) -> list:
    """List of (index, prominence) for every local maximum whose prominence is
    at least UV_PEAK_PROMINENCE_FRAC of the in-window absorbance span, sorted
    by prominence descending. Empty for flat / featureless / too-short input.
    """
    if y.size < 3:
        return []
    maxima = _local_maxima_indices(y)
    if not maxima:
        return []
    span = float(np.max(y) - np.min(y))
    if span <= 0:
        return []
    thr = UV_PEAK_PROMINENCE_FRAC * span
    out = [(p, _prominence(y, p)) for p in maxima]
    out = [(p, pr) for p, pr in out if pr >= thr]
    out.sort(key=lambda t: t[1], reverse=True)
    return out


# ===========================================================================
# CRITERIA
# ===========================================================================
def _evaluate_criterion1(wl: np.ndarray, cd: np.ndarray,
                         reasons: list) -> Criterion1:
    """Criterion 1 -- bisignate CD couplet within CD_WINDOW_NM.

    Simple max/min within the window per spec. A genuine couplet needs opposite
    signs (a positive AND a negative peak). If signs are NOT opposite, the data
    is fine but there's no couplet -> criterion 1 FAILS (the caller then labels
    the record "staircase", since ladder always shows the couplet). Only a
    window with NO samples is un-computable -> the caller flags it.
    """
    wlw, cdw = _window(wl, cd, *CD_WINDOW_NM)
    if wlw.size == 0:
        # No samples to evaluate -> un-computable -> caller flags it.
        reasons.append(
            f"no CD data in {CD_WINDOW_NM[0]:.0f}-{CD_WINDOW_NM[1]:.0f} nm "
            f"window (cannot classify -> flagged)")
        return Criterion1(None, None, None, None, None, False, False,
                          has_data=False)

    pos_i = int(np.argmax(cdw))
    neg_i = int(np.argmin(cdw))
    pos_v, pos_wl = float(cdw[pos_i]), float(wlw[pos_i])
    neg_v, neg_wl = float(cdw[neg_i]), float(wlw[neg_i])

    opposite = (pos_v > 0.0) and (neg_v < 0.0)
    if not opposite:
        # Data present, but no bisignate couplet. Ladder ALWAYS has the couplet,
        # so its absence means not-ladder: criterion 1 fails and the record is a
        # staircase (caller's else branch). Recorded distinctly so Phase B can
        # tell this apart from a failed-ratio or failed-UV staircase. has_data
        # stays True -- this is a real classification, not a data problem.
        reasons.append("no bisignate couplet (criterion 1 fails)")
        return Criterion1(pos_v, pos_wl, neg_v, neg_wl, None, False, False)

    ratio = abs(pos_v) / abs(neg_v)
    passed = ratio >= CRIT1_RATIO_THRESHOLD
    return Criterion1(pos_v, pos_wl, neg_v, neg_wl, ratio, passed, True)


def _evaluate_criterion2(wl: np.ndarray, uv: np.ndarray,
                         reasons: list) -> Criterion2:
    """Criterion 2 -- UV-Vis two-peak ratio within UV_WINDOW_NM.

    Finds prominent absorption peaks; peak1 = shorter-wl, peak2 = longer-wl
    (ordered by wavelength). When more than two prominent peaks are found
    (noise on an otherwise clean spectrum), the two MOST prominent are kept and
    then ordered by wavelength. With only one (or zero) prominent peak, peak2 is
    absent -> criterion FAILS (single-peak / staircase contribution).
    [ASSUMPTION surfaced via single_peak + a reason note.]
    """
    wlw, uvw = _window(wl, uv, *UV_WINDOW_NM)
    if wlw.size == 0:
        # No samples to evaluate -> un-computable -> caller flags it.
        reasons.append(
            f"no UV data in {UV_WINDOW_NM[0]:.0f}-{UV_WINDOW_NM[1]:.0f} nm "
            f"window (cannot classify -> flagged)")
        return Criterion2(None, None, None, None, None, False, True,
                          has_data=False)

    peaks = _prominent_peaks(uvw)

    if len(peaks) >= 2:
        # Two most prominent, then ordered shorter-wl -> longer-wl.
        top2 = sorted(peaks[:2], key=lambda t: float(wlw[t[0]]))
        i1, i2 = top2[0][0], top2[1][0]
        p1v, p1wl = float(uvw[i1]), float(wlw[i1])
        p2v, p2wl = float(uvw[i2]), float(wlw[i2])
        if p1v == 0.0:
            reasons.append("UV peak1 absorbance is zero; ratio undefined")
            return Criterion2(p1v, p1wl, p2v, p2wl, None, False, False)
        ratio = p2v / p1v
        passed = ratio >= CRIT2_RATIO_THRESHOLD
        return Criterion2(p1v, p1wl, p2v, p2wl, ratio, passed, False)

    # Zero or one prominent peak -> no peak2 -> criterion fails.
    reasons.append("single UV peak / no peak2 (criterion 2 fails)")
    if len(peaks) == 1:
        i1 = peaks[0][0]
    else:
        # Featureless within the window: still report the largest absorbance as
        # peak1 for audit, but mark single_peak and fail.
        i1 = int(np.argmax(uvw))
    return Criterion2(float(uvw[i1]), float(wlw[i1]),
                      None, None, None, False, True)


# ===========================================================================
# CLASSIFICATION CORE
# ===========================================================================
def _in_band(ratio: Optional[float], threshold: float) -> bool:
    """True if `ratio` sits within +/- BORDERLINE_DELTA of `threshold`."""
    if ratio is None:
        return False
    return (threshold - BORDERLINE_DELTA) <= ratio <= (threshold + BORDERLINE_DELTA)


def classify_arrays(wavelength, cd, uv) -> ClassificationResult:
    """Classify ONE record from its raw spectral arrays. The primary entry point.

    `wavelength`, `cd`, `uv` are parallel numeric sequences (Python lists or
    numpy arrays); `g`-value is not needed. Arrays are cleaned defensively
    (length-aligned, NaN/inf dropped) without mutating the inputs. Any
    degenerate input (empty/short/non-numeric) yields a "flagged" result with a
    reason rather than raising.

    Returns a fully-populated ClassificationResult (both criteria filled in for
    audit even when the label is "flagged").
    """
    reasons: list = []

    # Clean each signal against the wavelength axis independently so a NaN in
    # one signal never discards a good sample from the other.
    wl_cd, cd_clean = _aligned_clean(wavelength, cd)
    wl_uv, uv_clean = _aligned_clean(wavelength, uv)

    c1 = _evaluate_criterion1(wl_cd, cd_clean, reasons)
    c2 = _evaluate_criterion2(wl_uv, uv_clean, reasons)

    # Label. "flagged" is reserved for un-computable data: a criterion had NO
    # samples in its window (or the arrays were empty/NaN). Everything that can
    # be computed gets a real label -- "ladder" only when BOTH criteria pass,
    # else "staircase". Staircase therefore covers a failed couplet ratio, a
    # single UV peak, a failed UV ratio, AND the absence of a bisignate couplet
    # (ladder always shows the couplet, so no couplet => not ladder). The
    # reasons list records which path produced the staircase.
    if not (c1.has_data and c2.has_data):
        label = "flagged"
    elif c1.passed and c2.passed:
        label = "ladder"
    else:
        label = "staircase"

    # Borderline only applies to classified records (a flagged record has no
    # couplet ratio to be "near"). Either criterion's ratio may flag it.
    borderline_criteria: list = []
    if label != "flagged":
        if _in_band(c1.couplet_ratio, CRIT1_RATIO_THRESHOLD):
            borderline_criteria.append("criterion1")
        if _in_band(c2.ratio, CRIT2_RATIO_THRESHOLD):
            borderline_criteria.append("criterion2")

    return ClassificationResult(
        label=label,
        borderline=bool(borderline_criteria),
        borderline_criteria=borderline_criteria,
        criterion1=c1,
        criterion2=c2,
        reasons=reasons,
    )


def classify_record(rec: dict) -> ClassificationResult:
    """Classify a cloud document dict using its embedded spectra.

    Reads the `wavelength` / `cd` / `uv` arrays stored by
    mongo_db.promote_records. Read-only -- the document is never modified.
    """
    return classify_arrays(rec.get("wavelength"), rec.get("cd"), rec.get("uv"))


# ===========================================================================
# CLOUD-CACHE DOC HELPERS  (additive tags written back to each cloud doc)
# ===========================================================================
def thresholds_snapshot() -> dict:
    """The PROVISIONAL tuning constants as a plain dict, sourced straight from
    the module-level constants above (never duplicate the numbers elsewhere --
    read them from here). Embedded in every auto_classification cache doc so a
    later pass can detect that the thresholds changed and the cache is stale.
    """
    return {
        "cd_window_nm": [CD_WINDOW_NM[0], CD_WINDOW_NM[1]],
        "crit1_ratio_threshold": CRIT1_RATIO_THRESHOLD,
        "uv_window_nm": [UV_WINDOW_NM[0], UV_WINDOW_NM[1]],
        "crit2_ratio_threshold": CRIT2_RATIO_THRESHOLD,
        "uv_peak_prominence_frac": UV_PEAK_PROMINENCE_FRAC,
        "borderline_delta": BORDERLINE_DELTA,
    }


def auto_classification_doc(result: ClassificationResult) -> dict:
    """Compact, queryable cache document for a cloud doc's `auto_classification`
    field: the hard label + borderline flag + the key metrics (couplet ratio,
    UV ratio, peak locations) + the thresholds snapshot that produced them + a
    UTC timestamp.

    This is a REFRESHABLE CACHE derived from PROVISIONAL thresholds: re-run the
    classify+write pass (which is idempotent) whenever the constants above
    change. Contains only derived scalars -- NEVER the spectral arrays.
    """
    c1, c2 = result.criterion1, result.criterion2
    return {
        "label": result.label,
        "borderline": result.borderline,
        "borderline_criteria": list(result.borderline_criteria),
        "couplet_ratio": c1.couplet_ratio,
        "uv_ratio": c2.ratio,
        "criterion1_passed": c1.passed,
        "criterion2_passed": c2.passed,
        "pos_peak_wl": c1.pos_peak_wl,
        "pos_peak_value": c1.pos_peak_value,
        "neg_peak_wl": c1.neg_peak_wl,
        "neg_peak_value": c1.neg_peak_value,
        "peak1_wl": c2.peak1_wl,
        "peak1_value": c2.peak1_value,
        "peak2_wl": c2.peak2_wl,
        "peak2_value": c2.peak2_value,
        "single_uv_peak": c2.single_peak,
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
    c1, c2 = res.criterion1, res.criterion2
    # "max" / "min" rather than "+"/"-": the values carry their own sign, and a
    # flagged record's CD maximum can itself be negative (both extremes same
    # sign -> no couplet), which a hardcoded "+" would misrepresent.
    c1s = (f"C1 couplet={_f(c1.couplet_ratio, 2)} "
           f"[max={_f(c1.pos_peak_value)}@{_wl(c1.pos_peak_wl)} / "
           f"min={_f(c1.neg_peak_value)}@{_wl(c1.neg_peak_wl)} "
           f"opp={'Y' if c1.opposite_signs else 'N'} "
           f"pass={'Y' if c1.passed else 'N'}]")
    c2s = (f"C2 uv={_f(c2.ratio, 2)} "
           f"[p1={_f(c2.peak1_value)}@{_wl(c2.peak1_wl)} / "
           f"p2={_f(c2.peak2_value)}@{_wl(c2.peak2_wl)} "
           f"single={'Y' if c2.single_peak else 'N'} "
           f"pass={'Y' if c2.passed else 'N'}]")
    return f"{c1s}  {c2s}"


def _borderline_text(res: ClassificationResult) -> str:
    if not res.borderline:
        return "-"
    return "BORDERLINE(" + ",".join(res.borderline_criteria) + ")"


def _is_verified(rec: dict) -> bool:
    try:
        return int(rec.get("verified") or 0) == 1
    except (TypeError, ValueError):
        return False


def print_tally(results: list, log=print) -> dict:
    """Print and return the summary tally over a list of (rec, result) pairs."""
    total = len(results)
    n_ladder = sum(1 for _, r in results if r.label == "ladder")
    n_stair = sum(1 for _, r in results if r.label == "staircase")
    n_flag = sum(1 for _, r in results if r.label == "flagged")
    n_border = sum(1 for _, r in results if r.borderline)
    classified = n_ladder + n_stair

    log("")
    log("=" * 64)
    log("CLASSIFICATION SUMMARY")
    log(f"  total records          : {total}")
    log(f"  ladder                 : {n_ladder}")
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
        "total": total, "ladder": n_ladder, "staircase": n_stair,
        "flagged": n_flag, "borderline": n_border,
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
        log(f"{_record_label(rec, i)}  {res.label.upper():<9} "
            f"{_borderline_text(res):<24} {_format_metrics(res)}")
        if res.reasons:
            log(f"        reasons: {'; '.join(res.reasons)}")

    print_tally(results, log=log)
    return results


if __name__ == "__main__":
    # Direct run: classify whatever is in the configured Atlas collection and
    # print the per-record lines + tally. Degrades to a clear message (no
    # crash) when the cloud is unconfigured/unreachable.
    run_cloud_classification()
