"""
CD-shape REVIEW window -- Phase B worklist + human-classification layer over the
Phase A MEASURER (`classifier.py`).

A NEW, SEPARATE window from the read-only Cloud Browser (this module never
imports or modifies `cloud_browser_window`). It reads the SAME source -- the
verified CLOUD (MongoDB Atlas) records via `mongo_db.fetch_records` -- runs
`classifier.classify_record` on each to compute OBJECTIVE METRICS (no verdict),
and lays them out as:

  * a permanent left-hand WORKLIST of EVERY record (entries never leave when
    classified) with a done marker + an optional All / Done / Not-done filter;
  * three HUMAN-ONLY category columns (LADDER | STAIRCASE | FLAGGED-unsure) that
    fill ONLY from the reviewer's saved human_classification.

A record with no human label appears only in the worklist, rendered GREY
(unreviewed). Saving a Ladder/Staircase/Unsure category turns the whole strip
GREEN and files it into the matching column; clearing the category returns it to
grey/unreviewed. The classifier NEVER suggests a verdict -- the human decides
from the metrics.

Clicking an entry opens a single-record DETAIL view: three stacked,
Origin-faithful matplotlib subplots (CD / g / UV, raw embedded arrays drawn as
plain lines -- no smoothing) with the classifier's selected peaks marked and the
analysis window shaded, plus a metrics panel (UV peak ratio / CD peak ratio) that
makes the measurement auditable.

WRITES (the only ones this window performs -- spectra are NEVER touched):
  - computed_metrics    : refreshable cache of the classifier's OBJECTIVE metrics
    (no label), synced to each cloud doc keyed by record_id. Idempotent;
    re-synced when stale (after the PROVISIONAL thresholds in classifier.py
    change, OR when a record's manual_window changes). Writers also $unset the
    legacy `auto_classification` block so old docs migrate.
  - manual_window       : per-record CD analysis window {min_nm, max_nm} set via
    the dual-handle range slider. Replaces the global default window. Persisted
    TOGETHER with a freshly recomputed computed_metrics (one atomic write via
    mongo_db.set_manual_window). "Reset to default" $unsets it.
  - human_classification: the reviewer's category (Ladder/Staircase/Unsure +
    timestamp) -- the ONLY category label and the single source of truth for the
    columns and the ladder-% stat. Saved on demand; CLEARED ($unset) on "Clear
    saved".

Thresholds / the default window seed are read from classifier.py's named
constants -- not duplicated here.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)

import classifier


# Embedded-array keys + plot styling (kept local so this window is independent
# of the Cloud Browser module).
_SIGNAL_DOC_KEY = {"CD": "cd", "g": "g", "UV": "uv"}
_SIGNAL_TITLES = {"CD": "CD (mdeg)", "g": "g-value", "UV": "UV-Vis (abs)"}
_SIGNAL_COLORS = {"CD": "tab:blue", "g": "tab:green", "UV": "tab:red"}
_PLOT_X_RANGE = (300, 700)

# Default INITIAL handle positions for the CD analysis window when a record has
# no saved manual_window. This is ONLY where the draggable handles START -- it
# is NOT a clamp: the handles drag across the record's full spectral extent
# (see CDReviewWindow._window_bounds). It is deliberately wider than the
# classifier's measurement default (classifier.WINDOW_DEFAULT_*), which is left
# untouched and still governs how a no-manual-window record is MEASURED.
_DEFAULT_WINDOW_LO = 400.0
_DEFAULT_WINDOW_HI = 550.0

# Pixel radius for grabbing a window handle by clicking near it on the plot.
_HANDLE_GRAB_PX = 10

# Entry lifecycle tints. There is only ONE axis of state now -- reviewed or not.
# GREY (the main-UI staging "not yet sorted" gray) = unreviewed / no human label;
# whole-strip GREEN (the staging "confirmed" green) = a saved human_classification.
# Borderline is shown as a text badge, not a background, so it stays legible on
# either tint.
_UNREVIEWED_GREY = QColor("#d3d3d3")   # staging-area gray -- "not yet sorted"
_CLASSIFIED_GREEN = QColor("#d4edda")  # whole-strip green -- "classified / done"


def _polymer_system(rec: dict) -> str:
    """The polymer system -- p1, or 'p1+p2' for a two-component blend."""
    p1 = rec.get("p1_name") or "?"
    p2 = rec.get("p2_name")
    return p1 if (not p2 or p2 == "None") else f"{p1}+{p2}"


def _num_str(x) -> str:
    """A numeric metadata value as a clean string ('' when blank). Drops a
    trailing '.0' so conc/anneal_temp read as 20 / 160, not 20.0 / 160.0."""
    if x is None or x == "" or x == "None" or isinstance(x, bool):
        return ""
    if isinstance(x, (int, float)):
        return str(int(x)) if float(x).is_integer() else str(x)
    return str(x).strip()


def _record_label(rec: dict) -> str:
    """Metadata-rich entry label, segments joined by ' | ':

        components | ratio | conc+solvent | anneal

    Built ONLY from fields already on the record (no computation). Empty segments
    are dropped so singles don't show '| |' gaps (e.g. 'R-F8BT | 20CB | AN160').
    """
    segments = []

    # components: p1, or 'p1+p2' for a real two-component blend. n_components is
    # authoritative for "single"; a blank/'None' p2 means the same.
    p1 = (rec.get("p1_name") or "").strip()
    p2 = (rec.get("p2_name") or "").strip()
    single = (rec.get("n_components") == 1) or (not p2) or (p2 == "None")
    components = p1 if single else f"{p1}+{p2}"
    if components:
        segments.append(components)

    # ratio (e.g. '10x90'); omitted for singles / blank.
    ratio = (rec.get("ratio") or "").strip()
    if ratio and ratio != "None":
        segments.append(ratio)

    # conc+solvent concatenated, no space (20 + 'CB' -> '20CB'); omit if no conc.
    conc = _num_str(rec.get("conc"))
    if conc:
        solvent = (rec.get("solvent") or "").strip()
        if solvent == "None":
            solvent = ""
        segments.append(f"{conc}{solvent}")

    # anneal keyed off film_state (authoritative): AN+temp, or plain AP.
    fs = (rec.get("film_state") or "").strip()
    if fs == "AN":
        segments.append("AN" + _num_str(rec.get("anneal_temp")))
    elif fs == "AP":
        segments.append("AP")          # ignore anneal_temp -> never 'AP160'
    elif fs:
        segments.append(fs)

    return " | ".join(segments) if segments else "(no metadata)"


def _human_label(rec: dict):
    """The saved human category ('ladder'/'staircase'/'unsure') or None. This is
    the ONLY category label -- the classifier emits none."""
    h = rec.get("human_classification")
    if isinstance(h, dict):
        lab = (h.get("label") or "").lower()
        return lab or None
    return None


def _f(x, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _wl(x) -> str:
    return "n/a" if x is None else f"{x:.0f} nm"


def _round6(x):
    return None if x is None else round(float(x), 6)


def _metrics_cache_current(stored, fresh) -> bool:
    """True if the stored computed_metrics cache still matches a freshly computed
    one -- so the sync pass can skip an idempotent no-op write.

    A change in the PROVISIONAL thresholds snapshot, the WINDOW the record was
    measured under, the borderline flag, or either ratio (to 6 dp) marks the cache
    STALE so it is rewritten. The window is included because computed_metrics is a
    cache of BOTH the constants and the window used -- a manual_window edit must
    invalidate it. Everything else is derived from these, so this is sufficient.
    """
    if not isinstance(stored, dict):
        return False
    if stored.get("thresholds") != fresh.get("thresholds"):
        return False
    if stored.get("window_source") != fresh.get("window_source"):
        return False
    sw = stored.get("window_used") or [None, None]
    fw = fresh.get("window_used") or [None, None]
    if [_round6(x) for x in sw] != [_round6(x) for x in fw]:
        return False
    if bool(stored.get("borderline")) != bool(fresh.get("borderline")):
        return False
    for k in ("uv_two_peak_ratio", "lobe_ratio"):
        if _round6(stored.get(k)) != _round6(fresh.get(k)):
            return False
    return True


class CDReviewWindow(QDialog):
    """Non-modal worklist + human-classification review over the verified cloud
    records. Reads + MEASURES on the main thread; the only cloud writes are the
    additive computed_metrics cache, the per-record manual_window, and the
    human_classification category."""

    def __init__(self, log=print, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("CD-Shape Review  (worklist · human classification)")
        self.resize(1480, 860)

        self._log = log
        self.records: list[dict] = []          # verified cloud records
        self.results: list = []                # ClassificationResult per record
        self.current_index: int = -1
        self._worklist_filter = "all"          # "all" | "done" | "notdone"
        # All FOUR column lists (worklist + 3 category), so click handlers can
        # clear each other's selection.
        self._lists: list[QListWidget] = []

        # CD analysis-window handle-drag state. The window is now selected by
        # dragging two handle lines ON the spectral plot (no off-plot slider).
        # _win_lo/_win_hi are the current handle positions (nm); _win_bounds is
        # the draggable extent (the record's full spectral span); _win_patches
        # / _handle_lines are the matplotlib artists; _active_handle is the
        # handle being dragged; _dragged guards against a click committing a
        # no-op.
        self._win_lo: float | None = None
        self._win_hi: float | None = None
        self._win_bounds = (float(_PLOT_X_RANGE[0]), float(_PLOT_X_RANGE[1]))
        self._win_patches: list = []
        self._handle_lines: list = []
        self._active_handle: str | None = None   # 'lo' | 'hi' | None
        self._dragged = False
        # True only during a synchronous cloud persist; blocks the plot-drag
        # handlers from re-entering via QApplication.processEvents.
        self._persisting = False

        self._build_ui()
        self.refresh(initial=True)

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)

        banner = QLabel(
            "CD-shape review — the classifier MEASURES each verified cloud "
            "record (UV/CD metrics, no verdict); YOU assign the category. The "
            "worklist on the left lists every record and tracks what's done; "
            "pick Ladder / Staircase / Unsure to sort a record and mark it "
            "reviewed. Computed metrics are cached back to the cloud; spectra "
            "are never modified.")
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "background:#d1ecf1;color:#0c5460;padding:6px;border-radius:4px;")
        root.addWidget(banner)

        # Top control row: ladder %, refresh, force re-sync.
        top = QHBoxLayout()
        self.pct_label = QLabel("—")
        self.pct_label.setStyleSheet("font-weight:bold;")
        top.addWidget(self.pct_label)
        top.addStretch(1)
        self.refresh_btn = QPushButton("Refresh from cloud")
        self.refresh_btn.setToolTip(
            "Re-fetch verified cloud records, re-measure, and sync any stale "
            "computed metrics.")
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        top.addWidget(self.refresh_btn)
        self.resync_btn = QPushButton("Re-sync ALL metrics")
        self.resync_btn.setToolTip(
            "Force-write the computed_metrics cache for every record "
            "(use after changing the thresholds/windows in classifier.py).")
        self.resync_btn.clicked.connect(self._on_force_resync)
        top.addWidget(self.resync_btn)
        root.addLayout(top)

        # Main splitter: [ worklist + category columns ]  |  [ detail ]
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- left: worklist + human-only columns ----
        left = QWidget()
        left_v = QVBoxLayout(left)
        cols = QHBoxLayout()

        # column 0 (far left): WORKLIST -- a permanent roster of ALL records.
        self.worklist_box = QGroupBox("WORKLIST — all records")
        wlv = QVBoxLayout(self.worklist_box)
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Show:"))
        self.worklist_filter_combo = QComboBox()
        self.worklist_filter_combo.addItems(
            ["All", "Done only", "Not-done only"])
        self.worklist_filter_combo.setToolTip(
            "Filter the worklist by review state. 'Done' = has a saved human "
            "classification (Ladder, Staircase, or Unsure).")
        self.worklist_filter_combo.currentIndexChanged.connect(
            self._on_worklist_filter_changed)
        filt_row.addWidget(self.worklist_filter_combo)
        filt_row.addStretch(1)
        wlv.addLayout(filt_row)
        self.worklist = QListWidget()
        wlv.addWidget(self.worklist)
        cols.addWidget(self.worklist_box, stretch=2)

        # human-only category columns
        self.ladder_box = QGroupBox("LADDER")
        lv = QVBoxLayout(self.ladder_box)
        self.ladder_list = QListWidget()
        lv.addWidget(self.ladder_list)
        cols.addWidget(self.ladder_box, stretch=2)

        self.staircase_box = QGroupBox("STAIRCASE")
        sv = QVBoxLayout(self.staircase_box)
        self.staircase_list = QListWidget()
        sv.addWidget(self.staircase_list)
        cols.addWidget(self.staircase_box, stretch=2)

        left_v.addLayout(cols, stretch=1)

        self.flagged_box = QGroupBox("FLAGGED / unsure + bad data")
        fv = QVBoxLayout(self.flagged_box)
        self.flagged_list = QListWidget()
        self.flagged_list.setMaximumHeight(120)
        fv.addWidget(self.flagged_list)
        left_v.addWidget(self.flagged_box)

        legend = QLabel(
            "Worklist holds every record and never empties. "
            "Grey = unreviewed (no human label) · whole-strip green = confirmed "
            "(human classified) · ⚠ = borderline (a ratio within ±%.2f of the "
            "UV or CD peak-ratio threshold). Category columns fill only from "
            "your Ladder/Staircase/Unsure choice; the Flagged section also "
            "holds un-computable 'bad data' records (tagged). "
            "✓ = done · ✎ win = manual CD window."
            % classifier.BORDERLINE_BAND)
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#555;font-size:11px;")
        left_v.addWidget(legend)

        for lst in (self.worklist, self.ladder_list,
                    self.staircase_list, self.flagged_list):
            lst.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection)
            # Over-long labels truncate from the RIGHT with an ellipsis (the full
            # label is on hover via the item tooltip); no mid-row wrapping.
            lst.setTextElideMode(Qt.TextElideMode.ElideRight)
            lst.setWordWrap(False)
            lst.itemClicked.connect(self._on_item_clicked)
            self._lists.append(lst)

        splitter.addWidget(left)

        # ---- right: detail ----
        splitter.addWidget(self._build_detail())

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([720, 740])
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("padding:6px 14px;font-weight:bold;")
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

    def _build_detail(self) -> QWidget:
        detail = QWidget()
        v = QVBoxLayout(detail)

        self.detail_header = QLabel("Select a record to review.")
        self.detail_header.setTextFormat(Qt.TextFormat.RichText)
        self.detail_header.setWordWrap(True)
        v.addWidget(self.detail_header)

        # Review banner (hidden until a record with a saved human label is
        # shown). States only the human's own decision -- no auto verdict.
        self.review_banner = QLabel("")
        self.review_banner.setWordWrap(True)
        self.review_banner.setVisible(False)
        v.addWidget(self.review_banner)

        dsplit = QSplitter(Qt.Orientation.Horizontal)

        # plots
        plot_host = QWidget()
        pv = QVBoxLayout(plot_host)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax_cd, self.ax_g, self.ax_uv = self.figure.subplots(
            3, 1, sharex=True)
        self._axes = {"CD": self.ax_cd, "g": self.ax_g, "UV": self.ax_uv}
        # Dual-handle CD-window selection lives directly ON the plot: press
        # near a handle line to grab it, drag to move, release to re-measure.
        self.canvas.mpl_connect("button_press_event", self._on_plot_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("button_release_event", self._on_plot_release)
        pv.addWidget(self.canvas, stretch=1)
        # View-only zoom/pan toolbar (consistent with the rest of the app):
        # changes displayed limits only, never the data.
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        pv.addWidget(self.toolbar)
        dsplit.addWidget(plot_host)

        # metrics panel
        self.metrics_label = QLabel("")
        self.metrics_label.setTextFormat(Qt.TextFormat.RichText)
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.metrics_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setWidget(self.metrics_label)
        metrics_scroll.setMinimumWidth(280)
        dsplit.addWidget(metrics_scroll)

        dsplit.setStretchFactor(0, 3)
        dsplit.setStretchFactor(1, 2)
        v.addWidget(dsplit, stretch=1)

        # ---- CD analysis-window control: drag the handles ON the plot;
        # this row is the live READ-ONLY readout of where they sit (kept
        # visible because manual_window is persisted per-record and drives
        # computed_metrics, so the exact window must stay auditable). ----
        win_box = QGroupBox("CD analysis window  (drag the handle lines on the "
                            "plot; release to apply + re-measure + save)")
        wv = QVBoxLayout(win_box)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Window:"))
        srow.addWidget(QLabel("min"))
        self.win_min_lbl = QLabel("—")
        self.win_min_lbl.setMinimumWidth(58)
        self.win_min_lbl.setStyleSheet("font-weight:bold;")
        srow.addWidget(self.win_min_lbl)
        srow.addSpacing(12)
        srow.addWidget(QLabel("max"))
        self.win_max_lbl = QLabel("—")
        self.win_max_lbl.setMinimumWidth(58)
        self.win_max_lbl.setStyleSheet("font-weight:bold;")
        srow.addWidget(self.win_max_lbl)
        srow.addStretch(1)
        self.reset_window_btn = QPushButton("Reset to default")
        self.reset_window_btn.setToolTip(
            "Clear this record's manual window and revert to the default CD "
            "window (re-saves the refreshed metrics cache).")
        self.reset_window_btn.clicked.connect(self._on_reset_window)
        srow.addWidget(self.reset_window_btn)
        wv.addLayout(srow)
        self.window_status = QLabel("")
        self.window_status.setTextFormat(Qt.TextFormat.RichText)
        self.window_status.setWordWrap(True)
        self.window_status.setStyleSheet("color:#555;font-size:11px;")
        wv.addWidget(self.window_status)
        v.addWidget(win_box)

        # human classification controls -- the ONLY category label
        rc = QGroupBox("Classify  (your decision — the only category label)")
        rc_h = QHBoxLayout(rc)
        rc_h.addWidget(QLabel("Human:"))
        self.human_group = QButtonGroup(self)
        self.human_group.setExclusive(True)
        self._human_btns = {}
        for lab, text in ((classifier.HUMAN_LADDER, "Ladder"),
                          (classifier.HUMAN_STAIRCASE, "Staircase"),
                          (classifier.HUMAN_UNSURE, "Unsure")):
            btn = QRadioButton(text)
            self.human_group.addButton(btn)
            self._human_btns[lab] = btn
            rc_h.addWidget(btn)
        self.save_override_btn = QPushButton("Save")
        self.save_override_btn.clicked.connect(self._save_override)
        rc_h.addWidget(self.save_override_btn)
        self.clear_override_btn = QPushButton("Clear saved")
        self.clear_override_btn.setToolTip(
            "Delete this record's saved classification and return it to "
            "unreviewed (grey / not-done).")
        self.clear_override_btn.clicked.connect(self._clear_override)
        rc_h.addWidget(self.clear_override_btn)
        rc_h.addStretch(1)
        self.override_status = QLabel("")
        self.override_status.setStyleSheet("color:#555;")
        rc_h.addWidget(self.override_status)
        self._set_detail_enabled(False)
        v.addWidget(rc)

        return detail

    # --------------------------------------------------------- data load ----
    def refresh(self, initial: bool = False):
        """Re-fetch verified cloud records, measure, sync stale metrics, and
        rebuild the worklist + columns. Network runs on the main thread (quick +
        guarded); processEvents keeps the UI responsive while it blocks."""
        self.pct_label.setText("Loading…")
        self.refresh_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            self._fetch_and_classify()
        except Exception as e:
            self._log(f"CD review fetch error: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            self.records, self.results = [], []
        finally:
            self.refresh_btn.setEnabled(True)

        self._populate_columns()
        # Self-healing cache: write only the missing/stale computed metrics.
        self._sync_metrics(force=False)

        if self.records:
            self._load_detail(0)
        else:
            self._show_empty_detail()

    def _fetch_and_classify(self):
        from mongo_db import fetch_records
        records = fetch_records({}, log=self._log)
        # Belt-and-suspenders: cloud docs are verified by construction, but
        # honor the "verified records" contract explicitly.
        records = [r for r in records if classifier._is_verified(r)]
        self.records = records
        self.results = [classifier.classify_record(r) for r in records]

    # ----------------------------------------------------------- columns ----
    def _populate_columns(self):
        """Rebuild the worklist (ALL records, filtered) and the category columns.
        Ladder/Staircase fill from human_classification only; the Flagged section
        holds human=Unsure records AND un-computable 'bad data' records that have
        no human label yet."""
        for lst in self._lists:
            lst.clear()
        show = self._worklist_filter
        n_ladder = n_stair = n_unsure = n_baddata = n_reviewed = 0
        for idx, (rec, res) in enumerate(zip(self.records, self.results)):
            h = _human_label(rec)
            done = bool(h)
            # Which category column (if any) does this record belong in? A human
            # label wins; otherwise an un-computable record falls to Flagged.
            cat = None
            if h == classifier.HUMAN_LADDER:
                cat = self.ladder_list
                n_ladder += 1
            elif h == classifier.HUMAN_STAIRCASE:
                cat = self.staircase_list
                n_stair += 1
            elif h == classifier.HUMAN_UNSURE:
                cat = self.flagged_list
                n_unsure += 1
            elif not done and not res.computable:
                cat = self.flagged_list        # bad data, not yet human-reviewed
                n_baddata += 1
            if done:
                n_reviewed += 1
            # Worklist: a PERMANENT roster of every record, honoring the filter.
            if (show == "all" or (show == "done" and done)
                    or (show == "notdone" and not done)):
                self.worklist.addItem(self._make_item(idx, rec, res))
            # Category column (human label, or bad-data -> Flagged).
            if cat is not None:
                cat.addItem(self._make_item(idx, rec, res))

        total = len(self.records)
        n_notdone = total - n_reviewed
        n_flag = n_unsure + n_baddata
        self.worklist_box.setTitle(f"WORKLIST — all records  ({total})")
        self.ladder_box.setTitle(f"LADDER  ({n_ladder})")
        self.staircase_box.setTitle(f"STAIRCASE  ({n_stair})")
        self.flagged_box.setTitle(f"FLAGGED / unsure + bad data  ({n_flag})")
        # Honest denominators: ladder as a fraction of REVIEWED, then the raw
        # not-done / total / borderline counts.
        pct = (100.0 * n_ladder / n_reviewed) if n_reviewed else 0.0
        n_border = sum(1 for r in self.results if r.borderline)
        self.pct_label.setText(
            f"Ladder {n_ladder} of {n_reviewed} reviewed ({pct:.0f}%)  "
            f"·  {n_notdone} not done  ·  {total} total  "
            f"·  {n_border} borderline")

    def _make_item(self, idx: int, rec: dict, res) -> QListWidgetItem:
        item = QListWidgetItem(self._item_text(rec, res))
        item.setData(Qt.ItemDataRole.UserRole, idx)
        item.setBackground(QBrush(self._item_brush(rec)))
        # The FULL label (untruncated) plus the hex record_id and filename live
        # in the tooltip, so a row elided to fit the column is still readable on
        # hover and the record_id stays available as the join key.
        rid = rec.get("record_id") or "(no id)"
        fname = rec.get("filename") or ""
        tip = f"{_record_label(rec)}\nrecord_id: {rid}"
        if fname:
            tip += f"\nfile: {fname}"
        item.setToolTip(tip)
        return item

    def _item_text(self, rec: dict, res) -> str:
        # Lead with the metadata-rich label (components | ratio | conc+solvent |
        # anneal) -- same everywhere. The hex record_id / series are NOT shown
        # inline; the full label + id live in the tooltip (set in _make_item).
        lead = _record_label(rec)
        badges = []
        h = _human_label(rec)
        if h:
            badges.append(f"✓ {h}")          # DONE marker = the human category
        if not res.computable:
            badges.append("bad data")        # classifier couldn't measure it
        if res.borderline:
            badges.append("⚠ borderline")
        # Mark records running on a hand-set window vs. the default.
        if classifier._resolve_manual_window(rec.get("manual_window")) is not None:
            badges.append("✎ win")
        tail = ("  [" + " · ".join(badges) + "]") if badges else ""
        return f"{lead}{tail}"

    def _item_brush(self, rec: dict) -> QColor:
        # Lifecycle: grey (unreviewed) -> whole-strip green on classification.
        return _CLASSIFIED_GREEN if _human_label(rec) else _UNREVIEWED_GREY

    def _on_worklist_filter_changed(self, index: int):
        self._worklist_filter = {0: "all", 1: "done", 2: "notdone"}.get(
            index, "all")
        self._populate_columns()
        if 0 <= self.current_index < len(self.records):
            self._select_index_in_columns(self.current_index)

    def _on_item_clicked(self, item: QListWidgetItem):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        # Single visual selection across the lists.
        for lst in self._lists:
            if lst is not item.listWidget():
                lst.blockSignals(True)
                lst.clearSelection()
                lst.setCurrentRow(-1)
                lst.blockSignals(False)
        self._load_detail(int(idx))

    def _select_index_in_columns(self, idx: int):
        for lst in self._lists:
            for row in range(lst.count()):
                it = lst.item(row)
                if it.data(Qt.ItemDataRole.UserRole) == idx:
                    lst.setCurrentItem(it)
                    lst.scrollToItem(it)
                    return

    # ------------------------------------------------------ detail view -----
    def _show_empty_detail(self):
        self.current_index = -1
        self.detail_header.setText(
            "<b>No verified cloud records.</b> The cloud may be "
            "unconfigured/unreachable, or empty — see the log.")
        self.metrics_label.setText("")
        self.review_banner.setVisible(False)
        self.window_status.setText("")
        self.win_min_lbl.setText("—")
        self.win_max_lbl.setText("—")
        self._win_lo = self._win_hi = None
        self._active_handle = None
        self._win_patches = []
        self._handle_lines = []
        for ax in self._axes.values():
            ax.clear()
        self._draw_axes_chrome()
        self.canvas.draw_idle()
        self._set_detail_enabled(False)

    def _load_detail(self, idx: int):
        if not (0 <= idx < len(self.records)):
            return
        self.current_index = idx
        rec, res = self.records[idx], self.results[idx]
        rid = rec.get("record_id") or "(no id)"
        fname = rec.get("filename") or ""
        # Lead with the human identity; keep the title to ONE line by truncating
        # a long filename (full filename + record_id available on hover).
        series = rec.get("series") or "?"
        head = (f"<b>Record {idx + 1} of {len(self.records)}</b> "
                f"&nbsp;·&nbsp; {series} · {_polymer_system(rec)}")
        if fname:
            fdisp = fname if len(fname) <= 44 else fname[:43] + "…"
            head += (f" &nbsp;·&nbsp; <span style='color:#555;"
                     f"font-size:11px;'>{fdisp}</span>")
        self.detail_header.setText(head)
        self.detail_header.setToolTip(
            f"record_id: {rid}" + (f"\nfile: {fname}" if fname else ""))
        self.metrics_label.setText(self._metrics_html(rec, res))
        self._seed_window(rec)
        self._draw_detail_plots(idx)
        self._set_detail_enabled(True)
        self._set_human_radios(_human_label(rec))
        self._refresh_override_banner(rec, res)
        self._update_window_status(rec, res)
        self._select_index_in_columns(idx)

    def _metrics_html(self, rec: dict, res) -> str:
        g = res.gates
        psmin, psmax = (classifier.UV_PEAK_SEARCH_MIN,
                        classifier.UV_PEAK_SEARCH_MAX)
        t_uv = classifier.UV_RATIO_THRESHOLD
        t_lobe = classifier.LOBE_RATIO_THRESHOLD

        def chip(ok, pass_text="PASS", fail_text="FAIL"):
            color = "#28a745" if ok else "#c82333"
            return (f"<span style='color:white;background:{color};"
                    f"padding:1px 6px;border-radius:3px;'>"
                    f"{pass_text if ok else fail_text}</span>")

        bl = ""
        if res.borderline:
            bl = (" &nbsp;<span style='background:#ffe5b4;padding:1px 5px;"
                  "border-radius:3px;'>BORDERLINE: "
                  f"{', '.join(res.borderline_flags)}</span>")

        # CD window line: the SINGLE window the metrics were computed under
        # (res.window_*), labelled manual vs default from res.window_source --
        # the SAME values the status line and the plotted markers read.
        if res.window_left is not None and res.window_source:
            src = res.window_source            # "manual" | "default"
            colour = "#7a5c00" if src == "manual" else "#444"
            cd_window = (f"CD window <span style='color:{colour};"
                         f"font-weight:bold;'>({src})</span>: "
                         f"<b>[{_wl(res.window_left)} … {_wl(res.window_right)}]</b>")
        else:
            cd_window = "CD window: <b>— (not measured)</b>"

        uv_ratio_metric = (f"peak2/peak1 (raw) = "
                           f"<b>{_f(res.uv_two_peak_ratio, 2)}</b> "
                           f"(threshold ≥ {t_uv})")
        lobe_metric = (f"min/max lobe = <b>{_f(res.lobe_ratio, 2)}</b> "
                       f"(threshold ≥ {t_lobe})")

        # The reviewer's own decision (not a suggested verdict).
        human = _human_label(rec)
        human_line = (f"<b style='color:#155724;'>{human.upper()}</b>"
                      if human else
                      "<span style='color:#777;'>unreviewed</span>")

        # Couplet cue: when a couplet IS present but a peak-ratio gate fails, say
        # so explicitly so "yes" doesn't read as contradictory above failed gates.
        if res.bisignate:
            evolved = res.gates.uv_ratio_pass and res.gates.lobe_ratio_pass
            couplet_txt = ("yes" if evolved
                           else "yes — couplet present (not yet evolved)")
        else:
            couplet_txt = "no"

        notes = res.notes or "—"

        return f"""
        <div style='font-size:13px;'>
        <h3 style='margin:2px 0;'>Computed metrics
          <span style='font-size:11px;color:#777;'>(no verdict — you decide)</span>{bl}</h3>
        <div style='color:#444;'>your classification: {human_line}
          &nbsp;·&nbsp; bisignate couplet: <b>{couplet_txt}</b></div>
        <hr>
        <b>UV bands ({psmin:.0f}–{psmax:.0f} nm search)</b>
          &nbsp;{chip(g.uv_band_detected, "2 BANDS", "1 BAND")}<br>
        &nbsp;&nbsp;baseline (tail, display-only): <b>{_f(res.uv_baseline)}</b><br>
        &nbsp;&nbsp;peak1 (shorter λ): <b>{_f(res.uv_peak1.value)}</b> @ {_wl(res.uv_peak1.wl)}<br>
        &nbsp;&nbsp;peak2 (longer λ): <b>{_f(res.uv_peak2.value)}</b> @ {_wl(res.uv_peak2.wl)}<br>
        &nbsp;&nbsp;{cd_window}<br>
        <br>
        <b>UV peak ratio</b> &nbsp;{chip(g.uv_ratio_pass)}<br>
        &nbsp;&nbsp;{uv_ratio_metric}<br>
        <br>
        <b>CD couplet (inside window)</b>
          &nbsp;{chip(g.cd_couplet, "COUPLET", "NONE")}<br>
        &nbsp;&nbsp;positive lobe: <b>{_f(res.cd_pos_lobe.value)}</b> @ {_wl(res.cd_pos_lobe.wl)}<br>
        &nbsp;&nbsp;negative lobe: <b>{_f(res.cd_neg_lobe.value)}</b> @ {_wl(res.cd_neg_lobe.wl)}<br>
        &nbsp;&nbsp;crossover (interp): <b>{_wl(res.crossover_wavelength)}</b><br>
        <br>
        <b>CD peak ratio</b> &nbsp;{chip(g.lobe_ratio_pass)}<br>
        &nbsp;&nbsp;{lobe_metric}<br>
        <hr>
        <b>notes:</b> {notes}
        </div>
        """

    # ----- plots -----
    def _signal_arrays(self, rec: dict, sig: str):
        wl_raw = rec.get("wavelength")
        y_raw = rec.get(_SIGNAL_DOC_KEY[sig])
        if not wl_raw or not y_raw:
            return None
        wl = np.asarray(wl_raw, dtype=float)
        y = np.asarray(y_raw, dtype=float)
        n = min(len(wl), len(y))
        if n == 0:
            return None
        return wl[:n], y[:n]

    def _draw_axes_chrome(self):
        self.ax_cd.set_ylabel("CD")
        self.ax_g.set_ylabel("g")
        self.ax_uv.set_ylabel("UV")
        self.ax_uv.set_xlabel("Wavelength (nm)")
        self.ax_cd.set_xlim(*_PLOT_X_RANGE)   # sharex propagates to g + uv
        for sig, ax in self._axes.items():
            ax.set_title(_SIGNAL_TITLES[sig], fontsize=10)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    def _redraw_window_artists(self):
        """(Re)draw the analysis window from self._win_lo/_win_hi: a shaded span
        plus a draggable handle line at each edge, on BOTH the CD and UV axes
        (the CD couplet is read inside it). The CD and UV axes share one x-range
        (sharex), so the box and handles stay vertically aligned across both.
        Cheap enough to call on every drag step -- it touches only the window
        artists, never the spectral curves. A no-op draws nothing when the
        window is unset/degenerate."""
        for art in self._win_patches + self._handle_lines:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass
        self._win_patches = []
        self._handle_lines = []
        lo, hi = self._win_lo, self._win_hi
        if lo is not None and hi is not None and hi > lo:
            for ax in (self.ax_cd, self.ax_uv):
                self._win_patches.append(
                    ax.axvspan(lo, hi, color="#ffd24d", alpha=0.15, zorder=0))
                for x in (lo, hi):
                    self._handle_lines.append(
                        ax.axvline(x, color="#f0a500", linewidth=2.0,
                                   zorder=8))
        self.canvas.draw_idle()

    def _draw_detail_plots(self, idx: int):
        rec, res = self.records[idx], self.results[idx]
        # ax.clear() drops the old window artists, so reset our handles to them.
        self._win_patches = []
        self._handle_lines = []
        for ax in self._axes.values():
            ax.clear()

        # raw embedded arrays, drawn faithfully (no smoothing)
        for sig, ax in self._axes.items():
            arr = self._signal_arrays(rec, sig)
            if arr is not None:
                wl, y = arr
                ax.plot(wl, y, color=_SIGNAL_COLORS[sig], linewidth=1.3)

        # --- analysis window: shade + draggable handles on BOTH CD and UV at
        # the CURRENT handle position (self._win_lo/_win_hi). _seed_window set
        # these from the record's manual_window, or the default 400/550 start.
        # On a live drag the box/handles follow the cursor; on release the
        # record is re-measured so the committed manual window == the handles.
        self._redraw_window_artists()

        # --- CD: mark the two lobes + the zero-crossing (crossover) ---
        pos, neg = res.cd_pos_lobe, res.cd_neg_lobe
        if pos.wl is not None and pos.value is not None:
            self._mark(self.ax_cd, pos.wl, pos.value, "^",
                       "#d62728", f"+lobe {pos.wl:.0f}nm")
        if neg.wl is not None and neg.value is not None:
            self._mark(self.ax_cd, neg.wl, neg.value, "v",
                       "#1f77b4", f"-lobe {neg.wl:.0f}nm", below=True)
        if res.crossover_wavelength is not None:
            self.ax_cd.axvline(res.crossover_wavelength, color="#6f42c1",
                               linestyle="--", linewidth=1.1, zorder=5)
            self.ax_cd.annotate(
                f"x0 {res.crossover_wavelength:.0f}nm",
                (res.crossover_wavelength, 0.0), textcoords="offset points",
                xytext=(4, 4), fontsize=8, color="#6f42c1", zorder=7)

        # --- UV: draw the baseline and mark peak1 / peak2. ---
        if res.uv_baseline is not None:
            self.ax_uv.axhline(res.uv_baseline, color="#999999",
                               linestyle=":", linewidth=1.0, zorder=1)
        p1, p2 = res.uv_peak1, res.uv_peak2
        if p1.wl is not None and p1.value is not None:
            self._mark(self.ax_uv, p1.wl, p1.value, "o",
                       "#d62728", f"peak1 {p1.wl:.0f}nm")
        if p2.wl is not None and p2.value is not None:
            self._mark(self.ax_uv, p2.wl, p2.value, "s",
                       "#9467bd", f"peak2 {p2.wl:.0f}nm")

        self._draw_axes_chrome()
        self.toolbar.update()      # reset zoom/pan history for the new record
        self.canvas.draw_idle()

    def _mark(self, ax, x, y, marker, color, label, below=False):
        ax.plot([x], [y], marker, color=color, markersize=8, zorder=6)
        ax.annotate(
            label, (x, y), textcoords="offset points",
            xytext=(4, -12 if below else 6), fontsize=8, color=color,
            zorder=7)

    # --------------------------------------------- analysis-window slider ----
    def _window_bounds(self, rec: dict) -> tuple:
        """The draggable extent for the window handles: the record's FULL
        spectral span (so a bisignate couplet's lobes outside the 400-550
        default are still reachable). Falls back to the plot's fixed x-range
        when the wavelength array is missing/odd."""
        wl = rec.get("wavelength")
        if wl:
            try:
                arr = [float(v) for v in wl]
            except (TypeError, ValueError):
                arr = []
            if arr:
                lo, hi = min(arr), max(arr)
                if hi > lo:
                    return lo, hi
        return float(_PLOT_X_RANGE[0]), float(_PLOT_X_RANGE[1])

    def _seed_window(self, rec: dict):
        """Set the handle positions for a record: its stored manual_window if
        present, else the default START position (_DEFAULT_WINDOW_LO/HI =
        400/550), clamped into the record's draggable bounds. Only updates the
        in-memory handle state + the nm readout; the artists are (re)drawn by
        the subsequent _draw_detail_plots. No re-measure/commit happens here --
        restore-on-open stays a pure read of manual_window."""
        self._win_bounds = self._window_bounds(rec)
        lo_b, hi_b = self._win_bounds
        mw = classifier._resolve_manual_window(rec.get("manual_window"))
        if mw is not None:
            lo, hi = mw
        else:
            lo, hi = _DEFAULT_WINDOW_LO, _DEFAULT_WINDOW_HI
        lo = max(lo_b, min(hi_b, float(lo)))
        hi = max(lo, min(hi_b, float(hi)))
        self._win_lo, self._win_hi = lo, hi
        self._active_handle = None
        self.win_min_lbl.setText(f"{lo:.0f} nm")
        self.win_max_lbl.setText(f"{hi:.0f} nm")

    def _update_window_status(self, rec: dict, res):
        mw = classifier._resolve_manual_window(rec.get("manual_window"))
        # Where the shaded handles currently sit (always set once a record is
        # loaded; guarded just in case).
        handles = (f"{self._win_lo:.0f}–{self._win_hi:.0f} nm"
                   if self._win_lo is not None else "—")
        if mw is not None:
            self.window_status.setText(
                f"<b>Manual window</b> {mw[0]:.0f}–{mw[1]:.0f} nm. Drag the "
                f"handles on the plot to adjust, or <i>Reset to default</i>.")
        elif res.window_left is not None:
            self.window_status.setText(
                f"No manual window set — metrics use the <b>default</b> "
                f"{res.window_left:.0f}–{res.window_right:.0f} nm. The shaded "
                f"handles start at {handles}; drag and release on the plot to "
                f"set a manual window for this record.")
        else:
            self.window_status.setText(
                "No manual window set, and this record has no measurable CD "
                "window (un-computable / single UV band). Drag the shaded "
                "handles on the plot and release to set one.")

    # ---- on-plot handle dragging (x in nm; CD/g/UV share one x-axis) ----
    def _nm_to_px(self, nm: float) -> float:
        return float(self.ax_cd.transData.transform((nm, 0.0))[0])

    def _px_to_nm(self, px: float) -> float:
        return float(self.ax_cd.transData.inverted().transform((px, 0.0))[0])

    def _handle_near(self, px: float):
        """Which window handle ('lo'/'hi') is within grab range of pixel-x px,
        or None. Ties (overlapped handles) resolve to 'lo'."""
        if self._win_lo is None or px is None:
            return None
        d_lo = abs(px - self._nm_to_px(self._win_lo))
        d_hi = abs(px - self._nm_to_px(self._win_hi))
        if min(d_lo, d_hi) > _HANDLE_GRAB_PX:
            return None
        return "lo" if d_lo <= d_hi else "hi"

    def _on_plot_press(self, event):
        """Grab the nearest window handle if the click landed near one. Yields
        to the matplotlib zoom/pan tools when they're active so they keep
        working; does not move the handle yet (a no-motion click commits
        nothing)."""
        if (event.button != 1 or self.current_index < 0
                or self._win_lo is None or event.x is None
                or self._persisting):
            return
        if getattr(self.toolbar, "mode", ""):          # zoom/pan tool active
            return
        if event.inaxes not in (self.ax_cd, self.ax_g, self.ax_uv):
            return
        self._active_handle = self._handle_near(event.x)
        self._dragged = False

    def _on_plot_motion(self, event):
        """While a handle is grabbed: move it (clamped to the draggable bounds,
        no crossing), update the nm readout, and redraw the box + handles. When
        idle, give resize-cursor feedback near a handle. Re-measure is DEFERRED
        to release."""
        if self._active_handle is None:
            self._update_hover_cursor(event)
            return
        if event.x is None:
            return
        lo_b, hi_b = self._win_bounds
        nm = max(lo_b, min(hi_b, self._px_to_nm(event.x)))
        if self._active_handle == "lo":
            self._win_lo = min(nm, self._win_hi)
        else:
            self._win_hi = max(nm, self._win_lo)
        self._dragged = True
        self.win_min_lbl.setText(f"{self._win_lo:.0f} nm")
        self.win_max_lbl.setText(f"{self._win_hi:.0f} nm")
        self._redraw_window_artists()

    def _on_plot_release(self, _event):
        """On handle release, commit the window (re-measure + persist) -- same
        commit point the old slider used. A click with no drag commits
        nothing."""
        if self._active_handle is None:
            return
        self._active_handle = None
        if not self._dragged:
            return
        self._dragged = False
        self._commit_window()

    def _update_hover_cursor(self, event):
        over = (event is not None and event.x is not None
                and event.inaxes in (self.ax_cd, self.ax_g, self.ax_uv)
                and not getattr(self.toolbar, "mode", "")
                and self._handle_near(event.x) is not None)
        self.canvas.setCursor(
            Qt.CursorShape.SplitHCursor if over
            else Qt.CursorShape.ArrowCursor)

    def _commit_window(self):
        """Set this record's manual_window from the current handle positions,
        re-measure under it, persist BOTH the window and the refreshed metrics
        cache together, and refresh the metrics + markers + columns. Unchanged
        persistence/recompute semantics -- only the input affordance moved from
        a slider to the on-plot handles."""
        if not (0 <= self.current_index < len(self.records)):
            return
        idx = self.current_index
        rec = self.records[idx]
        lo, hi = self._win_lo, self._win_hi
        mw = {"min_nm": round(float(lo), 1), "max_nm": round(float(hi), 1)}
        rec["manual_window"] = mw                       # mirror locally
        res = classifier.classify_record(rec)           # recompute under window
        self.results[idx] = res
        metrics = classifier.computed_metrics_doc(res)
        rec["computed_metrics"] = metrics               # mirror locally
        rec.pop("auto_classification", None)            # mirror legacy migration
        self.win_min_lbl.setText(f"{lo:.0f} nm")
        self.win_max_lbl.setText(f"{hi:.0f} nm")
        self.metrics_label.setText(self._metrics_html(rec, res))
        self._draw_detail_plots(idx)
        self._update_window_status(rec, res)
        self._populate_columns()
        self._select_index_in_columns(idx)
        self._persist_window(rec.get("record_id"), mw, metrics)

    def _on_reset_window(self):
        """Clear this record's manual_window (revert to the default window),
        re-measure, and persist the cleared field + refreshed cache together."""
        if not (0 <= self.current_index < len(self.records)):
            return
        idx = self.current_index
        rec = self.records[idx]
        rec.pop("manual_window", None)                  # mirror clear locally
        res = classifier.classify_record(rec)           # default window again
        self.results[idx] = res
        metrics = classifier.computed_metrics_doc(res)
        rec["computed_metrics"] = metrics
        rec.pop("auto_classification", None)            # mirror legacy migration
        self._seed_window(rec)                          # back to default seed
        self.metrics_label.setText(self._metrics_html(rec, res))
        self._draw_detail_plots(idx)
        self._update_window_status(rec, res)
        self._populate_columns()
        self._select_index_in_columns(idx)
        self._persist_window(rec.get("record_id"), None, metrics)

    def _persist_window(self, rid, manual_window, metrics):
        """Atomically write manual_window (or clear it) + the refreshed metrics
        cache via mongo_db.set_manual_window. Guarded; a failure is surfaced in
        the window status line."""
        if not rid:
            self._log("Manual window not persisted: record has no record_id.")
            self.window_status.setText(
                "⚠ Not saved to cloud: this record has no record_id.")
            return
        self._persisting = True
        self.reset_window_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from mongo_db import set_manual_window
            res = set_manual_window(rid, manual_window, metrics, log=self._log)
        except Exception as e:
            self._log(f"Manual window save failed: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            res = {"ok": False, "message": str(e)}
        finally:
            self._persisting = False
            self.reset_window_btn.setEnabled(True)
        if not res.get("ok"):
            self.window_status.setText(
                "⚠ Cloud save failed: " + res.get("message", "unknown error"))

    # ------------------------------------------------- human classify -------
    def _set_detail_enabled(self, enabled: bool):
        for btn in self._human_btns.values():
            btn.setEnabled(enabled)
        self.save_override_btn.setEnabled(enabled)
        self.clear_override_btn.setEnabled(enabled)
        self.reset_window_btn.setEnabled(enabled)

    def _set_human_radios(self, label):
        # Allow "none checked" by toggling exclusivity around the change.
        self.human_group.setExclusive(False)
        for lab, btn in self._human_btns.items():
            btn.setChecked(lab == label)
        self.human_group.setExclusive(True)

    def _selected_human_label(self):
        for lab, btn in self._human_btns.items():
            if btn.isChecked():
                return lab
        return None

    def _save_override(self):
        if not (0 <= self.current_index < len(self.records)):
            return
        rec = self.records[self.current_index]
        rid = rec.get("record_id")
        label = self._selected_human_label()
        if not label:
            self.override_status.setText("Pick Ladder / Staircase / Unsure first.")
            return
        if not rid:
            self.override_status.setText("This record has no record_id — "
                                         "cannot save classification.")
            self._log("Classification not saved: record has no record_id.")
            return

        human_doc = {"label": label,
                     "reviewed_at": datetime.now(timezone.utc).isoformat()}
        self.save_override_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from mongo_db import set_human_classification
            res = set_human_classification(rid, human_doc, log=self._log)
        except Exception as e:
            self._log(f"Classification save failed: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            res = {"ok": False, "message": str(e)}
        finally:
            self.save_override_btn.setEnabled(True)

        if res.get("ok"):
            rec["human_classification"] = human_doc       # mirror locally
            self.override_status.setText(
                f"Saved: {label} · {human_doc['reviewed_at'][:19]}Z")
            self._populate_columns()
            self._select_index_in_columns(self.current_index)
            self._refresh_override_banner(rec, self.results[self.current_index])
        else:
            self.override_status.setText(res.get("message", "Save failed."))

    def _clear_override(self):
        """Delete a saved human_classification (returns the record to grey /
        unreviewed and out of its category column). If nothing is saved, just
        unselect the radios locally."""
        if not (0 <= self.current_index < len(self.records)):
            return
        rec = self.records[self.current_index]
        rid = rec.get("record_id")
        if not _human_label(rec):
            self._set_human_radios(None)
            self.override_status.setText("No saved classification to clear.")
            return
        if not rid:
            self.override_status.setText("This record has no record_id — "
                                         "cannot clear.")
            return

        self.clear_override_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from mongo_db import set_human_classification
            # None -> $unset the field on the cloud doc.
            res = set_human_classification(rid, None, log=self._log)
        except Exception as e:
            self._log(f"Clear classification failed: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            res = {"ok": False, "message": str(e)}
        finally:
            self.clear_override_btn.setEnabled(True)

        if res.get("ok"):
            rec.pop("human_classification", None)          # mirror locally
            self._set_human_radios(None)
            self.override_status.setText("Cleared — back to unreviewed.")
            self._populate_columns()
            self._select_index_in_columns(self.current_index)
            self._refresh_override_banner(rec, self.results[self.current_index])
        else:
            self.override_status.setText(res.get("message", "Clear failed."))

    def _refresh_override_banner(self, rec: dict, res):
        h = _human_label(rec)
        if not h:
            self.review_banner.setVisible(False)
            self.override_status.setText("Not yet classified.")
            return
        hd = rec.get("human_classification") or {}
        when = (hd.get("reviewed_at") or "")[:19]
        self.override_status.setText(
            f"Saved: {h}" + (f" · {when}Z" if when else ""))
        self._banner(
            "#d4edda", "#155724",
            f"Classified as <b>{h.upper()}</b>"
            + (f" · {when}Z" if when else "")
            + " — <i>Clear saved</i> returns it to unreviewed.")

    def _banner(self, bg, fg, html):
        self.review_banner.setText(html)
        self.review_banner.setStyleSheet(
            f"background:{bg};color:{fg};padding:5px;border-radius:4px;")
        self.review_banner.setVisible(True)

    # ----------------------------------------------------- cloud caching ----
    def _sync_metrics(self, force: bool):
        """Write the computed_metrics cache for stale/missing records (or all,
        when force=True). Also migrates legacy docs: the cloud writers $unset the
        old auto_classification block, and a record still carrying it is treated
        as stale so it gets rewritten. Idempotent; updates the in-memory mirror
        so repeated syncs become no-ops until the measurement output changes."""
        pairs = []
        for rec, res in zip(self.records, self.results):
            rid = rec.get("record_id")
            metrics = classifier.computed_metrics_doc(res)
            stored = rec.get("computed_metrics")
            legacy = rec.get("auto_classification") is not None
            if force or legacy or not _metrics_cache_current(stored, metrics):
                rec["computed_metrics"] = metrics          # mirror locally
                rec.pop("auto_classification", None)        # mirror migration
                if rid:
                    pairs.append((rid, metrics))
        if not pairs:
            self._log("Computed-metrics cache already current "
                      "(nothing to sync).")
            return
        try:
            from mongo_db import sync_computed_metrics
            sync_computed_metrics(pairs, log=self._log)
        except Exception as e:
            self._log(f"Computed-metrics sync error: "
                      f"{type(e).__name__}: {e}")
            self._log(traceback.format_exc())

    def _on_force_resync(self):
        if not self.records:
            self._log("No records to re-sync.")
            return
        self.resync_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            self._log(f"Force re-syncing computed metrics for "
                      f"{len(self.records)} record(s)…")
            self._sync_metrics(force=True)
        finally:
            self.resync_btn.setEnabled(True)
