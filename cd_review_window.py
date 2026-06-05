"""
CD-shape REVIEW window -- Phase B visual audit / validation layer for the
Phase A classifier (`classifier.py`).

A NEW, SEPARATE window from the read-only Cloud Browser (this module never
imports or modifies `cloud_browser_window`). It reads the SAME source -- the
verified CLOUD (MongoDB Atlas) records via `mongo_db.fetch_records` -- runs
`classifier.classify_record` on each, and lays them out in two columns by hard
label (LADDER | STAIRCASE) plus a small FLAGGED section for un-computable /
bad-data records. Borderline records stay in their computed column with a clear
visual marker rather than getting their own column.

Clicking an entry opens a single-record DETAIL view: three stacked,
Origin-faithful matplotlib subplots (CD / g / UV, raw embedded arrays drawn as
plain lines -- no smoothing) with the classifier's selected peaks marked and
its search windows shaded, plus a metrics panel that makes the classifier's
REASONING auditable.

WRITES (the only ones this window performs -- spectra are NEVER touched):
  - auto_classification : refreshable cache of the classifier output, synced to
    each cloud doc keyed by record_id. Idempotent; re-synced when stale (e.g.
    after the PROVISIONAL thresholds in classifier.py change, OR when a record's
    manual_window changes -- the cache records window_used and is invalidated by
    a window edit).
  - manual_window       : per-record CD analysis window {min_nm, max_nm} set via
    the dual-handle range slider. Overrides the data-driven window. Persisted
    TOGETHER with a freshly recomputed auto_classification (one atomic write via
    mongo_db.set_manual_window) so the two never drift out of sync. "Reset to
    auto" $unsets it and reverts the record to the data-driven window.
  - human_classification: durable reviewer override (Ladder/Staircase/Unsure +
    timestamp), written on demand. Never overwrites the auto label -- both are
    kept side by side; disagreements are surfaced as the tuning signal.

Thresholds / the default window seed are read from classifier.py's named
constants -- not duplicated here.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRect, QRectF, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QDialog, QFrame,
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

# Row tints. Disagreement (human override differs from auto) wins over
# borderline, which wins over the plain per-label tint -- so the most
# tuning-relevant rows are the most visually prominent.
_LADDER_TINT = QColor("#d4edda")      # light green
_STAIRCASE_TINT = QColor("#e7f1fb")   # light blue
_FLAGGED_TINT = QColor("#d3d3d3")     # gray
_BORDERLINE_TINT = QColor("#ffe5b4")  # peach -- "eyeball this one"
_DISAGREE_TINT = QColor("#f8d7da")    # pink/red -- human vs auto disagreement


def _human_readable(rec: dict) -> str:
    """series | polymers | solvent | film -- the short human label for a row."""
    series = rec.get("series") or "?"
    p1 = rec.get("p1_name") or "?"
    p2 = rec.get("p2_name")
    poly = p1 if (not p2 or p2 == "None") else f"{p1}+{p2}"
    solvent = rec.get("solvent") or "?"
    film = rec.get("film_state") or "?"
    return f"{series} | {poly} | {solvent} | {film}"


def _human_label(rec: dict):
    """The stored human override label ('ladder'/'staircase'/'unsure') or None."""
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


def _auto_cache_current(stored, fresh) -> bool:
    """True if the stored auto_classification cache still matches a freshly
    computed one -- so the sync pass can skip an idempotent no-op write.

    A change in the PROVISIONAL thresholds snapshot, the WINDOW the record was
    classified under, the hard label, the borderline flag, or either evolution
    ratio (to 6 dp) marks the cache STALE so it is rewritten. The window is
    included because auto_classification is a cache of BOTH the constants and the
    window used -- a manual_window edit must invalidate it even if the label is
    unchanged. Everything else is derived from these, so this is sufficient.
    """
    if not isinstance(stored, dict):
        return False
    if stored.get("thresholds") != fresh.get("thresholds"):
        return False
    if stored.get("label") != fresh.get("label"):
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


class RangeSlider(QWidget):
    """Minimal native dual-handle range slider over a float [minimum, maximum]
    span (data units = nm). Self-contained -- no third-party dependency.

    Emits `sliderMoved(low, high)` CONTINUOUSLY while a handle is dragged (live
    visual feedback) and `rangeChanged(low, high)` ONCE on release (the commit
    point). The two handles cannot cross; values are clamped to the bounds.
    """

    sliderMoved = pyqtSignal(float, float)
    rangeChanged = pyqtSignal(float, float)

    def __init__(self, minimum=0.0, maximum=1.0, parent=None):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum)
        self._low = self._min
        self._high = self._max
        self._radius = 8                 # handle radius (px)
        self._active = None              # 'low' | 'high' | None
        self.setMinimumHeight(34)
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    # ---- public API ----
    def setBounds(self, minimum, maximum):
        self._min, self._max = float(minimum), float(maximum)
        if self._max <= self._min:
            self._max = self._min + 1.0
        self._low = min(max(self._low, self._min), self._max)
        self._high = min(max(self._high, self._low), self._max)
        self.update()

    def setValues(self, low, high):
        lo, hi = float(low), float(high)
        if hi < lo:
            lo, hi = hi, lo
        self._low = min(max(lo, self._min), self._max)
        self._high = min(max(hi, self._low), self._max)
        self.update()

    def values(self) -> tuple:
        return (self._low, self._high)

    # ---- geometry ----
    def _track(self) -> QRect:
        m = self._radius + 2
        return QRect(m, self.height() // 2 - 3,
                     max(1, self.width() - 2 * m), 6)

    def _val_to_x(self, v: float) -> float:
        tr = self._track()
        span = self._max - self._min
        frac = (v - self._min) / span if span else 0.0
        return tr.left() + frac * tr.width()

    def _x_to_val(self, x: float) -> float:
        tr = self._track()
        frac = (x - tr.left()) / tr.width() if tr.width() else 0.0
        frac = min(max(frac, 0.0), 1.0)
        return self._min + frac * (self._max - self._min)

    # ---- painting ----
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        tr = self._track()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#cfd6dd"))                 # groove
        p.drawRoundedRect(tr, 3, 3)
        xlo, xhi = self._val_to_x(self._low), self._val_to_x(self._high)
        p.setBrush(QColor("#ffd24d"))                 # selected span
        p.drawRoundedRect(QRectF(xlo, float(tr.top()),
                                 max(0.0, xhi - xlo), float(tr.height())), 3, 3)
        cy = self.height() / 2.0
        for x, on in ((xlo, self._active == "low"), (xhi, self._active == "high")):
            p.setBrush(QColor("#f0a500") if on else QColor("#ffffff"))
            p.setPen(QPen(QColor("#7a5c00"), 1.5))
            p.drawEllipse(QPointF(x, cy), self._radius, self._radius)
        p.end()

    # ---- interaction ----
    def _nearest_handle(self, x: float) -> str:
        xlo, xhi = self._val_to_x(self._low), self._val_to_x(self._high)
        if xlo == xhi:                                # overlapped -> pick by side
            return "low" if x < xlo else "high"
        return "low" if abs(x - xlo) <= abs(x - xhi) else "high"

    def _drag_to(self, x: float):
        v = self._x_to_val(x)
        if self._active == "low":
            self._low = min(v, self._high)
        elif self._active == "high":
            self._high = max(v, self._low)
        self.update()
        self.sliderMoved.emit(self._low, self._high)

    def mousePressEvent(self, ev):
        self._active = self._nearest_handle(ev.position().x())
        self._drag_to(ev.position().x())

    def mouseMoveEvent(self, ev):
        if self._active is not None:
            self._drag_to(ev.position().x())

    def mouseReleaseEvent(self, _ev):
        if self._active is None:
            return
        self._active = None
        self.update()
        self.rangeChanged.emit(self._low, self._high)


class CDReviewWindow(QDialog):
    """Non-modal two-column ladder/staircase review over the verified cloud
    records. Reads + classifies on the main thread; the only cloud writes are
    the additive classification fields and the per-record manual_window."""

    def __init__(self, log=print, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("CD-Shape Review  (ladder vs staircase)")
        self.resize(1360, 820)

        self._log = log
        self.records: list[dict] = []          # verified cloud records
        self.results: list = []                # ClassificationResult per record
        self.current_index: int = -1
        # The three column lists, so click handlers can clear each other.
        self._lists: list[QListWidget] = []

        self._build_ui()
        self.refresh(initial=True)

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)

        banner = QLabel(
            "CD-shape review — verified cloud records classified by "
            "classifier.py (PROVISIONAL thresholds). Auto labels are cached "
            "back to the cloud; your Ladder/Staircase/Unsure override is saved "
            "separately and never overwrites the auto label. Spectra are never "
            "modified.")
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
            "Re-fetch verified cloud records, re-classify, and sync any stale "
            "auto labels.")
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        top.addWidget(self.refresh_btn)
        self.resync_btn = QPushButton("Re-sync ALL auto labels")
        self.resync_btn.setToolTip(
            "Force-write the auto_classification cache for every record "
            "(use after changing the thresholds/windows in classifier.py).")
        self.resync_btn.clicked.connect(self._on_force_resync)
        top.addWidget(self.resync_btn)
        root.addLayout(top)

        # Main splitter: [ two columns + flagged ]  |  [ detail ]
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- left: columns ----
        left = QWidget()
        left_v = QVBoxLayout(left)
        cols = QHBoxLayout()

        self.ladder_box = QGroupBox("LADDER")
        lv = QVBoxLayout(self.ladder_box)
        self.ladder_list = QListWidget()
        lv.addWidget(self.ladder_list)
        cols.addWidget(self.ladder_box, stretch=1)

        self.staircase_box = QGroupBox("STAIRCASE")
        sv = QVBoxLayout(self.staircase_box)
        self.staircase_list = QListWidget()
        sv.addWidget(self.staircase_list)
        cols.addWidget(self.staircase_box, stretch=1)

        left_v.addLayout(cols, stretch=1)

        self.flagged_box = QGroupBox("FLAGGED / bad data")
        fv = QVBoxLayout(self.flagged_box)
        self.flagged_list = QListWidget()
        self.flagged_list.setMaximumHeight(120)
        fv.addWidget(self.flagged_list)
        left_v.addWidget(self.flagged_box)

        legend = QLabel(
            "Tints: ladder=green, staircase=blue, flagged=gray · "
            "⚠ peach = borderline (±%.2f of the UV/lobe ratio threshold) · "
            "pink = human override disagrees with auto."
            % classifier.BORDERLINE_BAND)
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#555;font-size:11px;")
        left_v.addWidget(legend)

        for lst in (self.ladder_list, self.staircase_list, self.flagged_list):
            lst.setSelectionMode(
                QAbstractItemView.SelectionMode.SingleSelection)
            lst.itemClicked.connect(self._on_item_clicked)
            self._lists.append(lst)

        splitter.addWidget(left)

        # ---- right: detail ----
        splitter.addWidget(self._build_detail())

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([560, 800])
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

        # Disagreement / override banner (hidden until a record with a human
        # override is shown).
        self.disagree_banner = QLabel("")
        self.disagree_banner.setWordWrap(True)
        self.disagree_banner.setVisible(False)
        v.addWidget(self.disagree_banner)

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

        # ---- CD analysis-window slider (manual per-record override) ----
        win_box = QGroupBox("CD analysis window  (drag handles; release to "
                            "apply + reclassify + save)")
        wv = QVBoxLayout(win_box)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("min"))
        self.win_min_lbl = QLabel("—")
        self.win_min_lbl.setMinimumWidth(58)
        self.win_min_lbl.setStyleSheet("font-weight:bold;")
        srow.addWidget(self.win_min_lbl)
        self.window_slider = RangeSlider(_PLOT_X_RANGE[0], _PLOT_X_RANGE[1])
        self.window_slider.sliderMoved.connect(self._on_window_dragged)
        self.window_slider.rangeChanged.connect(self._on_window_committed)
        srow.addWidget(self.window_slider, stretch=1)
        self.win_max_lbl = QLabel("—")
        self.win_max_lbl.setMinimumWidth(58)
        self.win_max_lbl.setStyleSheet("font-weight:bold;")
        srow.addWidget(QLabel("max"))
        srow.addWidget(self.win_max_lbl)
        self.reset_window_btn = QPushButton("Reset to auto")
        self.reset_window_btn.setToolTip(
            "Clear this record's manual window and revert to the data-driven "
            "window (re-saves the refreshed auto cache).")
        self.reset_window_btn.clicked.connect(self._on_reset_window)
        srow.addWidget(self.reset_window_btn)
        wv.addLayout(srow)
        self.window_status = QLabel("")
        self.window_status.setTextFormat(Qt.TextFormat.RichText)
        self.window_status.setWordWrap(True)
        self.window_status.setStyleSheet("color:#555;font-size:11px;")
        wv.addWidget(self.window_status)
        v.addWidget(win_box)

        # manual reclassify controls
        rc = QGroupBox("Manual reclassify (override — kept beside the auto label)")
        rc_h = QHBoxLayout(rc)
        rc_h.addWidget(QLabel("Human:"))
        self.human_group = QButtonGroup(self)
        self.human_group.setExclusive(True)
        self._human_btns = {}
        for lab, text in (("ladder", "Ladder"), ("staircase", "Staircase"),
                          ("unsure", "Unsure")):
            btn = QRadioButton(text)
            self.human_group.addButton(btn)
            self._human_btns[lab] = btn
            rc_h.addWidget(btn)
        self.save_override_btn = QPushButton("Save override")
        self.save_override_btn.clicked.connect(self._save_override)
        rc_h.addWidget(self.save_override_btn)
        self.clear_override_btn = QPushButton("Clear")
        self.clear_override_btn.setToolTip("Unselect (does not delete a "
                                           "previously saved override).")
        self.clear_override_btn.clicked.connect(lambda: self._set_human_radios(None))
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
        """Re-fetch verified cloud records, classify, sync stale auto labels,
        and rebuild the columns. Network runs on the main thread (quick +
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
        # Self-healing cache: write only the missing/stale auto labels.
        self._sync_auto(force=False)

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
        for lst in self._lists:
            lst.clear()
        n_ladder = n_stair = n_flag = 0
        for idx, (rec, res) in enumerate(zip(self.records, self.results)):
            item = QListWidgetItem(self._item_text(rec, res))
            item.setData(Qt.ItemDataRole.UserRole, idx)
            item.setBackground(QBrush(self._item_brush(rec, res)))
            item.setToolTip(rec.get("filename") or rec.get("record_id") or "")
            if classifier.is_ladder(res.label):
                self.ladder_list.addItem(item)
                n_ladder += 1
            elif res.label == classifier.LABEL_STAIRCASE:
                self.staircase_list.addItem(item)
                n_stair += 1
            else:
                self.flagged_list.addItem(item)
                n_flag += 1

        classified = n_ladder + n_stair
        total = len(self.records)
        pct = (100.0 * n_ladder / classified) if classified else 0.0
        pct_all = (100.0 * n_ladder / total) if total else 0.0
        self.ladder_box.setTitle(f"LADDER  ({n_ladder})")
        self.staircase_box.setTitle(f"STAIRCASE  ({n_stair})")
        self.flagged_box.setTitle(f"FLAGGED / bad data  ({n_flag})")
        self.pct_label.setText(
            f"Ladder %: {pct:.1f}% of {classified} classified  "
            f"·  {pct_all:.1f}% of all {total}  "
            f"·  borderline: {sum(1 for r in self.results if r.borderline)}  "
            f"·  overrides: {sum(1 for rec in self.records if _human_label(rec))}")

    def _item_text(self, rec: dict, res) -> str:
        rid = rec.get("record_id") or ""
        short = rid.split("-")[0] if rid else "(no id)"
        badges = []
        # Handedness badge so S vs R is visible inside the shared LADDER column.
        if classifier.is_ladder(res.label) and res.ladder_type:
            badges.append(f"{res.ladder_type}-ladder")
        if res.borderline:
            badges.append("⚠borderline")
        # Mark records running on a hand-set window vs. the default.
        if classifier._resolve_manual_window(rec.get("manual_window")) is not None:
            badges.append("✎win")
        h = _human_label(rec)
        if h:
            # Human override is a coarse family (ladder/staircase/unsure); compare
            # it against the auto label's family so 'ladder' agrees with either
            # handedness.
            if h == "unsure":
                badges.append("H:unsure?")
            elif h != classifier.label_family(res.label):
                badges.append(f"H:{h}✗")
            else:
                badges.append(f"H:{h}✓")
        tail = ("  [" + " ".join(badges) + "]") if badges else ""
        return f"{short} | {_human_readable(rec)}{tail}"

    def _item_brush(self, rec: dict, res) -> QColor:
        h = _human_label(rec)
        if h and h != "unsure" and h != classifier.label_family(res.label):
            return _DISAGREE_TINT
        if res.borderline:
            return _BORDERLINE_TINT
        return {"ladder": _LADDER_TINT,
                "staircase": _STAIRCASE_TINT}.get(
                    classifier.label_family(res.label), _FLAGGED_TINT)

    def _on_item_clicked(self, item: QListWidgetItem):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        # Single visual selection across the three lists.
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
        self.disagree_banner.setVisible(False)
        self.window_status.setText("")
        self.win_min_lbl.setText("—")
        self.win_max_lbl.setText("—")
        self._win_patches = []
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
        self.detail_header.setText(
            f"<b>Record {idx + 1} of {len(self.records)}</b> &nbsp;|&nbsp; "
            f"<code>{rid}</code>"
            + (f" &nbsp;|&nbsp; {fname}" if fname else ""))
        self.metrics_label.setText(self._metrics_html(rec, res))
        self._seed_window_slider(rec)
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

        # Label + handedness + borderline.
        hand = (f" &nbsp;<span style='color:#555;'>(handedness "
                f"<b>{res.ladder_type}</b>)</span>" if res.ladder_type else "")
        bl = ""
        if res.borderline:
            bl = (" &nbsp;<span style='background:#ffe5b4;padding:1px 5px;"
                  "border-radius:3px;'>BORDERLINE: "
                  f"{', '.join(res.borderline_flags)}</span>")

        win = (f"[{_wl(res.window_left)} … {_wl(res.window_right)}]"
               if res.window_left is not None else "—")
        if res.window_source == "manual":
            win_src = ("<span style='color:#7a5c00;font-weight:bold;'>"
                       "manual</span>")
        elif res.window_source == "data-driven":
            win_src = "auto (data-driven)"
        else:
            win_src = "—"

        uv_ratio_metric = (f"peak2/peak1 (baseline-subtracted) = "
                           f"<b>{_f(res.uv_two_peak_ratio, 2)}</b> "
                           f"(threshold ≥ {t_uv})")
        lobe_metric = (f"min/max lobe = <b>{_f(res.lobe_ratio, 2)}</b> "
                       f"(threshold ≥ {t_lobe})")

        reason = res.audit_reason or "—"

        return f"""
        <div style='font-size:13px;'>
        <h3 style='margin:2px 0;'>Auto label:
          <span style='text-transform:uppercase;'>{res.label}</span>{hand}{bl}</h3>
        <div style='color:#444;'>bisignate couplet:
          <b>{'yes' if res.bisignate else 'no'}</b></div>
        <hr>
        <b>UV bands ({psmin:.0f}–{psmax:.0f} nm search)</b>
          &nbsp;{chip(g.uv_band_detected, "2 BANDS", "1 BAND")}<br>
        &nbsp;&nbsp;baseline (tail): <b>{_f(res.uv_baseline)}</b><br>
        &nbsp;&nbsp;peak1 (shorter λ): <b>{_f(res.uv_peak1.value)}</b> @ {_wl(res.uv_peak1.wl)}<br>
        &nbsp;&nbsp;peak2 (longer λ): <b>{_f(res.uv_peak2.value)}</b> @ {_wl(res.uv_peak2.wl)}<br>
        &nbsp;&nbsp;inter-band valley: {_wl(res.interband_valley_lambda)}<br>
        &nbsp;&nbsp;CD window used ({win_src}): <b>{win}</b><br>
        <br>
        <b>Evolution gate — UV</b> &nbsp;{chip(g.uv_ratio_pass)}<br>
        &nbsp;&nbsp;{uv_ratio_metric}<br>
        <br>
        <b>CD couplet (inside window)</b>
          &nbsp;{chip(g.cd_couplet, "COUPLET", "NONE")}<br>
        &nbsp;&nbsp;positive lobe: <b>{_f(res.cd_pos_lobe.value)}</b> @ {_wl(res.cd_pos_lobe.wl)}<br>
        &nbsp;&nbsp;negative lobe: <b>{_f(res.cd_neg_lobe.value)}</b> @ {_wl(res.cd_neg_lobe.wl)}<br>
        &nbsp;&nbsp;crossover (interp): <b>{_wl(res.crossover_wavelength)}</b><br>
        <br>
        <b>Evolution gate — CD lobes</b> &nbsp;{chip(g.lobe_ratio_pass)}<br>
        &nbsp;&nbsp;{lobe_metric}<br>
        <hr>
        <b>audit reason:</b> {reason}
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

    def _draw_window_box(self, lo, hi):
        """Shade the analysis window on BOTH the CD and UV axes (the CD couplet
        is read inside it). Tracks the patches so a live slider drag can move
        the box cheaply without replotting the curves. Pass None/None to clear.
        The CD and UV axes share one x-range (sharex), so the box is vertically
        aligned across both plots."""
        for patch in getattr(self, "_win_patches", []):
            try:
                patch.remove()
            except (ValueError, NotImplementedError):
                pass
        self._win_patches = []
        if lo is not None and hi is not None and hi > lo:
            for ax in (self.ax_cd, self.ax_uv):
                self._win_patches.append(
                    ax.axvspan(lo, hi, color="#ffd24d", alpha=0.15, zorder=0))
        self.canvas.draw_idle()

    def _draw_detail_plots(self, idx: int):
        rec, res = self.records[idx], self.results[idx]
        self._win_patches = []        # cleared with the axes below
        for ax in self._axes.values():
            ax.clear()

        # raw embedded arrays, drawn faithfully (no smoothing)
        for sig, ax in self._axes.items():
            arr = self._signal_arrays(rec, sig)
            if arr is not None:
                wl, y = arr
                ax.plot(wl, y, color=_SIGNAL_COLORS[sig], linewidth=1.3)

        # --- analysis window: shade on BOTH CD and UV at the ACTUAL window the
        # classifier used (res.window). A live slider drag re-draws this box at
        # the candidate window via _draw_window_box; on release the record is
        # reclassified so res.window == the committed manual window. None for
        # flagged / single-band records (nothing to shade).
        self._draw_window_box(res.window_left, res.window_right)

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

        # --- UV: draw the baseline, mark peak1/peak2, and the inter-band
        # valley (the window's left edge). ---
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
        if res.interband_valley_lambda is not None:
            self.ax_uv.axvline(res.interband_valley_lambda, color="#fd7e14",
                               linestyle="--", linewidth=1.0, zorder=4)

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
    def _seed_window_slider(self, rec: dict):
        """Initialize the slider to the record's stored manual_window if present,
        else the global default (classifier.WINDOW_DEFAULT_*). blockSignals so
        seeding never triggers a spurious commit/reclassify."""
        self.window_slider.blockSignals(True)
        self.window_slider.setBounds(_PLOT_X_RANGE[0], _PLOT_X_RANGE[1])
        mw = classifier._resolve_manual_window(rec.get("manual_window"))
        if mw is not None:
            lo, hi = mw
        else:
            lo = classifier.WINDOW_DEFAULT_MIN
            hi = classifier.WINDOW_DEFAULT_MAX
        self.window_slider.setValues(lo, hi)
        self.window_slider.blockSignals(False)
        self.win_min_lbl.setText(f"{lo:.0f} nm")
        self.win_max_lbl.setText(f"{hi:.0f} nm")

    def _update_window_status(self, rec: dict, res):
        mw = classifier._resolve_manual_window(rec.get("manual_window"))
        if mw is not None:
            self.window_status.setText(
                f"<b>Manual window ACTIVE</b> — {mw[0]:.0f}–{mw[1]:.0f} nm "
                f"overrides the auto window. Drag to adjust, or "
                f"<i>Reset to auto</i>.")
        elif res.window_left is not None:
            self.window_status.setText(
                f"No manual window — classifier uses the <b>auto "
                f"(data-driven)</b> window "
                f"{res.window_left:.0f}–{res.window_right:.0f} nm. Slider shows "
                f"the default "
                f"{classifier.WINDOW_DEFAULT_MIN:.0f}–"
                f"{classifier.WINDOW_DEFAULT_MAX:.0f} nm; drag + release to set "
                f"a manual window for this record.")
        else:
            self.window_status.setText(
                "No manual window. The auto window is unavailable for this "
                "record (flagged / single UV band). Drag + release to force a "
                "manual CD window.")

    def _on_window_dragged(self, lo: float, hi: float):
        """Live during a handle drag: update the nm readouts and redraw the
        shaded box only. Reclassification is DEFERRED to release."""
        self.win_min_lbl.setText(f"{lo:.0f} nm")
        self.win_max_lbl.setText(f"{hi:.0f} nm")
        self._draw_window_box(lo, hi)

    def _on_window_committed(self, lo: float, hi: float):
        """On handle RELEASE: set this record's manual_window, reclassify under
        it, persist BOTH the window and the refreshed auto cache together, and
        refresh the metrics + markers + columns."""
        if not (0 <= self.current_index < len(self.records)):
            return
        idx = self.current_index
        rec = self.records[idx]
        mw = {"min_nm": round(float(lo), 1), "max_nm": round(float(hi), 1)}
        rec["manual_window"] = mw                       # mirror locally
        res = classifier.classify_record(rec)           # recompute under window
        self.results[idx] = res
        auto = classifier.auto_classification_doc(res)
        rec["auto_classification"] = auto               # mirror locally
        self.win_min_lbl.setText(f"{lo:.0f} nm")
        self.win_max_lbl.setText(f"{hi:.0f} nm")
        self.metrics_label.setText(self._metrics_html(rec, res))
        self._draw_detail_plots(idx)
        self._update_window_status(rec, res)
        self._populate_columns()
        self._select_index_in_columns(idx)
        self._persist_window(rec.get("record_id"), mw, auto)

    def _on_reset_window(self):
        """Clear this record's manual_window (revert to the data-driven window),
        reclassify, and persist the cleared field + refreshed cache together."""
        if not (0 <= self.current_index < len(self.records)):
            return
        idx = self.current_index
        rec = self.records[idx]
        rec.pop("manual_window", None)                  # mirror clear locally
        res = classifier.classify_record(rec)           # data-driven again
        self.results[idx] = res
        auto = classifier.auto_classification_doc(res)
        rec["auto_classification"] = auto
        self._seed_window_slider(rec)                   # back to default seed
        self.metrics_label.setText(self._metrics_html(rec, res))
        self._draw_detail_plots(idx)
        self._update_window_status(rec, res)
        self._populate_columns()
        self._select_index_in_columns(idx)
        self._persist_window(rec.get("record_id"), None, auto)

    def _persist_window(self, rid, manual_window, auto):
        """Atomically write manual_window (or clear it) + the refreshed auto
        cache via mongo_db.set_manual_window. Guarded + non-blocking-ish like the
        override save; a failure is surfaced in the window status line."""
        if not rid:
            self._log("Manual window not persisted: record has no record_id.")
            self.window_status.setText(
                "⚠ Not saved to cloud: this record has no record_id.")
            return
        self.window_slider.setEnabled(False)
        self.reset_window_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from mongo_db import set_manual_window
            res = set_manual_window(rid, manual_window, auto, log=self._log)
        except Exception as e:
            self._log(f"Manual window save failed: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            res = {"ok": False, "message": str(e)}
        finally:
            self.window_slider.setEnabled(True)
            self.reset_window_btn.setEnabled(True)
        if not res.get("ok"):
            self.window_status.setText(
                "⚠ Cloud save failed: " + res.get("message", "unknown error"))

    # ------------------------------------------------- manual reclassify ----
    def _set_detail_enabled(self, enabled: bool):
        for btn in self._human_btns.values():
            btn.setEnabled(enabled)
        self.save_override_btn.setEnabled(enabled)
        self.clear_override_btn.setEnabled(enabled)
        self.window_slider.setEnabled(enabled)
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
                                         "cannot save override.")
            self._log("Override not saved: record has no record_id.")
            return

        human_doc = {"label": label,
                     "reviewed_at": datetime.now(timezone.utc).isoformat()}
        self.save_override_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            from mongo_db import set_human_classification
            res = set_human_classification(rid, human_doc, log=self._log)
        except Exception as e:
            self._log(f"Override save failed: {type(e).__name__}: {e}")
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

    def _refresh_override_banner(self, rec: dict, res):
        h = _human_label(rec)
        if not h:
            self.disagree_banner.setVisible(False)
            self.override_status.setText("No human override yet.")
            return
        hd = rec.get("human_classification") or {}
        when = (hd.get("reviewed_at") or "")[:19]
        self.override_status.setText(
            f"Saved: {h}" + (f" · {when}Z" if when else ""))
        if h == "unsure":
            self._banner("#fff3cd", "#856404",
                         "Human marked <b>UNSURE</b> — flagged for a second look.")
        elif h != classifier.label_family(res.label):
            self._banner(
                "#f8d7da", "#721c24",
                f"Human override <b>{h.upper()}</b> DISAGREES with auto "
                f"<b>{res.label.upper()}</b> — this is a tuning signal.")
        else:
            self._banner("#d4edda", "#155724",
                         f"Human override <b>{h.upper()}</b> agrees with auto.")

    def _banner(self, bg, fg, html):
        self.disagree_banner.setText(html)
        self.disagree_banner.setStyleSheet(
            f"background:{bg};color:{fg};padding:5px;border-radius:4px;")
        self.disagree_banner.setVisible(True)

    # ----------------------------------------------------- cloud caching ----
    def _sync_auto(self, force: bool):
        """Write the auto_classification cache for stale/missing records (or
        all, when force=True). Idempotent; updates the in-memory mirror so
        repeated syncs become no-ops until the classifier output changes."""
        pairs = []
        for rec, res in zip(self.records, self.results):
            rid = rec.get("record_id")
            auto = classifier.auto_classification_doc(res)
            stored = rec.get("auto_classification")
            if force or not _auto_cache_current(stored, auto):
                rec["auto_classification"] = auto      # mirror locally
                if rid:
                    pairs.append((rid, auto))
        if not pairs:
            self._log("Auto-classification cache already current "
                      "(nothing to sync).")
            return
        try:
            from mongo_db import sync_auto_classifications
            sync_auto_classifications(pairs, log=self._log)
        except Exception as e:
            self._log(f"Auto-classification sync error: "
                      f"{type(e).__name__}: {e}")
            self._log(traceback.format_exc())

    def _on_force_resync(self):
        if not self.records:
            self._log("No records to re-sync.")
            return
        self.resync_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            self._log(f"Force re-syncing auto labels for {len(self.records)} "
                      f"record(s)…")
            self._sync_auto(force=True)
        finally:
            self.resync_btn.setEnabled(True)
