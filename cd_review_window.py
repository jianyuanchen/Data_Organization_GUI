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
    after the PROVISIONAL thresholds in classifier.py change).
  - human_classification: durable reviewer override (Ladder/Staircase/Unsure +
    timestamp), written on demand. Never overwrites the auto label -- both are
    kept side by side; disagreements are surfaced as the tuning signal.

Thresholds/windows are read from classifier.py's named constants -- not
duplicated here.
"""
from __future__ import annotations

import traceback
from datetime import datetime, timezone

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
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

    A change in the PROVISIONAL thresholds snapshot, the hard label, the
    borderline flag, or either ratio (to 6 dp) marks the cache STALE so it is
    rewritten. Everything else is derived from these, so this is sufficient.
    """
    if not isinstance(stored, dict):
        return False
    if stored.get("thresholds") != fresh.get("thresholds"):
        return False
    if stored.get("label") != fresh.get("label"):
        return False
    if bool(stored.get("borderline")) != bool(fresh.get("borderline")):
        return False
    for k in ("couplet_ratio", "uv_ratio"):
        if _round6(stored.get(k)) != _round6(fresh.get(k)):
            return False
    return True


class CDReviewWindow(QDialog):
    """Non-modal two-column ladder/staircase review over the verified cloud
    records. Reads + classifies on the main thread; the only cloud writes are
    the two additive classification fields."""

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
            "⚠ peach = borderline (±%.2f of a threshold) · "
            "pink = human override disagrees with auto."
            % classifier.BORDERLINE_DELTA)
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
            if res.label == "ladder":
                self.ladder_list.addItem(item)
                n_ladder += 1
            elif res.label == "staircase":
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
        if res.borderline:
            badges.append("⚠borderline")
        h = _human_label(rec)
        if h:
            if h == "unsure":
                badges.append("H:unsure?")
            elif h != res.label:
                badges.append(f"H:{h}✗")
            else:
                badges.append(f"H:{h}✓")
        tail = ("  [" + " ".join(badges) + "]") if badges else ""
        return f"{short} | {_human_readable(rec)}{tail}"

    def _item_brush(self, rec: dict, res) -> QColor:
        h = _human_label(rec)
        if h and h != "unsure" and h != res.label:
            return _DISAGREE_TINT
        if res.borderline:
            return _BORDERLINE_TINT
        return {"ladder": _LADDER_TINT,
                "staircase": _STAIRCASE_TINT}.get(res.label, _FLAGGED_TINT)

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
        self._draw_detail_plots(idx)
        self._set_detail_enabled(True)
        self._set_human_radios(_human_label(rec))
        self._refresh_override_banner(rec, res)
        self._select_index_in_columns(idx)

    def _metrics_html(self, rec: dict, res) -> str:
        c1, c2 = res.criterion1, res.criterion2
        cdlo, cdhi = classifier.CD_WINDOW_NM
        uvlo, uvhi = classifier.UV_WINDOW_NM
        t1 = classifier.CRIT1_RATIO_THRESHOLD
        t2 = classifier.CRIT2_RATIO_THRESHOLD

        def chip(ok):
            color = "#28a745" if ok else "#c82333"
            return (f"<span style='color:white;background:{color};"
                    f"padding:1px 6px;border-radius:3px;'>"
                    f"{'PASS' if ok else 'FAIL'}</span>")

        bl = ""
        if res.borderline:
            bl = (" &nbsp;<span style='background:#ffe5b4;padding:1px 5px;"
                  "border-radius:3px;'>BORDERLINE: "
                  f"{', '.join(res.borderline_criteria)}</span>")

        if c2.single_peak:
            c2_metric = "single UV peak (no peak2) → criterion fails"
        else:
            c2_metric = (f"peak2/peak1 = <b>{_f(c2.ratio, 2)}</b> "
                         f"(threshold ≥ {t2})")

        reasons = "; ".join(res.reasons) if res.reasons else "—"

        return f"""
        <div style='font-size:13px;'>
        <h3 style='margin:2px 0;'>Auto label:
          <span style='text-transform:uppercase;'>{res.label}</span>{bl}</h3>
        <hr>
        <b>Criterion 1 — CD couplet ({cdlo:.0f}–{cdhi:.0f} nm)</b> &nbsp;{chip(c1.passed)}<br>
        &nbsp;&nbsp;positive peak: <b>{_f(c1.pos_peak_value)}</b> @ {_wl(c1.pos_peak_wl)}<br>
        &nbsp;&nbsp;negative peak: <b>{_f(c1.neg_peak_value)}</b> @ {_wl(c1.neg_peak_wl)}<br>
        &nbsp;&nbsp;opposite signs (bisignate couplet): <b>{'yes' if c1.opposite_signs else 'no'}</b><br>
        &nbsp;&nbsp;|pos|/|neg| = <b>{_f(c1.couplet_ratio, 2)}</b> (threshold ≥ {t1})<br>
        <br>
        <b>Criterion 2 — UV two-peak ({uvlo:.0f}–{uvhi:.0f} nm)</b> &nbsp;{chip(c2.passed)}<br>
        &nbsp;&nbsp;peak1 (shorter λ): <b>{_f(c2.peak1_value)}</b> @ {_wl(c2.peak1_wl)}<br>
        &nbsp;&nbsp;peak2 (longer λ): <b>{_f(c2.peak2_value)}</b> @ {_wl(c2.peak2_wl)}<br>
        &nbsp;&nbsp;{c2_metric}<br>
        <br>
        <b>reason notes:</b> {reasons}
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

    def _draw_detail_plots(self, idx: int):
        rec, res = self.records[idx], self.results[idx]
        for ax in self._axes.values():
            ax.clear()

        # raw embedded arrays, drawn faithfully (no smoothing)
        for sig, ax in self._axes.items():
            arr = self._signal_arrays(rec, sig)
            if arr is not None:
                wl, y = arr
                ax.plot(wl, y, color=_SIGNAL_COLORS[sig], linewidth=1.3)

        # --- CD: shade couplet window + mark selected pos/neg peaks ---
        c1 = res.criterion1
        cdlo, cdhi = classifier.CD_WINDOW_NM
        self.ax_cd.axvspan(cdlo, cdhi, color="#ffd24d", alpha=0.13, zorder=0)
        if c1.pos_peak_wl is not None and c1.pos_peak_value is not None:
            self._mark(self.ax_cd, c1.pos_peak_wl, c1.pos_peak_value, "^",
                       "#d62728", f"pos {c1.pos_peak_wl:.0f}nm")
        if c1.neg_peak_wl is not None and c1.neg_peak_value is not None:
            self._mark(self.ax_cd, c1.neg_peak_wl, c1.neg_peak_value, "v",
                       "#1f77b4", f"neg {c1.neg_peak_wl:.0f}nm", below=True)

        # --- UV: shade window + mark peak1/peak2 (or single-peak note) ---
        c2 = res.criterion2
        uvlo, uvhi = classifier.UV_WINDOW_NM
        self.ax_uv.axvspan(uvlo, uvhi, color="#ffd24d", alpha=0.13, zorder=0)
        if c2.peak1_wl is not None and c2.peak1_value is not None:
            lbl = ("single UV peak" if c2.single_peak
                   else f"peak1 {c2.peak1_wl:.0f}nm")
            self._mark(self.ax_uv, c2.peak1_wl, c2.peak1_value, "o",
                       "#d62728", lbl)
        if c2.peak2_wl is not None and c2.peak2_value is not None:
            self._mark(self.ax_uv, c2.peak2_wl, c2.peak2_value, "s",
                       "#9467bd", f"peak2 {c2.peak2_wl:.0f}nm")

        self._draw_axes_chrome()
        self.toolbar.update()      # reset zoom/pan history for the new record
        self.canvas.draw_idle()

    def _mark(self, ax, x, y, marker, color, label, below=False):
        ax.plot([x], [y], marker, color=color, markersize=8, zorder=6)
        ax.annotate(
            label, (x, y), textcoords="offset points",
            xytext=(4, -12 if below else 6), fontsize=8, color=color,
            zorder=7)

    # ------------------------------------------------- manual reclassify ----
    def _set_detail_enabled(self, enabled: bool):
        for btn in self._human_btns.values():
            btn.setEnabled(enabled)
        self.save_override_btn.setEnabled(enabled)
        self.clear_override_btn.setEnabled(enabled)

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
        elif h != res.label:
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
