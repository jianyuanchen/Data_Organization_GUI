"""
Verification / import review window. Non-modal QDialog for reviewing an
arbitrary set of records one at a time. The caller chooses the set: a
whole batch (Review Imported Batch) or a hand-picked subset of staging-
table rows (Review Selected). The reviewer walks through records, edits
parsed fields, optionally focuses on parser-flagged-uncertain fields,
and marks each one confirmed / rejected / needs-work in the local
SQLite store.

Phase 2a: shell + workflow + local SQLite writes. No plotting yet -- a
labeled placeholder pane on the right reserves space for the per-record
inline plot that Phase 2b will fill in.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCloseEvent, QDoubleValidator
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QCheckBox, QMessageBox, QSizePolicy,
)

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from models import REVIEW_STATUS_COLORS, VISIBLE_COLUMNS
from plotting import read_csv_columns


# Index of each signal's column in the CSV layout produced by
# plotting.read_csv_columns. Matches the existing Quantity.src_col
# constants in plotting.py: 1=g-value, 2=CD, 3=UV-Vis (0 is wavelength).
_SIGNAL_CSV_COL = {"g": 1, "CD": 2, "UV": 3}
_SIGNAL_TITLES = {"CD": "CD (mdeg)", "g": "g-value", "UV": "UV-Vis (abs)"}
_SIGNAL_COLORS = {"CD": "tab:blue", "g": "tab:green", "UV": "tab:red"}
_SIGNAL_ORDER = ("CD", "g", "UV")          # top-to-bottom subplot order
# Field mapping for Apply: signal -> (value column, wavelength column).
# The g-value plot writes back to the existing peak_g / peak_wl pair;
# CD and UV write to the new peak_cd / peak_cd_wl and peak_uv / peak_uv_wl.
_SIGNAL_DB_FIELDS = {
    "CD": ("peak_cd", "peak_cd_wl"),
    "g":  ("peak_g",  "peak_wl"),
    "UV": ("peak_uv", "peak_uv_wl"),
}
_PLOT_X_RANGE = (300, 700)                  # full wavelength axis, always visible


# Sidebar row tint per review_status. Built from the shared hex codes in
# models so the main staging table and this view stay in sync; the sidebar
# substitutes a very light gray for the "no tint" pending entry so an
# untouched row still reads as a distinct list item rather than blending
# into the QListWidget background.
_SIDEBAR_PENDING_TINT = QColor("#f5f5f5")
_STATUS_COLORS = {
    status: (QColor(hex_code) if hex_code else _SIDEBAR_PENDING_TINT)
    for status, hex_code in REVIEW_STATUS_COLORS.items()
}

# Style applied to a field input whose name is listed in the row's `flags`.
# Same yellow as the staging table's pending-edit tint so the visual
# vocabulary is consistent.
_FLAGGED_STYLE = "background:#fff3cd;font-weight:bold;"

# Every visible column except csv_path is editable in the review pane. This
# matches the staging table's affordances; csv_path is always locked because
# it's the DB primary key. config / n_components stay editable on purpose --
# the staging table allows it too, and the reviewer occasionally needs to
# correct a misderived config.
_EDITABLE_FIELDS = [c for c in VISIBLE_COLUMNS if c != "csv_path"]


def _default_added_by() -> str:
    """Best-effort current-user string for the `added_by` column."""
    return os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"


def _flagged_fields(record: dict) -> set:
    """Parse a row's `flags` column as a JSON list of field names. Empty
    string / invalid JSON / non-list payload all yield the empty set.
    """
    f = record.get("flags") or ""
    if not f:
        return set()
    try:
        data = json.loads(f)
        if isinstance(data, list):
            return {str(x) for x in data}
    except Exception:
        pass
    return set()


class VerificationWindow(QDialog):
    """Non-modal review window for one batch of imported records."""

    # Emitted whenever review work is finished and the main window should
    # reload from the DB. Fired on Save & Close, on X-button close, and on
    # QDialog.finished too (the existing finished signal is also kept as a
    # belt-and-braces hookup).
    reviewFinished = pyqtSignal()

    def __init__(self, db, records, title_suffix: str = "", parent=None):
        super().__init__(parent)
        self.db = db
        self.setModal(False)
        title = "Review Records"
        if title_suffix:
            title += f"  —  {title_suffix}"
        self.setWindowTitle(title)
        self.resize(1280, 760)

        # Caller hands us the records to review (a full batch, a selected
        # subset, anything). We work off this list in place; navigation +
        # status counts stay in sync with what we hold here, and the DB
        # is the durable record.
        self.records: list[dict] = list(records)
        self.current_index: int = 0 if self.records else -1
        # Pending field edits for the currently-displayed record only.
        # Persisted to the DB by Confirm; discarded on Reject / Needs Work /
        # navigation -- explicit, simple, no per-record buffer.
        self.pending_edits: dict[str, str] = {}

        self._build_ui()
        if self.records:
            self._load_record(0)
        else:
            self._show_empty_state()

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)

        # Top strip: progress + sort toggle.
        top = QHBoxLayout()
        self.progress_label = QLabel()
        self.progress_label.setStyleSheet("font-weight:bold;")
        top.addWidget(self.progress_label, stretch=1)
        self.flagged_first = QCheckBox("Flagged-uncertain rows on top")
        self.flagged_first.setToolTip(
            "Sort the sidebar so records with non-empty `flags` appear first.")
        self.flagged_first.toggled.connect(self._rebuild_sidebar)
        top.addWidget(self.flagged_first)
        root.addLayout(top)

        # Splitter: sidebar | center pane | plot placeholder.
        splitter = QSplitter()

        # ---- Sidebar ----
        sidebar_box = QGroupBox("Batch records")
        sidebar_v = QVBoxLayout(sidebar_box)
        self.sidebar = QListWidget()
        self.sidebar.itemClicked.connect(self._on_sidebar_click)
        sidebar_v.addWidget(self.sidebar)
        splitter.addWidget(sidebar_box)

        # ---- Center pane ----
        center_box = QGroupBox("Record")
        center_v = QVBoxLayout(center_box)
        self.record_header = QLabel()
        self.record_header.setTextFormat(Qt.TextFormat.RichText)
        center_v.addWidget(self.record_header)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color:#666;font-size:11px;")
        center_v.addWidget(self.path_label)

        # Parse-error banner: shown only for unparsed rows. The filename
        # didn't decode, so the form below is empty and read-only; the
        # banner tells the reviewer why and what to do about it.
        self.error_banner = QLabel()
        self.error_banner.setWordWrap(True)
        self.error_banner.setStyleSheet(
            "background:#f5c6cb;color:#721c24;"
            "padding:8px;border-radius:4px;font-weight:bold;")
        self.error_banner.hide()
        center_v.addWidget(self.error_banner)

        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.field_inputs: dict[str, QLineEdit] = {}
        for col in _EDITABLE_FIELDS:
            le = QLineEdit()
            # Capture col at lambda-build time -- otherwise every callback
            # would close over the loop variable's final value.
            le.editingFinished.connect(lambda c=col: self._on_field_edit(c))
            self.field_inputs[col] = le
            form.addRow(QLabel(col + ":"), le)
        center_v.addWidget(form_host)
        center_v.addStretch(1)
        splitter.addWidget(center_box)

        # ---- Plot pane: 3 sharex'd subplots + range band + peak finder ----
        plot_box = QGroupBox("Per-record plots")
        plot_v = QVBoxLayout(plot_box)

        # Range inputs: typed numeric boxes drive an axvspan band on all
        # three subplots. Drag-select on the canvas is a future improvement.
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Range min (nm):"))
        self.range_min_input = QLineEdit()
        self.range_min_input.setValidator(QDoubleValidator())
        self.range_min_input.setMaximumWidth(80)
        self.range_min_input.setPlaceholderText("e.g. 400")
        self.range_min_input.textChanged.connect(self._on_range_changed)
        range_row.addWidget(self.range_min_input)
        range_row.addSpacing(10)
        range_row.addWidget(QLabel("Range max (nm):"))
        self.range_max_input = QLineEdit()
        self.range_max_input.setValidator(QDoubleValidator())
        self.range_max_input.setMaximumWidth(80)
        self.range_max_input.setPlaceholderText("e.g. 550")
        self.range_max_input.textChanged.connect(self._on_range_changed)
        range_row.addWidget(self.range_max_input)
        range_row.addStretch(1)
        plot_v.addLayout(range_row)

        # Figure + canvas. 3 stacked subplots, sharex aligns the x-axes
        # vertically so a peak at 500nm in CD sits directly above 500nm
        # in g and UV.
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax_cd, self.ax_g, self.ax_uv = self.figure.subplots(
            3, 1, sharex=True)
        self._sig_axes = {"CD": self.ax_cd, "g": self.ax_g, "UV": self.ax_uv}
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        plot_v.addWidget(self.canvas, stretch=1)

        # Active-plot selector + Find Max/Min + Apply.
        ctl_row = QHBoxLayout()
        ctl_row.addWidget(QLabel("Active:"))
        self.active_combo = QComboBox()
        self.active_combo.addItems(["CD", "g", "UV"])
        self.active_combo.setToolTip(
            "Which subplot the Find Max/Min and Apply buttons operate on. "
            "Click directly on a subplot to switch.")
        self.active_combo.currentTextChanged.connect(
            self._set_active_signal)
        ctl_row.addWidget(self.active_combo)
        ctl_row.addSpacing(10)
        self.find_max_btn = QPushButton("Find Max")
        self.find_max_btn.clicked.connect(lambda: self._find_extreme("max"))
        ctl_row.addWidget(self.find_max_btn)
        self.find_min_btn = QPushButton("Find Min")
        self.find_min_btn.clicked.connect(lambda: self._find_extreme("min"))
        ctl_row.addWidget(self.find_min_btn)
        ctl_row.addSpacing(10)
        self.apply_peak_btn = QPushButton("Apply to record...")
        self.apply_peak_btn.setStyleSheet("font-weight:bold;")
        self.apply_peak_btn.setToolTip(
            "Replace the active signal's stored peak value with the most "
            "recently computed peak. Prompts for confirmation before "
            "overwriting; sets edited=1 on the row.")
        self.apply_peak_btn.clicked.connect(self.on_apply_peak)
        ctl_row.addWidget(self.apply_peak_btn)
        ctl_row.addStretch(1)
        plot_v.addLayout(ctl_row)

        splitter.addWidget(plot_box)

        # ---- Plot-state bookkeeping ----
        # Which subplot the Find/Apply controls operate on.
        self._active_signal = "g"
        # Per-signal latest computed peak: signal -> (wl, value, kind).
        # Cleared when navigating to a new record so peaks from a prior
        # record don't bleed in.
        self._computed_peaks: dict = {"CD": None, "g": None, "UV": None}
        # Matplotlib artist handles we need to remove individually on
        # redraws (axvspan band + per-signal marker / annotation / corner
        # readout). Kept separate from the curves themselves (which go
        # away via ax.clear()).
        self._band_artists: list = []
        self._marker_artists: dict = {}
        self._readout_artists: dict = {}
        # Set False when the current record has no plottable data (unparsed
        # row, CSV read failed, etc.). Find Max/Min/Apply refuse in that
        # state and the buttons are disabled.
        self._has_plot_data = False

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 3)
        root.addWidget(splitter, stretch=1)

        # Bottom: navigation + action buttons.
        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.on_prev)
        bottom.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.on_next)
        bottom.addWidget(self.next_btn)
        bottom.addStretch(1)

        # Order: Needs Work / Reject / Confirm so the green Confirm is the
        # right-most, most prominent affordance.
        self.needs_work_btn = QPushButton("Needs Work")
        self.needs_work_btn.setStyleSheet(
            "background:#fff3cd;padding:6px 14px;")
        self.needs_work_btn.clicked.connect(
            lambda: self._apply_action("needs_work"))
        bottom.addWidget(self.needs_work_btn)

        self.reject_btn = QPushButton("Reject")
        self.reject_btn.setStyleSheet(
            "background:#f8d7da;padding:6px 14px;")
        self.reject_btn.clicked.connect(lambda: self._apply_action("rejected"))
        bottom.addWidget(self.reject_btn)

        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.setStyleSheet(
            "background:#d4edda;font-weight:bold;padding:6px 14px;")
        self.confirm_btn.clicked.connect(
            lambda: self._apply_action("confirmed"))
        bottom.addWidget(self.confirm_btn)

        # Visual separator before the close button so it doesn't read as a
        # fourth review-action.
        bottom.addSpacing(20)
        self.close_btn = QPushButton("Save && Close")
        self.close_btn.setToolTip(
            "Close the review window. Statuses and edits are already saved "
            "to the local DB; closing refreshes the main staging table.")
        self.close_btn.setStyleSheet(
            "padding:6px 14px;font-weight:bold;")
        self.close_btn.clicked.connect(self.on_save_close)
        bottom.addWidget(self.close_btn)

        root.addLayout(bottom)

        self._rebuild_sidebar()
        self._update_progress()

    def _show_empty_state(self):
        self.record_header.setText("<b>No records in this batch.</b>")
        self.path_label.setText("")
        for w in (self.prev_btn, self.next_btn, self.confirm_btn,
                  self.reject_btn, self.needs_work_btn):
            w.setEnabled(False)
        for le in self.field_inputs.values():
            le.setEnabled(False)
        # Paint blank plots so the empty pane still has axes + labels
        # rather than three big empty rectangles.
        self._show_no_data("(no records)")

    # --------------------------------------------------------- close path --
    def _resolve_pending_or_cancel(self) -> bool:
        """If field edits are staged on the current record, ask the user
        what to do. Returns True if the caller may proceed to close, False
        if the user wants to stay in the window.
        """
        if not self.pending_edits or self.current_index < 0:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved field edits")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            f"You have {len(self.pending_edits)} unsaved field edit(s) on "
            f"record {self.current_index + 1}.")
        box.setInformativeText(
            "Confirm this record (saving the edits with verified=1), "
            "discard the edits and close, or cancel close?")
        confirm = box.addButton(
            "Confirm record", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton(
            "Discard edits", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton(
            "Cancel close", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cancel:
            return False
        if clicked is confirm:
            # Reuses the Confirm action: persists edits + flips review_status
            # to 'confirmed' + sets verified/verified_date/added_by, then
            # advances. Safe to proceed to close afterwards.
            self._apply_action("confirmed")
        else:
            # Discard: drop the in-memory edits without touching the DB.
            self.pending_edits.clear()
        return True

    def on_save_close(self):
        """Save & Close button: resolve any pending edits, then close."""
        if not self._resolve_pending_or_cancel():
            return
        # accept() emits finished(Accepted) which the main window listens to;
        # we additionally emit reviewFinished for any future subscribers.
        self.reviewFinished.emit()
        self.accept()

    def closeEvent(self, event: QCloseEvent):
        """Window X / Alt-F4 / any other close goes through here. Re-uses
        the same prompt so the user can't silently drop pending edits.
        """
        if not self._resolve_pending_or_cancel():
            event.ignore()
            return
        self.reviewFinished.emit()
        super().closeEvent(event)

    # ----------------------------------------------------------- sidebar ----
    def _rebuild_sidebar(self):
        """Repopulate the sidebar from self.records, optionally re-sorting
        flagged rows to the top. Preserves the current record's selection.
        """
        # Preserve which record was active so a sort doesn't yank the user
        # away. Match by csv_path (the durable per-row key in this view).
        prev_path = None
        if self.records and 0 <= self.current_index < len(self.records):
            prev_path = self.records[self.current_index]["csv_path"]

        if self.flagged_first.isChecked():
            self.records.sort(key=lambda r: 0 if _flagged_fields(r) else 1)
        else:
            self.records.sort(key=lambda r: r["csv_path"])

        if prev_path:
            for i, r in enumerate(self.records):
                if r["csv_path"] == prev_path:
                    self.current_index = i
                    break

        self.sidebar.clear()
        for i, r in enumerate(self.records):
            status = r.get("review_status") or "pending"
            flagged_mark = "⚠ " if _flagged_fields(r) else ""
            name = Path(r["csv_path"]).name
            item = QListWidgetItem(f"{flagged_mark}{i + 1:>3}. {name}")
            item.setBackground(QBrush(
                _STATUS_COLORS.get(status, _STATUS_COLORS["pending"])))
            self.sidebar.addItem(item)
        if 0 <= self.current_index < self.sidebar.count():
            self.sidebar.setCurrentRow(self.current_index)

    def _refresh_sidebar_row(self, index: int):
        """Update one sidebar row's background after a status change."""
        item = self.sidebar.item(index)
        if item is None:
            return
        status = self.records[index].get("review_status") or "pending"
        item.setBackground(QBrush(
            _STATUS_COLORS.get(status, _STATUS_COLORS["pending"])))

    def _on_sidebar_click(self, item: QListWidgetItem):
        idx = self.sidebar.row(item)
        self._load_record(idx)

    # -------------------------------------------------------- record load --
    def _load_record(self, index: int):
        if not (0 <= index < len(self.records)):
            return
        # Switching records discards any pending edits on the previous one.
        # Phase 2a contract: edits live only as long as the current record is
        # active; Confirm persists them, anything else throws them away.
        self.pending_edits.clear()
        self.current_index = index
        r = self.records[index]
        status = r.get("review_status") or "pending"
        is_unparsed = status == "unparsed"
        self.record_header.setText(
            f"<b>Record {index + 1} of {len(self.records)}</b>  "
            f"&nbsp;|&nbsp; status: <code>{status}</code>")
        self.path_label.setText(r["csv_path"])

        # Unparsed rows: surface the parse error, disable the form + the
        # Confirm button. Reject and Needs Work stay enabled so the user
        # can still triage. The fix is "rename the file outside the app
        # and re-browse" -- not editing fields in this window.
        if is_unparsed:
            err = r.get("parse_error") or "(no error message stored)"
            self.error_banner.setText(
                f"Parse error: {err}\n"
                f"Rename the file to match the convention and re-browse "
                f"to create a parsed row.")
            self.error_banner.show()
        else:
            self.error_banner.hide()

        flagged = _flagged_fields(r)
        for col, le in self.field_inputs.items():
            val = r.get(col)
            le.blockSignals(True)
            le.setText("" if val is None else str(val))
            le.blockSignals(False)
            le.setEnabled(not is_unparsed)
            le.setStyleSheet(_FLAGGED_STYLE if col in flagged else "")

        self.confirm_btn.setEnabled(not is_unparsed)
        self.confirm_btn.setToolTip(
            "Cannot confirm an unparsed record -- rename the file to match "
            "the filename convention and re-browse to produce a parsed row."
            if is_unparsed
            else "Mark this record verified and save any field edits.")

        self.sidebar.setCurrentRow(index)
        self._update_progress()
        self._refresh_plots()

    def _on_field_edit(self, col: str):
        le = self.field_inputs.get(col)
        if le is None:
            return
        self.pending_edits[col] = le.text()

    # ------------------------------------------------------- navigation ----
    def on_prev(self):
        if self.current_index > 0:
            self._load_record(self.current_index - 1)

    def on_next(self):
        if self.current_index < len(self.records) - 1:
            self._load_record(self.current_index + 1)

    # ----------------------------------------------------------- actions ----
    def _apply_action(self, status: str):
        if not (0 <= self.current_index < len(self.records)):
            return
        r = self.records[self.current_index]
        csv_path = r["csv_path"]

        # Defence-in-depth: the Confirm button is already disabled for
        # unparsed rows, but if anything ever invokes _apply_action(
        # "confirmed") on one (e.g. via the close-prompt's "Confirm
        # record" branch), refuse loudly rather than silently flipping
        # the row to verified=1.
        if status == "confirmed" and r.get("review_status") == "unparsed":
            QMessageBox.warning(
                self, "Cannot confirm unparsed record",
                "This file's name did not match the filename convention "
                "(see the parse error above). Rename it to a valid name "
                "and re-browse the folder to create a parsed row; then "
                "verify the parsed row instead.")
            return

        if status == "confirmed":
            now = datetime.now().isoformat(timespec="seconds")
            user = _default_added_by()
            edits = dict(self.pending_edits) if self.pending_edits else None
            self.db.set_review(
                csv_path, status="confirmed",
                verified=1, verified_date=now, added_by=user,
                field_edits=edits)
            r["review_status"] = "confirmed"
            r["verified"] = 1
            r["verified_date"] = now
            r["added_by"] = user
            if edits:
                for c, v in edits.items():
                    r[c] = v
        else:
            # Reject / Needs Work: status only. Field edits are discarded;
            # the row stays in SQLite either way (never deleted by these
            # actions -- "Reject" means "this scan won't be promoted to the
            # shared store later", not "remove it from my local history").
            self.db.set_review(csv_path, status=status)
            r["review_status"] = status

        self.pending_edits.clear()
        self._refresh_sidebar_row(self.current_index)
        self._update_progress()

        # Advance to the next record if there is one; otherwise stay put
        # and refresh the header so the status badge reflects the update.
        if self.current_index < len(self.records) - 1:
            self._load_record(self.current_index + 1)
        else:
            self._load_record(self.current_index)

    # ------------------------------------------------- plot pane ----------
    def _refresh_plots(self):
        """Reload the current record's CSV into the three subplots.

        Resets all per-record plot state (curves, markers, corner read-
        outs, computed peaks). Unparsed rows / CSV read failures yield a
        labelled empty plot and disable the peak-finder controls.
        """
        # Drop per-record peak findings and artists; this is a new record.
        self._computed_peaks = {"CD": None, "g": None, "UV": None}
        self._marker_artists = {}
        self._readout_artists = {}
        self._band_artists = []

        for ax in self._sig_axes.values():
            ax.clear()

        r = self.records[self.current_index] if self.current_index >= 0 else None
        if not r:
            self._show_no_data("(no record)")
            return
        if r.get("review_status") == "unparsed":
            self._show_no_data("(unparsed — rename file and re-browse)")
            return
        try:
            cols = read_csv_columns(r["csv_path"])
        except Exception as e:
            self._show_no_data(f"(CSV read failed: {e})")
            return
        if not cols or len(cols) < 4:
            self._show_no_data("(no numeric data in CSV)")
            return

        wl = np.asarray(cols[0], dtype=float)
        for sig in _SIGNAL_ORDER:
            ax = self._sig_axes[sig]
            y = np.asarray(cols[_SIGNAL_CSV_COL[sig]], dtype=float)
            ax.plot(wl, y, color=_SIGNAL_COLORS[sig], linewidth=1.4)

        self._has_plot_data = True
        self._set_peak_controls_enabled(True)
        self._draw_axes_chrome()
        self._draw_highlight_band()
        self.canvas.draw_idle()

    def _show_no_data(self, msg: str):
        """Render a centered "no data" message on each subplot and lock
        out the peak-finding controls.
        """
        self._has_plot_data = False
        self._set_peak_controls_enabled(False)
        for ax in self._sig_axes.values():
            ax.text(0.5, 0.5, msg,
                    ha="center", va="center",
                    transform=ax.transAxes,
                    color="#888888", style="italic", fontsize=10)
        self._draw_axes_chrome()
        self.canvas.draw_idle()

    def _draw_axes_chrome(self):
        """Apply titles / labels / xlim / active highlight. Called after
        any ax.clear() since that wipes axis cosmetics along with curves.
        """
        self.ax_cd.set_ylabel("CD")
        self.ax_g.set_ylabel("g")
        self.ax_uv.set_ylabel("UV")
        self.ax_uv.set_xlabel("Wavelength (nm)")
        self.ax_cd.set_xlim(*_PLOT_X_RANGE)  # sharex propagates to g + uv
        for ax in self._sig_axes.values():
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        self._update_active_highlight()

    def _update_active_highlight(self):
        """Bold + colored title and thicker spines on the active subplot
        so the user can see at a glance which one Find/Apply act on.
        """
        for sig, ax in self._sig_axes.items():
            is_active = (sig == self._active_signal)
            ax.set_title(
                _SIGNAL_TITLES[sig],
                fontweight="bold" if is_active else "normal",
                color="#1f77b4" if is_active else "#333333",
                fontsize=11 if is_active else 10)
            for spine in ax.spines.values():
                spine.set_linewidth(2.0 if is_active else 0.8)
                spine.set_edgecolor("#1f77b4" if is_active else "#333333")
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        for sig, ax in self._sig_axes.items():
            if event.inaxes is ax:
                self._set_active_signal(sig)
                break

    def _set_active_signal(self, sig: str):
        if sig not in self._sig_axes or sig == self._active_signal:
            return
        self._active_signal = sig
        # Keep the combo in sync without re-emitting currentTextChanged.
        self.active_combo.blockSignals(True)
        self.active_combo.setCurrentText(sig)
        self.active_combo.blockSignals(False)
        self._update_active_highlight()

    def _set_peak_controls_enabled(self, enabled: bool):
        self.find_max_btn.setEnabled(enabled)
        self.find_min_btn.setEnabled(enabled)
        self.apply_peak_btn.setEnabled(enabled)

    # --- highlight band ----------------------------------------------------
    def _on_range_changed(self, _text: str):
        # Inputs are independent QLineEdits; either one changing rebuilds
        # the band. Empty / non-numeric => band hidden.
        self._draw_highlight_band()
        self.canvas.draw_idle()

    def _parse_range(self):
        try:
            lo = float(self.range_min_input.text())
            hi = float(self.range_max_input.text())
        except ValueError:
            return None
        if lo >= hi:
            return None
        return lo, hi

    def _draw_highlight_band(self):
        """Remove any previously-drawn axvspan handles and redraw on each
        subplot from the current range inputs. Sharex aligns the bands
        across the column so they read as one selection.
        """
        for h in self._band_artists:
            try:
                h.remove()
            except Exception:
                pass
        self._band_artists = []
        band = self._parse_range()
        if band is None:
            return
        lo, hi = band
        for ax in self._sig_axes.values():
            self._band_artists.append(
                ax.axvspan(lo, hi, alpha=0.18,
                           color="#ffd24d", zorder=0))

    # --- find max / min within the band -----------------------------------
    def _find_extreme(self, kind: str):
        """Compute argmax/argmin of the active signal restricted to the
        highlighted band. Marks the point on the active subplot and
        shows the wavelength + value in that subplot's bottom-right
        corner. Does NOT persist anything -- write-back is via Apply.
        """
        if not self._has_plot_data:
            return
        band = self._parse_range()
        if band is None:
            QMessageBox.information(
                self, "Set wavelength range first",
                "Enter Range min and Range max (in nm) before searching "
                "for a peak. Find Max / Find Min operate only within the "
                "highlighted band.")
            return
        lo, hi = band
        r = self.records[self.current_index]
        try:
            cols = read_csv_columns(r["csv_path"])
        except Exception as e:
            QMessageBox.warning(
                self, "Could not re-read CSV", str(e))
            return
        wl = np.asarray(cols[0], dtype=float)
        y = np.asarray(cols[_SIGNAL_CSV_COL[self._active_signal]],
                       dtype=float)
        mask = (wl >= lo) & (wl <= hi)
        if not mask.any():
            QMessageBox.information(
                self, "No data in band",
                f"No samples in the {self._active_signal} curve fall "
                f"between {lo:g} and {hi:g} nm.")
            return
        band_wl = wl[mask]
        band_y = y[mask]
        idx = int(np.argmax(band_y) if kind == "max" else np.argmin(band_y))
        peak_wl = float(band_wl[idx])
        peak_val = float(band_y[idx])
        self._computed_peaks[self._active_signal] = (peak_wl, peak_val, kind)
        self._draw_computed_marker(self._active_signal)
        self._draw_corner_readout(self._active_signal)
        self.canvas.draw_idle()

    def _draw_computed_marker(self, sig: str):
        ax = self._sig_axes[sig]
        old = self._marker_artists.pop(sig, None)
        if old is not None:
            for art in old:
                try:
                    art.remove()
                except Exception:
                    pass
        peak = self._computed_peaks.get(sig)
        if peak is None:
            return
        wl, val, _kind = peak
        # Small dot marking the peak's location on the curve. No on-curve
        # annotation -- the value+wavelength text lives in the bottom-right
        # corner readout (_draw_corner_readout) and the field inputs in
        # the center pane, so duplicating it next to the dot was noise.
        marker, = ax.plot([wl], [val], "o",
                          color="#d62728", markersize=4, zorder=6)
        self._marker_artists[sig] = (marker,)

    def _draw_corner_readout(self, sig: str):
        ax = self._sig_axes[sig]
        old = self._readout_artists.pop(sig, None)
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
        peak = self._computed_peaks.get(sig)
        if peak is None:
            return
        wl, val, kind = peak
        # "max: 9070, 491 nm" form -- no "@" so the readout reads as a
        # plain (value, wavelength) pair.
        txt = f"{kind}: {val:.4g}, {wl:.0f} nm"
        handle = ax.text(
            0.98, 0.04, txt,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=10, color="#222",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=0.85,
                      edgecolor="#aaaaaa"))
        self._readout_artists[sig] = handle

    # --- apply to record (with confirm prompt) -----------------------------
    def on_apply_peak(self):
        """Replace the active signal's stored peak with the most-recent
        Find result. Always prompts before overwriting; on confirm,
        writes both columns (value + wavelength), sets edited=1, and
        mirrors into the in-memory record + center-pane fields.
        """
        if not self._has_plot_data:
            return
        if not (0 <= self.current_index < len(self.records)):
            return
        sig = self._active_signal
        peak = self._computed_peaks.get(sig)
        if peak is None:
            QMessageBox.information(
                self, "Nothing to apply",
                f"Click Find Max or Find Min on the {sig} plot first.")
            return
        wl, val, kind = peak
        r = self.records[self.current_index]
        val_field, wl_field = _SIGNAL_DB_FIELDS[sig]
        old_val = r.get(val_field)
        old_wl = r.get(wl_field)
        old_str = (f"{val_field} = {old_val}, {wl_field} = {old_wl}"
                   if old_val is not None or old_wl is not None
                   else "(no stored value yet)")
        new_str = f"{val_field} = {val:.6g}, {wl_field} = {int(round(wl))}"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(f"Apply {sig} peak?")
        box.setText(f"Replace stored {sig} peak with the computed value?")
        box.setInformativeText(
            f"Current: {old_str}\n"
            f"New:     {new_str}\n\n"
            f"This writes both columns to the local DB and marks the row "
            f"as edited (so the values survive re-browse).")
        apply_btn = box.addButton(
            "Apply", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(apply_btn)
        box.exec()
        if box.clickedButton() is not apply_btn:
            return

        # Persist + mirror.
        new_val = float(val)
        new_wl = int(round(wl))
        self.db.update_cell(r["csv_path"], val_field, new_val)
        self.db.update_cell(r["csv_path"], wl_field, new_wl)
        r[val_field] = new_val
        r[wl_field] = new_wl
        # Mirror into the editable line edits in the center pane so the
        # form reflects what's in the DB now. Block signals so this
        # doesn't show up as a "pending edit" in pending_edits.
        for col, new_value in ((val_field, new_val), (wl_field, new_wl)):
            le = self.field_inputs.get(col)
            if le is None:
                continue
            le.blockSignals(True)
            le.setText(str(new_value))
            le.blockSignals(False)

    # --------------------------------------------------------- progress -----
    def _update_progress(self):
        n = len(self.records)
        if n == 0:
            self.progress_label.setText("No records in this batch.")
            return
        c = sum(1 for r in self.records
                if r.get("review_status") == "confirmed")
        rej = sum(1 for r in self.records
                  if r.get("review_status") == "rejected")
        nw = sum(1 for r in self.records
                 if r.get("review_status") == "needs_work")
        reviewed = c + rej + nw
        cur = self.current_index + 1 if self.current_index >= 0 else 0
        self.progress_label.setText(
            f"Record {cur} of {n}   |   "
            f"Reviewed {reviewed} of {n}   "
            f"(confirmed: {c},  rejected: {rej},  needs work: {nw})")
