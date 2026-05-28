"""
Verification / import review window. Non-modal QDialog for reviewing the
records of one freshly-imported batch (the most recent browse). The
reviewer walks through records one at a time, edits parsed fields,
optionally focuses on parser-flagged-uncertain fields, and marks each
record confirmed / rejected / needs-work in the local SQLite store.

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
from PyQt6.QtGui import QBrush, QColor, QCloseEvent
from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QCheckBox, QMessageBox,
)

from models import REVIEW_STATUS_COLORS, VISIBLE_COLUMNS


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

    def __init__(self, db, batch_id: str, parent=None):
        super().__init__(parent)
        self.db = db
        self.batch_id = batch_id
        self.setModal(False)
        self.setWindowTitle(f"Review Imported Batch  —  {batch_id}")
        self.resize(1280, 760)

        # Snapshot the batch contents. Subsequent reviews mutate this list
        # in place so navigation + status counts stay in sync with the DB.
        self.records: list[dict] = self.db.records_in_batch(batch_id)
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

        # ---- Plot placeholder (Phase 2b reserves this space) ----
        plot_box = QGroupBox("Plot (coming in 2b)")
        plot_v = QVBoxLayout(plot_box)
        msg = QLabel(
            "Per-record CD / g-factor / UV-Vis preview will render here "
            "in Phase 2b so peak values can be confirmed against the "
            "actual curves.")
        msg.setWordWrap(True)
        msg.setStyleSheet("color:#888;")
        plot_v.addWidget(msg)
        plot_v.addStretch(1)
        splitter.addWidget(plot_box)

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
        self.record_header.setText(
            f"<b>Record {index + 1} of {len(self.records)}</b>  "
            f"&nbsp;|&nbsp; status: <code>{status}</code>")
        self.path_label.setText(r["csv_path"])
        flagged = _flagged_fields(r)
        for col, le in self.field_inputs.items():
            val = r.get(col)
            le.blockSignals(True)
            le.setText("" if val is None else str(val))
            le.blockSignals(False)
            le.setStyleSheet(_FLAGGED_STYLE if col in flagged else "")
        self.sidebar.setCurrentRow(index)
        self._update_progress()

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
