"""
CD Data Automation -- main GUI window.

Parses metadata from strictly-named CSV files, stores it in a SQLite database
(the source of truth), lets you filter via a cascading GUI, and dispatches the
selected scans to OriginPro for batch plotting.

Filename convention (underscore-separated):
    Series _ Poly1 _ Poly2 _ Ratio _ ConcSolvent _ Speed _ State [_ Temp if AN] _ Gval _ Wavelength

    R1_C-PFBT100_S-F8BT_50x50_20CB_v0p005_AN_T160_gval=0p047_500nm
    R3_F8BT_None_100_20Tol_v0p005_AP_gval=0p042_493nm
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter, QSizePolicy,
    QPushButton, QLabel, QLineEdit, QComboBox, QRadioButton, QButtonGroup,
    QCheckBox, QTableWidget, QTableWidgetItem, QProgressBar, QTextEdit,
    QFileDialog, QGroupBox, QFrame, QHeaderView, QMessageBox, QMenu,
)

from models import COLUMNS, REVIEW_STATUS_COLORS, VISIBLE_COLUMNS, canon_path
from parser import parse_filename
from database import DB


def reveal_in_explorer(path: str) -> None:
    """Reveal a file in the OS file browser, HIGHLIGHTED, without OPENING it.

    Windows: ``explorer /select,<path>`` (explorer returns a nonzero exit code
    even on success, so we never check it). macOS: ``open -R``. Linux:
    ``xdg-open`` on the containing folder (no portable "highlight" verb).

    On Windows the command is passed as one string (shell=False) so CreateProcess
    receives exactly ``explorer /select,"<path>"`` -- the quotes cover spaces and,
    because cmd.exe is not involved, characters like ``&`` / ``^`` in the name are
    not special; valid Windows filenames cannot contain a double-quote, so this is
    always safe. On macOS/Linux the path is a single argv element, so spaces and
    special characters need no shell quoting either.
    """
    full = os.path.normpath(os.path.abspath(path))
    if sys.platform.startswith("win"):
        subprocess.run(f'explorer /select,"{full}"')
    elif sys.platform == "darwin":
        subprocess.run(["open", "-R", full])
    else:
        subprocess.run(["xdg-open", os.path.dirname(full)])


def new_batch_id() -> str:
    """ISO-timestamp-prefixed unique id for one browse-into-DB batch.

    Sortable by string DESC (most-recent batch first) and collision-proof
    even on rapid back-to-back browses thanks to the uuid suffix.
    """
    return (datetime.now().isoformat(timespec="seconds")
            + "_" + uuid.uuid4().hex[:8])


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CD Data Automation")
        self.resize(1100, 760)
        self.db = DB()
        # Staging table edits are gated by this flag. Default off so a stray
        # double-click can't overwrite a parsed value silently.
        self.edit_mode = False
        # In-memory buffer of staged edits, keyed by (csv_path, column).
        # Nothing reaches SQLite until on_save_edits / on_save_and_exit runs.
        self.pending_edits: dict[tuple[str, str], str] = {}
        # Most recent browse batch (a string from new_batch_id). None until
        # the user actually imports something this session; on_open_verification
        # falls back to db.latest_batch_id() so the button still works after
        # a fresh app launch.
        self.latest_batch_id: str | None = None
        # Single VerificationWindow instance, kept alive so it stays usable
        # non-modally and so re-clicks raise the existing window instead of
        # spawning a stack.
        self._verification_win = None
        # Single read-only CloudBrowserWindow instance, same lifecycle as the
        # verification window (kept alive non-modally; re-clicks raise it).
        self._cloud_browser_win = None
        # Cached QBrushes keyed by review_status -- one allocation per status,
        # reused across every refresh_table call. Empty hex -> empty QBrush
        # (default background), which is the "pending" row's natural look in
        # the main staging table.
        self._STATUS_BRUSHES = {
            status: (QBrush(QColor(hex_code)) if hex_code else QBrush())
            for status, hex_code in REVIEW_STATUS_COLORS.items()
        }
        # Active staging-table view: "ALL" or a specific batch_id. Default
        # picked from the DB so app launch already shows the most recent
        # batch rather than dumping every legacy row into the table.
        last = self.db.latest_batch_id()
        self._active_view: str = last if last else "ALL"

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        layout.addWidget(self._build_top_bar())

        # Vertical splitter with three draggable sections, top to bottom:
        #   0) staging table  -- flexible, takes most of the window
        #   1) Filters + Output Plots  -- keeps its needed size
        #   2) Execution / log  -- compact (~3 lines) by default, draggable
        # The staging table and the log are the flexible regions: dragging the
        # bottom handle taller to read more log shrinks the table, and vice
        # versa. The middle (filters/output) holds its size hint.
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_staging_table())

        mid_widget = QWidget()
        mid_layout = QHBoxLayout(mid_widget)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.addWidget(self._build_filter_panel(), stretch=2)
        mid_layout.addWidget(self._build_output_panel(), stretch=1)
        splitter.addWidget(mid_widget)

        splitter.addWidget(self._build_execution_area())

        # Only the staging table absorbs extra space when the window grows,
        # so enlarging/maximizing gives the table more rows while the filters
        # and the log stay compact (the user can still drag any handle).
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 0)
        # Initial heights: a generous table, the filters/output at their hint,
        # and a ~3-line log. The first value is a placeholder -- the stretch
        # factors route any window-vs-sum difference into the table pane.
        splitter.setSizes([
            640,
            mid_widget.sizeHint().height(),
            self._log_compact_h,
        ])
        # Don't let a drag collapse a section to zero; the table keeps its
        # scroll area and the log keeps its ~1-line floor.
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        # Order matters: populate the View dropdown (silently) before the
        # first refresh_table so the combo's selected item matches the
        # initial _active_view rather than snapping to index 0.
        self._refresh_view_options()
        self.refresh_table()
        self.refresh_filter_options()
        # Surface the one-time DB cleanup so users know what changed under them.
        dedup_n = getattr(self.db, "_dedup_count", 0)
        if dedup_n:
            self.log(
                f"De-duplicated DB: removed {dedup_n} duplicate path row(s).")
        backfill_n = getattr(self.db, "_backfill_count", 0)
        if backfill_n:
            self.log(
                f"Assigned record_id to {backfill_n} legacy row(s).")

    # --- top bar -----------------------------------------------------------
    def _build_top_bar(self):
        box = QGroupBox("Data Source")
        h = QHBoxLayout(box)
        self.path_field = QLineEdit()
        self.path_field.setReadOnly(True)
        self.path_field.setPlaceholderText("No folder selected")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.on_browse)
        prune = QPushButton("Prune Missing")
        prune.setToolTip(
            "Delete database rows whose CSV files no longer exist on disk.")
        prune.clicked.connect(self.on_prune_missing)
        review = QPushButton("Review Imported Batch")
        review.setToolTip(
            "Open the verification window for the most recent batch of "
            "imported records (or the latest batch in the DB if you haven't "
            "browsed yet this session).")
        review.clicked.connect(self.on_open_verification)
        review_sel = QPushButton("Review Selected")
        review_sel.setToolTip(
            "Open the verification window for the rows currently selected "
            "in the staging table (Ctrl/Shift-click to multi-select).")
        review_sel.clicked.connect(self.on_review_selected)
        # Cloud promote actions. Both go through the confirmed-only filter
        # in mongo_db.promote_records, so non-confirmed rows in the queue
        # are skipped + logged rather than pushed.
        self.promote_batch_btn = QPushButton("Promote Batch to Cloud")
        self.promote_batch_btn.setToolTip(
            "Upload all CONFIRMED records in the current batch (or the "
            "latest batch in the DB) to MongoDB Atlas. Non-confirmed rows "
            "are skipped and logged.")
        self.promote_batch_btn.clicked.connect(self.on_promote_batch)
        self.promote_sel_btn = QPushButton("Promote Selected to Cloud")
        self.promote_sel_btn.setToolTip(
            "Upload the CONFIRMED rows among the currently-selected staging "
            "rows to MongoDB Atlas. Non-confirmed rows are skipped and "
            "logged.")
        self.promote_sel_btn.clicked.connect(self.on_promote_selected)
        self.origin_status = QLabel()
        self._set_origin_status("neutral", "Origin: not connected")
        connect = QPushButton("Connect to Origin")
        connect.clicked.connect(self.on_connect_origin)
        # Cloud connection indicator. Same three-state pattern as Origin:
        # connected (green) / neutral (gray, idle or unconfigured) / failed
        # (red). Kicked into shape after construction by startup_cloud_check.
        self.cloud_status = QLabel()
        self._set_cloud_status("neutral", "Cloud: idle")
        test_cloud = QPushButton("Test Cloud Connection")
        test_cloud.setToolTip(
            "Ping MongoDB Atlas to confirm credentials. Uses MONGODB_URI "
            "from .env; if unconfigured, this just reports that.")
        test_cloud.clicked.connect(self.on_test_cloud)
        browse_cloud = QPushButton("Browse Cloud")
        browse_cloud.setToolTip(
            "Open the read-only cloud browser: fetch records from MongoDB "
            "Atlas and plot their embedded spectra. No editing -- corrections "
            "happen by re-promoting a local record.")
        browse_cloud.clicked.connect(self.on_browse_cloud)
        h.addWidget(QLabel("Folder:"))
        h.addWidget(self.path_field, stretch=1)
        h.addWidget(browse)
        h.addWidget(prune)
        h.addWidget(review)
        h.addWidget(self.promote_batch_btn)
        h.addWidget(review_sel)
        h.addWidget(self.promote_sel_btn)
        h.addSpacing(20)
        h.addWidget(connect)
        h.addWidget(self.origin_status)
        h.addSpacing(10)
        h.addWidget(test_cloud)
        h.addWidget(browse_cloud)
        h.addWidget(self.cloud_status)
        return box

    # --- staging table -----------------------------------------------------
    _STAGING_TITLE_READONLY = ("Staging Area  "
                               "(read-only - click Edit to change values)")
    _STAGING_TITLE_EDITING  = ("Staging Area  "
                               "(EDITING - changes save to database)")

    # Pending-edit cell tint (light yellow) -- makes staged-but-uncommitted
    # changes obvious before they hit the DB.
    _PENDING_TINT = QColor("#fff3cd")

    def _build_staging_table(self):
        self.staging_box = QGroupBox(self._STAGING_TITLE_READONLY)
        v = QVBoxLayout(self.staging_box)

        # Header row: [ toast ][ stretch ][ Edit  OR  Save / Cancel / Save&Exit ].
        # The toast and the trio live alongside the Edit button; visibility is
        # toggled so they occupy the same conceptual slot on the right.
        header = QHBoxLayout()

        # View selector: scopes the staging table to one batch (or "All").
        # Lives in the staging header rather than the top bar because it
        # only affects what THIS table shows -- the review buttons in the
        # top bar still act on their own targets (latest batch / selection).
        header.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.setMinimumWidth(320)
        self.view_combo.setToolTip(
            "Scope which records appear in the staging table. Other batches "
            "stay in the DB; switch the dropdown or pick 'All records' to "
            "see them.")
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        header.addWidget(self.view_combo)
        header.addSpacing(20)

        self.toast = QLabel()
        self.toast.setStyleSheet(
            "background:#27ae60;color:white;padding:4px 10px;"
            "border-radius:4px;font-weight:bold;")
        self.toast.hide()
        header.addWidget(self.toast)
        header.addStretch(1)

        # Local-only delete: removes selected rows from the working SQLite DB.
        # Deliberately worded "Local" so it's never confused with the cloud
        # vault -- promoted records stay safe in MongoDB.
        self.delete_btn = QPushButton("Delete Selected")
        self.delete_btn.setToolTip(
            "Delete the selected rows from the LOCAL database only. Does not "
            "affect records already promoted to the cloud. Cannot be undone.")
        self.delete_btn.clicked.connect(self.on_delete_selected)
        header.addWidget(self.delete_btn)

        self.edit_btn = QPushButton("Edit")
        self.edit_btn.setToolTip(
            "Enter edit mode. Cells stage in memory; nothing reaches the "
            "database until you click Save or Save & Exit.")
        self.edit_btn.clicked.connect(self.on_enter_edit)
        header.addWidget(self.edit_btn)

        # Trio container: occupies the same slot, hidden until Edit is clicked.
        self.edit_trio = QWidget()
        trio = QHBoxLayout(self.edit_trio)
        trio.setContentsMargins(0, 0, 0, 0)
        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Commit staged edits to the database. Stays in edit mode.")
        self.save_btn.clicked.connect(self.on_save_edits)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Discard staged edits and exit edit mode.")
        self.cancel_btn.clicked.connect(self.on_cancel_edits)
        self.save_exit_btn = QPushButton("Save && Exit")
        self.save_exit_btn.setToolTip("Commit staged edits and exit edit mode.")
        self.save_exit_btn.clicked.connect(self.on_save_and_exit)
        trio.addWidget(self.save_btn)
        trio.addWidget(self.cancel_btn)
        trio.addWidget(self.save_exit_btn)
        self.edit_trio.hide()
        header.addWidget(self.edit_trio)

        v.addLayout(header)

        self.table = QTableWidget()
        # Hidden user-metadata columns (record_id, flags, verified, ...) are
        # stored in the DB and round-tripped on edits, but never displayed
        # here -- VISIBLE_COLUMNS filters them out of the staging view.
        self.table.setColumnCount(len(VISIBLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(VISIBLE_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        # Row-level multi-select so the user can Ctrl/Shift-click groups of
        # records to feed into "Review Selected". Cell-level editing in edit
        # mode still works -- double-click enters the cell editor.
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.itemChanged.connect(self.on_cell_edited)
        # Right-click menu: reveal a row's CSV in the system file browser
        # (reveal only -- never opens the file).
        self.table.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(
            self._on_table_context_menu)
        # Expand vertically so dragging the splitter (or maximizing the
        # window) grows the visible row area instead of leaving empty space.
        # The table keeps its own scroll bar for rows beyond what fits.
        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.table, stretch=1)
        self.staging_box.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        return self.staging_box

    # --- cascading filter panel -------------------------------------------
    def _build_filter_panel(self):
        box = QGroupBox("Filters")
        g = QGridLayout(box)

        g.addWidget(QLabel("Solvent:"), 0, 0)
        self.f_solvent = QComboBox()
        g.addWidget(self.f_solvent, 0, 1)

        g.addWidget(QLabel("System:"), 1, 0)
        self.comp_group = QButtonGroup(self)
        comp_row = QHBoxLayout()
        for i, t in enumerate(["Any", "1-Component", "2-Component"]):
            rb = QRadioButton(t)
            if i == 0:
                rb.setChecked(True)
            self.comp_group.addButton(rb, i)
            comp_row.addWidget(rb)
        self.comp_group.buttonClicked.connect(self._toggle_conditional)
        cw = QWidget(); cw.setLayout(comp_row)
        g.addWidget(cw, 1, 1)

        self.lbl_config = QLabel("Configuration:")
        g.addWidget(self.lbl_config, 2, 0)
        self.f_config = QComboBox()
        self.f_config.addItems(
            ["Any", "achiral+achiral", "chiral+achiral", "chiral+chiral"])
        g.addWidget(self.f_config, 2, 1)

        # Polymer dropdowns. All three siblings live in the grid; visibility
        # is driven by the System radios via _toggle_conditional, so only the
        # row(s) relevant to the current selection appear. Options are filled
        # by _refresh_polymer_options from the actual rows in `scans`, so the
        # dropdowns only ever offer polymers that exist in the loaded data.
        self.lbl_polymer = QLabel("Polymer:")
        g.addWidget(self.lbl_polymer, 3, 0)
        self.f_polymer = QComboBox()
        g.addWidget(self.f_polymer, 3, 1)
        self.lbl_polymerA = QLabel("Polymer A:")
        g.addWidget(self.lbl_polymerA, 4, 0)
        self.f_polymerA = QComboBox()
        g.addWidget(self.f_polymerA, 4, 1)
        self.lbl_polymerB = QLabel("Polymer B:")
        g.addWidget(self.lbl_polymerB, 5, 0)
        self.f_polymerB = QComboBox()
        g.addWidget(self.f_polymerB, 5, 1)

        g.addWidget(QLabel("Film state:"), 6, 0)
        self.state_group = QButtonGroup(self)
        state_row = QHBoxLayout()
        for i, t in enumerate(["Any", "As Printed", "Annealed"]):
            rb = QRadioButton(t)
            if i == 0:
                rb.setChecked(True)
            self.state_group.addButton(rb, i)
            state_row.addWidget(rb)
        self.state_group.buttonClicked.connect(self._toggle_conditional)
        sw = QWidget(); sw.setLayout(state_row)
        g.addWidget(sw, 6, 1)

        self.lbl_temp = QLabel("Anneal T (°C):")
        g.addWidget(self.lbl_temp, 7, 0)
        self.f_temp = QComboBox()
        g.addWidget(self.f_temp, 7, 1)

        apply_btn = QPushButton("Apply Filters")
        apply_btn.clicked.connect(self.on_apply_filters)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.on_clear_filters)
        g.addWidget(apply_btn, 8, 0)
        g.addWidget(clear_btn, 8, 1)

        self._toggle_conditional()
        return box

    def _toggle_conditional(self, *_):
        # All conditional rows use setVisible so empty rows collapse out of
        # the grid -- previously Configuration / Anneal T merely disabled,
        # which left visible-but-grey rows in unrelated System selections.
        cid = self.comp_group.checkedId()
        is_one = cid == 1
        is_two = cid == 2
        self.lbl_config.setVisible(is_two)
        self.f_config.setVisible(is_two)
        self.lbl_polymer.setVisible(is_one)
        self.f_polymer.setVisible(is_one)
        self.lbl_polymerA.setVisible(is_two)
        self.f_polymerA.setVisible(is_two)
        self.lbl_polymerB.setVisible(is_two)
        self.f_polymerB.setVisible(is_two)
        is_annealed = self.state_group.checkedId() == 2
        self.lbl_temp.setVisible(is_annealed)
        self.f_temp.setVisible(is_annealed)

    # --- output panel ------------------------------------------------------
    def _build_output_panel(self):
        box = QGroupBox("Output Plots")
        v = QVBoxLayout(box)
        self.chk_cd = QCheckBox("Wavelength vs. CD"); self.chk_cd.setChecked(True)
        self.chk_g = QCheckBox("Wavelength vs. G-value")
        self.chk_uv = QCheckBox("Wavelength vs. UV-Vis")
        self.chk_mm = QCheckBox("Mueller Matrix  (coming soon)")
        self.chk_mm.setEnabled(False)

        # One row per real signal: [ checkbox ][ x button ]. The "x" clears that
        # one plot from Origin and unchecks its box. Store buttons on self so
        # tests / future code can poke at them by label.
        self.clear_btns = {}
        for chk, label in [(self.chk_cd, "CD"),
                           (self.chk_g,  "G-value"),
                           (self.chk_uv, "UV-Vis")]:
            row = QHBoxLayout()
            row.addWidget(chk, stretch=1)
            x_btn = QPushButton("x")
            x_btn.setFixedWidth(24)
            x_btn.setToolTip(f"Clear {label} from Origin and uncheck this box")
            # `sig=label` captures the value at lambda-definition time, otherwise
            # all three buttons would close over the loop variable's final value.
            x_btn.clicked.connect(lambda _checked, sig=label: self.on_clear_signal(sig))
            row.addWidget(x_btn)
            v.addLayout(row)
            self.clear_btns[label] = x_btn

        # Mueller Matrix is disabled and has no x button.
        v.addWidget(self.chk_mm)

        self.generate_btn = QPushButton("Plot Selected")
        self.generate_btn.clicked.connect(self.on_generate_plots)
        v.addWidget(self.generate_btn)
        v.addStretch(1)
        return box

    # --- execution area ----------------------------------------------------
    def _build_execution_area(self):
        box = QGroupBox("Execution")
        v = QVBoxLayout(box)
        self.progress = QProgressBar()
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        # Compact by default (~3 lines) so the staging table gets the room.
        # Lives in the main vertical splitter, so the user can drag it taller
        # to read more output; it still scrolls for content beyond what fits.
        self.log_box.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        fm = self.log_box.fontMetrics()
        v.addWidget(self.progress)
        v.addWidget(self.log_box)
        # Default pane height for the splitter: the group box's chrome
        # (title, margins, progress bar) -- which is its size hint minus the
        # log box's own (large) default hint -- plus an explicit 3-line log.
        # Floor the log at ~1 line so a drag can still make it shorter.
        three_lines = fm.lineSpacing() * 3 + 4
        chrome = box.sizeHint().height() - self.log_box.sizeHint().height()
        self._log_compact_h = max(chrome, 0) + three_lines
        self.log_box.setMinimumHeight(fm.lineSpacing() + 8)
        return box

    # ----- behavior --------------------------------------------------------
    def log(self, msg: str):
        self.log_box.append(msg)

    def on_browse(self):
        if not self._guard_pending():
            return
        folder = QFileDialog.getExistingDirectory(self, "Select CSV folder")
        if not folder:
            return
        self.path_field.setText(folder)
        # One batch_id covers this entire browse. Every file SEEN in the
        # folder gets re-tagged with it -- including files whose names fail
        # parsing, which land as placeholder "unparsed" rows so the user
        # can see (and fix) them rather than wondering why they vanished.
        batch_id = new_batch_id()
        new = updated = preserved = unparsed = err = ignored_meta = 0
        for fn in os.listdir(folder):
            if not fn.lower().endswith(".csv"):
                continue
            # macOS AppleDouble sidecar files ("._foo.csv") are metadata
            # created when files pass through Mac-synced storage, not real
            # data. Skip them entirely so they never become phantom rows.
            if fn.startswith("._"):
                ignored_meta += 1
                continue
            full = os.path.join(folder, fn)
            try:
                meta = parse_filename(full)
            except Exception as e:
                # Filename didn't fit the convention. Record a placeholder
                # row (review_status='unparsed', parse_error=msg) so the
                # file appears in the staging table as a worklist entry
                # for the upcoming rename pass instead of being silently
                # dropped.
                try:
                    self.db.upsert_unparsed(full, str(e), batch_id=batch_id)
                    unparsed += 1
                except Exception as ue:
                    err += 1
                    self.log(
                        f"  could not record unparsed file: {fn} -> {ue}")
                continue
            try:
                result = self.db.upsert_preserving_edits(
                    meta, batch_id=batch_id)
                if result == "new":
                    new += 1
                elif result == "updated":
                    updated += 1
                elif result == "preserved":
                    preserved += 1
            except Exception as e:
                err += 1
                self.log(f"  DB write failed: {fn} -> {e}")
        bits = [
            f"{new} new",
            f"{updated} updated",
            f"{preserved} preserved (manually edited)",
            f"{unparsed} unparsed (visible in table, flagged)",
        ]
        if err:
            bits.append(f"{err} errored")
        self.log("Ingested: " + ", ".join(bits) + ".")
        if ignored_meta:
            self.log(
                f"Ignored {ignored_meta} macOS metadata file(s) (._*).")
        # Only treat the batch as reviewable when at least one file actually
        # landed as a new row. A re-browse of an already-known folder isn't
        # something the reviewer cares about.
        # Every file SEEN in this browse -- newly-inserted, refreshed,
        # edit-preserved, or unparsed -- has now been re-tagged with this
        # batch_id, so the staging view scoped to it mirrors exactly
        # what's in the folder right now. Switch the active view to this
        # batch whenever any rows were touched.
        total_seen = new + updated + preserved + unparsed
        if total_seen > 0:
            self.latest_batch_id = batch_id
            self._active_view = batch_id
            self.log(
                f"Batch {batch_id}: {total_seen} record(s) tagged to "
                f"this folder and available in Review Imported Batch.")
        # Prune rows whose files are no longer on disk. Done after ingest so
        # the row counts in the staging table match the folder.
        self._do_prune("after browse")
        self._refresh_view_options()
        self.refresh_filter_options()
        self.refresh_table()

    def on_prune_missing(self):
        """Full-DB sweep for orphan rows. Honors the unsaved-changes guard so
        an in-progress edit isn't silently discarded by refresh_table.
        """
        if not self._guard_pending():
            return
        self._do_prune("manual sweep")
        self.refresh_filter_options()
        self.refresh_table()

    def _do_prune(self, source: str):
        try:
            pruned, pruned_edited = self.db.prune_missing()
        except Exception as e:
            self.log(f"Prune failed ({source}): {e}")
            return
        if pruned:
            tail = (f" ({pruned_edited} was manually edited)"
                    if pruned_edited else "")
            self.log(f"Pruned {pruned} missing file(s){tail}.")
        elif source == "manual sweep":
            self.log("No missing files to prune.")

    def on_open_verification(self):
        """Open the verification window for the most-recent batch.

        Prefers the batch from this session's last on_browse; falls back to
        the highest-sorting batch_id in the DB so the button still works at
        app startup before any browse.
        """
        if not self._guard_pending():
            return
        batch_id = self.latest_batch_id or self.db.latest_batch_id()
        if not batch_id:
            self.log("No batch to review. Browse a folder first.")
            return
        records = self.db.records_in_batch(batch_id)
        if not records:
            self.log(f"Batch {batch_id} has no records to review.")
            return
        self._launch_verification_window(
            records,
            title_suffix=f"batch {batch_id}",
            log_phrase=f"batch {batch_id} ({len(records)} record(s))")

    def on_review_selected(self):
        """Open the verification window for the currently-selected staging
        rows. Subset path -- whatever the user has multi-selected, that's
        the queue. No selection -> friendly log, no window.
        """
        if not self._guard_pending():
            return
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.log("Select one or more rows first.")
            return
        # selectedRows() can come back in click-order; sort by row index so
        # the verification queue mirrors what the user sees in the table.
        indices = sorted(idx.row() for idx in sel)
        paths = [self.current_rows[r]["csv_path"]
                 for r in indices
                 if 0 <= r < len(self.current_rows)]
        records = self.db.records_by_paths(paths)
        if not records:
            self.log("Selected rows have no matching records in the DB.")
            return
        self._launch_verification_window(
            records,
            title_suffix=f"selected ({len(records)} records)",
            log_phrase=f"{len(records)} selected record(s)")

    def _launch_verification_window(self, records, *,
                                     title_suffix: str, log_phrase: str):
        """Shared launcher used by both Review buttons. Reuses an existing
        open window (raise to front) and wires the close signals to
        _after_review so the staging table refreshes on exit.
        """
        if (self._verification_win is not None
                and self._verification_win.isVisible()):
            self._verification_win.raise_()
            self._verification_win.activateWindow()
            self.log("Verification window already open; raised to front.")
            return

        # Lazy import so the GUI module graph stays acyclic and a future
        # plotting-pull-in by verification_window can't break the rest of
        # the GUI just because originpro is missing.
        try:
            from verification_window import VerificationWindow
        except Exception as e:
            self.log(
                f"Failed to open verification window: "
                f"{type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return

        self._verification_win = VerificationWindow(
            self.db, records, title_suffix=title_suffix, parent=self)
        # Belt-and-braces: both signals route to the same idempotent slot.
        # QDialog.finished covers accept / reject / X; reviewFinished is
        # the explicit signal the window emits on Save & Close / X close.
        self._verification_win.finished.connect(
            lambda *_args: self._after_review())
        self._verification_win.reviewFinished.connect(self._after_review)
        self._verification_win.show()
        self.log(f"Opened verification window for {log_phrase}.")

    def _after_review(self):
        """Slot invoked when the verification window closes. Drops the
        cached reference (so the next click builds a fresh window) and
        refreshes the staging table so review_status / verified / edited
        changes show up immediately as the new row tinting.
        """
        self._verification_win = None
        self.refresh_table()

    # ----- view selector --------------------------------------------------
    def _refresh_view_options(self):
        """Repopulate the View dropdown from distinct batch_ids in the DB.

        Item user-data is the actual batch_id (or "ALL"); item text is a
        human-readable label combining the ISO timestamp, source folder,
        and row count. Preserves current selection across rebuilds; falls
        back to "All records" if the previously-active view has vanished
        (e.g. all its files were pruned).
        """
        summary = self.db.batches_summary()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("All records", "ALL")
        for bid, folder, count in summary:
            # batch_id format: ISO 'YYYY-MM-DDTHH:MM:SS_<uuid8>'. First 19
            # chars are the timestamp; swap the 'T' for a space so the
            # label reads naturally.
            ts = bid[:19].replace("T", " ")
            self.view_combo.addItem(
                f"{ts}  —  {folder}  ({count})", bid)
        idx = self.view_combo.findData(self._active_view)
        if idx < 0:
            # Active view no longer exists -- silently fall back to ALL.
            self._active_view = "ALL"
            idx = 0
        self.view_combo.setCurrentIndex(idx)
        self.view_combo.blockSignals(False)

    def _on_view_changed(self, *_):
        """User picked a different view. Refresh the table, guarding any
        pending staging-table edits and reverting the combo on cancel.
        """
        new_view = self.view_combo.currentData() or "ALL"
        if new_view == self._active_view:
            return
        if not self._guard_pending():
            # User cancelled -- snap the combo back to the old view.
            self.view_combo.blockSignals(True)
            idx = self.view_combo.findData(self._active_view)
            if idx >= 0:
                self.view_combo.setCurrentIndex(idx)
            self.view_combo.blockSignals(False)
            return
        self._active_view = new_view
        self.refresh_table()

    # Three distinct indicator states:
    #   connected -> green   (attached + verified)
    #   neutral   -> gray    (no Origin running, but nothing is wrong)
    #   failed    -> red     (originpro missing, or attach/verify exception)
    _STATUS_COLORS = {"connected": "#27ae60",
                      "neutral":   "#7f8c8d",
                      "failed":    "#c0392b"}

    def _set_origin_status(self, state: str, text: str):
        color = self._STATUS_COLORS.get(state, "#c0392b")
        self.origin_status.setText(f"  {text}  ")
        self.origin_status.setStyleSheet(
            f"background:{color};color:white;border-radius:4px;")

    def _set_cloud_status(self, state: str, text: str):
        """Same three-state indicator pattern as Origin -- different label."""
        color = self._STATUS_COLORS.get(state, "#c0392b")
        self.cloud_status.setText(f"  {text}  ")
        self.cloud_status.setStyleSheet(
            f"background:{color};color:white;border-radius:4px;")

    def _connect_origin(self, launch: bool, verbose: bool = True):
        """Attach to Origin and verify. Two modes:

            launch=False  -> detect-only. Attaches if an Origin instance is
                             already running; never spawns one. Used at startup
                             so opening the GUI doesn't boot Origin.
            launch=True   -> attach to a running instance if present, otherwise
                             launch Origin with a blank session.

        Verifies the connection by reading version + EXE path back. On any
        failure: red indicator, log the exception, no crash. Plotting does not
        depend on this -- originpro auto-attaches on first plot regardless.
        """
        # 1. Lazy import: GUI still launches when originpro isn't installed.
        try:
            import originpro as op
        except ImportError as e:
            self._set_origin_status("failed", "Origin: not available")
            if verbose:
                self.log(f"originpro is not installed: {e}")
            return

        # 2. Detect a RUNNING Origin without triggering a launch. Origin
        #    registers in the ROT under one of two ProgIDs depending on how it
        #    was started; the single-instance automation server name
        #    ('Origin.ApplicationSI') is what a normally-opened Origin uses, so
        #    try it FIRST. GetActiveObject raises if the ProgID isn't in the
        #    ROT, so each probe is harmless and cannot spawn Origin.
        running_progid = None
        try:
            import win32com.client
            for progid in ("Origin.ApplicationSI", "Origin.Application"):
                try:
                    win32com.client.GetActiveObject(progid)
                    running_progid = progid
                    break
                except Exception:
                    continue
        except Exception:
            running_progid = None
        running = running_progid is not None

        # 3. Detect-only mode: if nothing's running, stay neutral and hint.
        if not running and not launch:
            self._set_origin_status("neutral", "Origin: not connected")
            if verbose:
                self.log("Origin not running - click Connect to start it.")
            return

        # 4. Attach (and launch if needed). When an existing instance was
        #    detected, explicitly call op.attach() so originpro binds to THAT
        #    instance rather than spawning a fresh one. op.set_show(True) then
        #    just ensures the window is visible. Same call the graphing module
        #    uses on entry, so behavior is consistent.
        try:
            if running:
                if verbose:
                    self.log(f"Found running Origin via {running_progid}.")
                try:
                    op.attach()
                except Exception:
                    # Attach failure is non-fatal -- set_show below still tries
                    # to bind, and if that also fails we land in the outer
                    # except and report connection failed.
                    pass
            op.set_show(True)

            # 5. Verify with cheap read-backs. Both are wrapped so a missing
            #    API on one doesn't blow up the other.
            version = None
            try:
                version = op.lt_float("@V")        # numeric build version
            except Exception:
                version = None
            exe_path = ""
            try:
                exe_path = op.path("e") or ""      # Origin EXE folder
            except Exception:
                exe_path = ""

            # Fallback proof-of-life: create a hidden workbook and immediately
            # destroy it. Nothing left behind in Origin.
            if version is None and not exe_path:
                test = op.new_book("w", lname="_diao_connect_test_", hidden=True)
                if test is None:
                    raise RuntimeError("could not create test workbook")
                test.destroy()

            # 6. Single useful log line + green indicator. "Found running" vs
            #    "Launched" so the user knows whether *we* spawned Origin.
            action = "Found running" if running else "Launched"
            bits = [f"{action} OriginPro"]
            if version is not None:
                bits.append(str(version))
            if exe_path:
                bits.append(f"at {exe_path}")
            if verbose:
                self.log(" ".join(bits) + ".")
            self._set_origin_status("connected", "Origin: connected")
        except Exception as e:
            self._set_origin_status("failed", "Origin: connection failed")
            if verbose:
                self.log(f"Origin connect failed: {e}")

    def on_connect_origin(self):
        """User clicked Connect: attach if running, otherwise launch Origin."""
        self._connect_origin(launch=True)

    # ----- cloud (MongoDB Atlas) ------------------------------------------
    def on_test_cloud(self):
        """Ping Atlas. Updates the cloud_status indicator + logs the result.

        Wrapped so a missing pymongo / config import never reaches the GUI;
        the user just sees a 'failed' indicator and a clear log line.
        """
        self._set_cloud_status("neutral", "Cloud: testing...")
        QApplication.processEvents()
        try:
            from mongo_db import test_connection
        except Exception as e:
            self._set_cloud_status("failed", "Cloud: import failed")
            self.log(f"Cloud import failed: {type(e).__name__}: {e}")
            return
        try:
            ok, msg = test_connection()
        except Exception as e:
            # test_connection is meant to never raise, but belt-and-braces.
            self._set_cloud_status("failed", "Cloud: error")
            self.log(f"Cloud test error: {type(e).__name__}: {e}")
            return
        if ok:
            self._set_cloud_status("connected", "Cloud: connected")
            self.log(f"Cloud: {msg}")
        else:
            self._set_cloud_status("failed", "Cloud: not connected")
            self.log(f"Cloud: {msg}")

    def on_browse_cloud(self):
        """Open the read-only cloud browser. Reuses an existing open window
        (raise to front) like the verification window. Fetch failures are
        handled inside the dialog (it opens and shows a clear empty state),
        so the app never crashes if the cloud is unconfigured/unreachable.
        """
        if (self._cloud_browser_win is not None
                and self._cloud_browser_win.isVisible()):
            self._cloud_browser_win.raise_()
            self._cloud_browser_win.activateWindow()
            self.log("Cloud browser already open; raised to front.")
            return

        # Lazy import so the GUI launches even if matplotlib / pymongo aren't
        # importable -- the failure surfaces here as a log line, not a crash.
        try:
            from cloud_browser_window import CloudBrowserWindow
        except Exception as e:
            self.log(
                f"Failed to open cloud browser: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return

        self._cloud_browser_win = CloudBrowserWindow(log=self.log, parent=self)
        self._cloud_browser_win.finished.connect(
            lambda *_args: self._after_cloud_browse())
        self._cloud_browser_win.show()
        self.log("Opened cloud browser (read-only).")

    def _after_cloud_browse(self):
        """Drop the cached cloud-browser reference on close so the next click
        builds a fresh window. Nothing to refresh locally -- the browser is
        read-only and never touches the local DB.
        """
        self._cloud_browser_win = None

    def on_promote_batch(self):
        """Promote all confirmed records in the currently-viewed batch.

        'Viewed batch' = the active View dropdown selection if it's a real
        batch_id, else the most recent batch (session > DB). 'All records'
        view falls back to the latest batch -- promoting every confirmed
        row across history is too easy to trigger by accident.
        """
        if not self._guard_pending():
            return
        if self._active_view and self._active_view != "ALL":
            batch_id = self._active_view
        else:
            batch_id = self.latest_batch_id or self.db.latest_batch_id()
        if not batch_id:
            self.log("No batch to promote. Browse a folder first.")
            return
        records = self.db.records_in_batch(batch_id)
        if not records:
            self.log(f"Batch {batch_id} has no records.")
            return
        self._run_promote(records, source=f"batch {batch_id}")

    def on_promote_selected(self):
        """Promote the currently-selected staging rows (confirmed ones only).
        """
        if not self._guard_pending():
            return
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.log("Select one or more rows first.")
            return
        indices = sorted(idx.row() for idx in sel)
        paths = [self.current_rows[r]["csv_path"]
                 for r in indices
                 if 0 <= r < len(self.current_rows)]
        records = self.db.records_by_paths(paths)
        if not records:
            self.log("Selected rows have no matching records in the DB.")
            return
        self._run_promote(records, source=f"selection ({len(records)})")

    def on_delete_selected(self):
        """Delete the currently-selected staging rows from the LOCAL DB only.

        Confirmation-gated and irreversible locally, but never touches the
        cloud: anything already promoted stays in MongoDB. After deleting we
        refresh the table + filter options so the removed rows disappear.
        """
        if not self._guard_pending():
            return
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.log("Select one or more rows to delete.")
            return
        indices = sorted(idx.row() for idx in sel)
        paths = [self.current_rows[r]["csv_path"]
                 for r in indices
                 if 0 <= r < len(self.current_rows)]
        if not paths:
            self.log("Select one or more rows to delete.")
            return

        confirm = QMessageBox.question(
            self,
            "Delete from LOCAL database?",
            f"Delete {len(paths)} record(s) from the LOCAL database?\n\n"
            f"This does not affect any records already promoted to the "
            f"cloud. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if confirm != QMessageBox.StandardButton.Yes:
            self.log("Delete cancelled.")
            return

        try:
            n = self.db.delete_records(paths)
        except Exception as e:
            self.log(f"Delete failed: {type(e).__name__}: {e}")
            return

        self.log(f"Deleted {n} local record(s).")
        self.refresh_table()
        self.refresh_filter_options()

    def _on_table_context_menu(self, pos):
        """Right-click menu on a staging row: 'Show in Explorer' (reveal only).

        Acts on the row UNDER the cursor (independent of the multi-row
        selection used by promote/delete), so a single right-click reveals one
        file. A click on empty space (row < 0) shows no menu.
        """
        index = self.table.indexAt(pos)
        row = index.row()
        if row < 0 or row >= len(getattr(self, "current_rows", [])):
            return
        menu = QMenu(self.table)
        reveal_act = menu.addAction("Show in Explorer")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is reveal_act:
            self._reveal_local_row(row)

    def _reveal_local_row(self, row: int):
        """Reveal a local staging row's CSV in the file browser, or explain
        that the file is gone. Local rows are THIS machine's own files, so a
        missing file just means it was moved/deleted locally (no other-uploader
        concept here -- that's only relevant in the cloud browser).
        """
        csv_path = self.current_rows[row].get("csv_path")
        name = os.path.basename(csv_path) if csv_path else "(unknown)"
        try:
            exists = bool(csv_path) and os.path.exists(csv_path)
        except Exception:
            exists = False
        if not exists:
            QMessageBox.information(
                self, "File not found",
                f"File not found on this machine: {name}. "
                f"It may have been moved or deleted locally.")
            self.log(f"File not found on this machine: {name}.")
            return
        try:
            reveal_in_explorer(csv_path)
            self.log(f"Revealed in Explorer: {name}")
        except Exception as e:
            self.log(f"Could not reveal {name}: {type(e).__name__}: {e}")

    def _run_promote(self, records, *, source: str):
        """Shared driver for both promote buttons. Blocks the promote buttons,
        flips the cloud indicator to 'promoting', calls mongo_db, then
        mirrors successful upserts into the local DB and refreshes the
        staging table so the ↑ marker appears.

        Stays on the main thread -- the network op blocks briefly, like
        Origin. Caller-side processEvents keeps the UI responsive for
        users with many rows.
        """
        # Lazy import so the GUI still launches without pymongo / .env.
        try:
            from mongo_db import promote_records
        except Exception as e:
            self.log(f"Cloud import failed: {type(e).__name__}: {e}")
            self._set_cloud_status("failed", "Cloud: import failed")
            return

        self.promote_batch_btn.setEnabled(False)
        self.promote_sel_btn.setEnabled(False)
        old_text_b = self.promote_batch_btn.text()
        old_text_s = self.promote_sel_btn.text()
        self.promote_batch_btn.setText("Promoting...")
        self.promote_sel_btn.setText("Promoting...")
        self._set_cloud_status("neutral", "Cloud: promoting...")
        # Pre-count confirmed so the user sees what's about to be attempted.
        n_conf = sum(1 for r in records
                     if r.get("review_status") == "confirmed"
                     and r.get("verified"))
        self.log(
            f"Promoting {n_conf} confirmed record(s) from {source} "
            f"(of {len(records)} total)...")
        QApplication.processEvents()

        summary = {"pushed": 0, "already": 0, "conflicts": 0, "skipped": 0,
                   "failed": 0, "promoted": [], "conflict_details": []}
        try:
            summary = promote_records(records, log=self.log)
        except Exception as e:
            # promote_records is meant to never raise, but if pymongo throws
            # something unexpected we still need to recover the UI state.
            self.log(
                f"Promote failed unexpectedly: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        finally:
            self.promote_batch_btn.setText(old_text_b)
            self.promote_sel_btn.setText(old_text_s)
            self.promote_batch_btn.setEnabled(True)
            self.promote_sel_btn.setEnabled(True)

        # Mirror cloud inserts + already-present records into the local DB.
        # Done outside the try/except so a partial network failure still
        # records what DID land.
        for csv_path, promoted_at in summary["promoted"]:
            try:
                self.db.mark_promoted(csv_path, promoted_at)
            except Exception as e:
                self.log(
                    f"  could not mark promoted locally for {csv_path}: "
                    f"{type(e).__name__}: {e}")

        # Surface each cross-machine conflict as a modal error dialog. The
        # write was already cancelled server-side -- nothing was overwritten.
        # Main thread (we're in the GUI handler), so this is safe.
        for _kind, _filename, _existing_id, message in summary.get(
                "conflict_details", []):
            QMessageBox.critical(self, "Cloud upload cancelled", message)

        self.log(
            f"Promote summary: inserted {summary['pushed']}, "
            f"already-present {summary['already']}, "
            f"cancelled {summary['conflicts']} (conflicts), "
            f"skipped {summary['skipped']} (not confirmed), "
            f"failed {summary['failed']}.")
        # Permanent record of which cloud doc to check for each cancelled
        # conflict: filename + the existing record's _id.
        for _kind, filename, existing_id, _msg in summary.get(
                "conflict_details", []):
            self.log(
                f"  cancelled (conflict): {filename} -- existing _id: "
                f"{existing_id}")

        # Indicator: errors if anything failed OR a conflict was cancelled;
        # connected if anything landed (insert or already-present); neutral if
        # there was simply nothing eligible.
        landed = summary["pushed"] + summary["already"]
        if summary["failed"] or summary["conflicts"]:
            self._set_cloud_status("failed", "Cloud: errors (see log)")
        elif landed > 0:
            self._set_cloud_status("connected", "Cloud: connected")
        else:
            self._set_cloud_status("neutral", "Cloud: idle")

        self.refresh_table()

    def startup_origin_check(self):
        """Startup probe: attach if Origin is already open, otherwise stay
        neutral. Never spawns Origin -- the user must click Connect for that.
        """
        self._connect_origin(launch=False)

    def refresh_filter_options(self):
        self.f_solvent.blockSignals(True)
        self.f_solvent.clear()
        self.f_solvent.addItem("Any")
        self.f_solvent.addItems(self.db.distinct("solvent"))
        self.f_solvent.blockSignals(False)
        self.f_temp.clear()
        self.f_temp.addItem("Any")
        self.f_temp.addItems([str(t) for t in self.db.distinct("anneal_temp")])
        self._refresh_polymer_options()

    def _refresh_polymer_options(self):
        """Populate polymer dropdowns from distinct values actually in scans.

        - f_polymer        (1-component slot): distinct p1_name across rows
                            with n_components=1.
        - f_polymerA / B   (2-component slots): the UNION of distinct p1_name
                            and p2_name across rows with n_components=2 -- a
                            single combined list, because storage slot order
                            (p1 vs p2) doesn't constrain which polymer the
                            user might want to put in the A vs B box. 'None'
                            is excluded since it's a sentinel, not a polymer.

        Re-selection: stash the previous text per box and restore it after
        the rebuild, so a refresh triggered mid-session (browse / prune)
        doesn't silently snap the user back to 'Any'.
        """
        cur = self.db.conn
        one_comp = [r[0] for r in cur.execute(
            "SELECT DISTINCT p1_name FROM scans WHERE n_components=1 "
            "AND p1_name IS NOT NULL ORDER BY p1_name").fetchall()]
        two_comp = sorted({
            r[0] for r in cur.execute(
                "SELECT p1_name FROM scans WHERE n_components=2 "
                "AND p1_name IS NOT NULL AND p1_name != 'None'").fetchall()
        } | {
            r[0] for r in cur.execute(
                "SELECT p2_name FROM scans WHERE n_components=2 "
                "AND p2_name IS NOT NULL AND p2_name != 'None'").fetchall()
        })
        for cb, items in [(self.f_polymer, one_comp),
                          (self.f_polymerA, two_comp),
                          (self.f_polymerB, two_comp)]:
            prev = cb.currentText() if cb.count() else "Any"
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("Any")
            cb.addItems(items)
            idx = cb.findText(prev)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.blockSignals(False)

    def _build_where(self):
        clauses, params = [], []
        if self.f_solvent.currentText() != "Any":
            clauses.append("solvent=?"); params.append(self.f_solvent.currentText())
        cid = self.comp_group.checkedId()
        if cid == 1:
            clauses.append("n_components=1")
            poly = self.f_polymer.currentText()
            if poly != "Any":
                clauses.append("p1_name=?"); params.append(poly)
        elif cid == 2:
            clauses.append("n_components=2")
            if self.f_config.currentText() != "Any":
                clauses.append("config=?"); params.append(self.f_config.currentText())
            # Unordered 2-component pair match. Storage order (p1 vs p2) is
            # whatever the filename produced; the filter must match the SET,
            # not the sequence. Single-side selections match in either slot.
            a = self.f_polymerA.currentText()
            b = self.f_polymerB.currentText()
            a_set = a != "Any"
            b_set = b != "Any"
            if a_set and b_set:
                clauses.append(
                    "((p1_name=? AND p2_name=?) "
                    "OR (p1_name=? AND p2_name=?))")
                params.extend([a, b, b, a])
            elif a_set:
                clauses.append("(p1_name=? OR p2_name=?)")
                params.extend([a, a])
            elif b_set:
                clauses.append("(p1_name=? OR p2_name=?)")
                params.extend([b, b])
        sid = self.state_group.checkedId()
        if sid == 1:
            clauses.append("film_state='AP'")
        elif sid == 2:
            clauses.append("film_state='AN'")
            if self.f_temp.currentText() != "Any":
                clauses.append("anneal_temp=?"); params.append(int(self.f_temp.currentText()))
        return " AND ".join(clauses), tuple(params)

    def on_clear_filters(self):
        if not self._guard_pending():
            return
        self.f_solvent.setCurrentIndex(0)
        self.comp_group.button(0).setChecked(True)
        self.state_group.button(0).setChecked(True)
        self.f_config.setCurrentIndex(0)
        self.f_temp.setCurrentIndex(0)
        self.f_polymer.setCurrentIndex(0)
        self.f_polymerA.setCurrentIndex(0)
        self.f_polymerB.setCurrentIndex(0)
        self._toggle_conditional()
        self.refresh_table()

    def on_apply_filters(self):
        if not self._guard_pending():
            return
        self.refresh_table()

    def refresh_table(self):
        where, params = self._build_where()
        # Layer the active-view filter on top of the user's filter clauses.
        # "ALL" means no scoping; otherwise restrict to that batch_id. The
        # canonical key + dedup + prune logic still runs across the whole
        # DB -- only the displayed slice is scoped.
        if self._active_view and self._active_view != "ALL":
            if where:
                where = f"({where}) AND batch_id=?"
            else:
                where = "batch_id=?"
            params = params + (self._active_view,)
        self.current_rows = self.db.query(where, params)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.current_rows))
        for r, row in enumerate(self.current_rows):
            brush = self._brush_for_row(r)
            # Unparsed rows carry their parse error in a tooltip so the user
            # can hover any cell on the row and see the diagnostic without
            # opening the verification window.
            is_unparsed = row.get("review_status") == "unparsed"
            tip = (f"Parse error: {row['parse_error']}"
                   if is_unparsed and row.get("parse_error") else "")
            # Subtle row markers on the leftmost cell (csv_path), no color.
            # Color is reserved for review_status tinting -- these glyphs
            # stay orthogonal so promoted/edited/tinted can coexist.
            #   '*' (asterisk)  -> manually edited
            #   '^' (caret)     -> promoted to MongoDB Atlas
            # Both -> '^*'; neither -> no prefix.
            edited_mark = bool(row.get("edited"))
            promoted_mark = bool(row.get("promoted"))
            promoted_at = row.get("promoted_at") or ""
            for c, col in enumerate(VISIBLE_COLUMNS):
                val = "" if row[col] is None else str(row[col])
                if c == 0 and (edited_mark or promoted_mark):
                    prefix = ""
                    if promoted_mark:
                        prefix += "^"
                    if edited_mark:
                        prefix += "*"
                    val = prefix + " " + val
                item = QTableWidgetItem(val)
                # csv_path is ALWAYS locked. Unparsed rows are locked too --
                # they have no parsed data to edit; the user should rename
                # the file (outside the app) and re-browse. Otherwise lock
                # unless edit mode is on.
                if (col == "csv_path") or is_unparsed or (not self.edit_mode):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                # Review-status tint: green/red/peach/gray for confirmed/
                # rejected/needs_work/unparsed, default (empty brush) for
                # pending. Pending-edit yellow set in on_cell_edited still
                # overrides per cell during active editing.
                item.setBackground(brush)
                # Tooltip layering: parse error wins on unparsed rows (every
                # cell carries it). For parsed rows, the csv_path cell of a
                # promoted record shows the last push timestamp -- hover-only
                # surface for what the '^' glyph means.
                if tip:
                    item.setToolTip(tip)
                elif c == 0 and promoted_mark and promoted_at:
                    item.setToolTip(f"Promoted to cloud: {promoted_at}")
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self.log(f"Showing {len(self.current_rows)} scan(s).")

    def _brush_for_row(self, r: int) -> QBrush:
        """The status QBrush for row r, falling back to the default brush
        when r is out of range or review_status is missing/unknown.
        """
        if r < 0 or r >= len(self.current_rows):
            return QBrush()
        status = self.current_rows[r].get("review_status") or "pending"
        return self._STATUS_BRUSHES.get(status, QBrush())

    def on_cell_edited(self, item: QTableWidgetItem):
        """Stage the change in the in-memory buffer + tint the cell.

        Only fires when a cell is actually editable (edit mode on, not csv_path),
        so on_save_edits/on_save_and_exit is the only path that reaches SQLite.
        """
        row = item.row()
        col = VISIBLE_COLUMNS[item.column()]
        # Canonicalize defensively -- rows from the DB are already canonical,
        # but if anything ever bypasses that, this keeps the pending key in
        # sync with the row it must UPDATE.
        csv_path = canon_path(self.current_rows[row]["csv_path"])
        self.pending_edits[(csv_path, col)] = item.text()
        # Tint without retriggering itemChanged.
        self.table.blockSignals(True)
        try:
            item.setBackground(QBrush(self._PENDING_TINT))
        finally:
            self.table.blockSignals(False)

    # ------- edit-mode transitions ----------------------------------------
    def _apply_edit_flags(self):
        """Flip ItemIsEditable in place according to self.edit_mode.
        csv_path stays locked regardless. blockSignals so flag flips never
        masquerade as data edits.
        """
        self.table.blockSignals(True)
        try:
            for r in range(self.table.rowCount()):
                for c, col in enumerate(VISIBLE_COLUMNS):
                    item = self.table.item(r, c)
                    if item is None:
                        continue
                    locked = (not self.edit_mode) or (col == "csv_path")
                    flags = item.flags()
                    if locked:
                        flags &= ~Qt.ItemFlag.ItemIsEditable
                    else:
                        flags |= Qt.ItemFlag.ItemIsEditable
                    item.setFlags(flags)
        finally:
            self.table.blockSignals(False)

    def _enter_edit_mode(self):
        self.edit_mode = True
        self._apply_edit_flags()
        self.edit_btn.hide()
        self.edit_trio.show()
        self.staging_box.setTitle(self._STAGING_TITLE_EDITING)

    def _exit_edit_mode(self):
        self.edit_mode = False
        self._apply_edit_flags()
        self.edit_trio.hide()
        self.edit_btn.show()
        self.staging_box.setTitle(self._STAGING_TITLE_READONLY)

    def _clear_pending_tints(self):
        """Strip the yellow per-cell pending-edit tint and reapply each
        row's review-status brush. Called after a successful save -- the
        edit is committed, so the pending visual no longer applies, but
        the underlying status tint still does. blockSignals around it so
        background changes don't trigger itemChanged.
        """
        self.table.blockSignals(True)
        try:
            for r in range(self.table.rowCount()):
                brush = self._brush_for_row(r)
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    if item is not None:
                        item.setBackground(brush)
        finally:
            self.table.blockSignals(False)

    # ------- button handlers ----------------------------------------------
    def on_enter_edit(self):
        self._enter_edit_mode()
        self.log("Edit mode ON - cell changes stage in memory until you Save.")

    def on_save_edits(self) -> bool:
        """Commit pending_edits to SQLite in one transaction. Sets edited=1 on
        each affected row. On success: clears the buffer + tints, shows toast,
        logs, stays in edit mode. Returns True on success (including the
        zero-pending case), False on DB error.
        """
        count = len(self.pending_edits)
        if count == 0:
            msg = "No changes to save."
            self._show_toast(msg)
            self.log(msg)
            return True

        # Group edits by row so one UPDATE per row, all in a single transaction.
        by_path: dict[str, dict[str, str]] = {}
        for (csv_path, col), val in self.pending_edits.items():
            by_path.setdefault(csv_path, {})[col] = val

        try:
            cur = self.db.conn.cursor()
            cur.execute("BEGIN")
            for path, cols in by_path.items():
                set_parts = [f"{c}=?" for c in cols] + ["edited=1"]
                values = list(cols.values()) + [path]
                cur.execute(
                    f"UPDATE scans SET {', '.join(set_parts)} WHERE csv_path=?",
                    values)
            self.db.conn.commit()
        except Exception as e:
            self.db.conn.rollback()
            self.log(f"Save failed: {e}")
            self._show_toast(f"Save failed: {e}", success=False)
            return False

        # Mirror writes into current_rows so subsequent in-memory reads agree.
        for row in self.current_rows:
            p = row["csv_path"]
            if p in by_path:
                for c, v in by_path[p].items():
                    row[c] = v

        self.pending_edits.clear()
        self._clear_pending_tints()
        msg = f"Saved {count} change(s)."
        self._show_toast(msg)
        self.log(msg)
        return True

    def on_save_and_exit(self):
        if self.on_save_edits():
            self._exit_edit_mode()

    def on_cancel_edits(self):
        n = len(self.pending_edits)
        self.pending_edits.clear()
        self._exit_edit_mode()
        # Reload from DB to revert the displayed values (and naturally clear
        # any tints by rebuilding every QTableWidgetItem).
        self.refresh_table()
        if n:
            self.log(f"Discarded {n} change(s).")

    # ------- toast + unsaved-changes guard --------------------------------
    def _show_toast(self, text: str, success: bool = True):
        color = "#27ae60" if success else "#c0392b"
        self.toast.setStyleSheet(
            f"background:{color};color:white;padding:4px 10px;"
            f"border-radius:4px;font-weight:bold;")
        self.toast.setText(text)
        self.toast.show()
        # Each call schedules its own hide; later calls just push the deadline
        # out -- harmless if the user spams Save.
        QTimer.singleShot(2000, self.toast.hide)

    def _guard_pending(self) -> bool:
        """Return True if the caller may proceed past a destructive action
        (filter change, browse, etc.). If there are pending edits, prompt
        Save / Discard / Cancel-the-action and return accordingly.
        """
        if not self.pending_edits:
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText(
            f"You have {len(self.pending_edits)} unsaved change(s) in the "
            f"staging table.")
        box.setInformativeText(
            "Save them to the database, discard them, or cancel this action?")
        save_btn = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton("Cancel action", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save_btn:
            return self.on_save_edits()
        if clicked is discard_btn:
            self.pending_edits.clear()
            # Caller is about to rebuild the table (refresh_table / on_browse),
            # which will rebuild items without tints.
            return True
        return False

    def on_generate_plots(self):
        # Selection-driven: plot ONLY the currently selected staging-table
        # rows (Ctrl+A selects the whole filtered batch). Must stay on the
        # main thread -- originpro + COM is not reliably thread-safe.
        signals = [name for chk, name in
                   [(self.chk_cd, "CD"), (self.chk_g, "G-value"),
                    (self.chk_uv, "UV-Vis")] if chk.isChecked()]
        if not signals:
            self.log("Select at least one plot type.")
            return
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            self.log("Select one or more rows first (Ctrl+A selects all).")
            return
        # Sort by row index so the plot order mirrors what the user sees.
        indices = sorted(idx.row() for idx in sel)
        rows = [self.current_rows[r] for r in indices
                if 0 <= r < len(self.current_rows)]
        # Unparsed rows have no parsed signals to plot -- exclude them.
        # build_plots reads the actual CSV columns by index, so handing it
        # an unparsed file would either silently emit nothing or raise.
        files = [r["csv_path"] for r in rows
                 if r.get("review_status") != "unparsed"]
        skipped = len(rows) - len(files)
        if skipped:
            self.log(f"Skipping {skipped} unparsed/invalid selected row(s).")
        if not files:
            self.log("No parseable scans in the selection to plot.")
            return

        # Lazy import so the GUI still launches when originpro isn't installed.
        # Distinguish the graceful "originpro missing" case from any other
        # import / load failure -- previously every ImportError was lumped
        # under "is originpro installed?" which silently hid real bugs
        # (typos, renamed modules, syntax errors in plotting.py).
        try:
            from plotting import (
                build_plots, clear_quantities, quantities_for)
        except ImportError as e:
            if getattr(e, "name", None) == "originpro":
                self.log("OriginPro is not installed -- plotting unavailable.")
            else:
                self.log(
                    f"Failed to import plotting module: "
                    f"{type(e).__name__}: {e}")
                self.log(traceback.format_exc())
            return
        except Exception as e:
            # SyntaxError, NameError, AttributeError at module load, etc.
            self.log(
                f"Plotting module load failed: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Generating...")
        self.log(f"Generating {len(signals)} plot(s) for {len(files)} file(s)...")
        QApplication.processEvents()
        try:
            # Sync Origin to the checkboxes: nuke ALL THREE signal windows up
            # front, then rebuild only the checked ones. build_plots will
            # re-clear the ones it's about to draw, which is a harmless no-op.
            try:
                clear_quantities(
                    quantities_for(["CD", "G-value", "UV-Vis"]),
                    log=self.log)
            except Exception as e:
                self.log(
                    f"Pre-clear failed: {type(e).__name__}: {e}")
                self.log(traceback.format_exc())
            build_plots(files, quantities_for(signals), log=self.log)
            self.log("Done.")
        except Exception as e:
            self.log(
                f"Plot generation failed: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
        finally:
            self.generate_btn.setText("Plot Selected")
            self.generate_btn.setEnabled(True)

    def on_clear_signal(self, signal_label):
        """Clear one signal's Origin windows AND uncheck its checkbox.

        Intent: 'I don't want this plot.' The box unchecks even if Origin is
        unreachable, so the next Generate won't rebuild it.
        """
        chk_map = {"CD": self.chk_cd, "G-value": self.chk_g,
                   "UV-Vis": self.chk_uv}
        chk = chk_map.get(signal_label)
        if chk is not None:
            # Block signals so any future stateChanged handler doesn't fire as a
            # side effect of programmatic unchecking.
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)

        # Same distinguishing import as on_generate_plots: originpro-missing
        # is graceful, any other ImportError / load failure is logged loudly.
        try:
            from plotting import (
                clear_quantities, quantities_for)
        except ImportError as e:
            if getattr(e, "name", None) == "originpro":
                self.log("OriginPro is not installed -- nothing to clear.")
            else:
                self.log(
                    f"Failed to import plotting module: "
                    f"{type(e).__name__}: {e}")
                self.log(traceback.format_exc())
            return
        except Exception as e:
            self.log(
                f"Plotting module load failed: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            return

        try:
            clear_quantities(quantities_for([signal_label]), log=self.log)
            self.log(f"Cleared {signal_label} from Origin.")
        except Exception as e:
            self.log(
                f"Could not clear {signal_label} from Origin: "
                f"{type(e).__name__}: {e}")
            self.log(traceback.format_exc())
