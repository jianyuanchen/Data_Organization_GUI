"""
Manifest import window -- the verify-the-parse stage of the manifest ingest
path. Non-modal QDialog opened from main_window.

Shows one row per manifest entry with a colored status badge (ok / needs
review / error, plus grey orphan rows for CSVs present on disk but absent
from the manifest). Rows are editable inline; every edit re-runs
manifest.validate_row live and recolors the badge. Clicking a row shows its
full reasons list in the side panel.

METADATA ONLY: no spectral arrays are read here -- file checks are bare
os.path.exists. Arrays load exactly once, when Confirm hands the selected
rows to database.ingest_manifest_record, which stages them through the same
writer the regex path uses. The fix path for a badly-parsed manifest is
re-running Cowork and clicking Reload -- not heavy hand-editing here.
"""
from __future__ import annotations

import os
import re
import traceback

from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QListWidget, QMessageBox, QPushButton, QSplitter,
    QTableView, QVBoxLayout, QWidget,
)

import manifest
from config import ORIENTATION_SCAN_SUFFIXES, PROCESSED_FILE_MARKERS
from database import ingest_manifest_record
from models import canon_path

# Status badge fills. Same family as REVIEW_STATUS_COLORS but locally scoped:
# these are import-stage states (orphan isn't a review_status at all).
_BADGE = {
    "ok":           QColor("#d4edda"),   # green
    "needs_review": QColor("#ffd966"),   # amber
    "error":        QColor("#f8d7da"),   # red
    "orphan":       QColor("#d3d3d3"),   # grey
}


def _fmt(v) -> str:
    return "" if v is None else str(v)


def _is_expected_non_manifest_file(name: str) -> bool:
    """True for files that are SUPPOSED to be absent from the manifest, so the
    orphan check must never flag them: macOS AppleDouble sidecars, the binary
    .jws sources, and the raw per-orientation JASCO captures. The exclusion
    patterns live in config (shared with Cowork's manifest-generation logic)."""
    low = name.lower()
    if low.startswith("._"):                         # AppleDouble metadata
        return True
    if low.endswith(".jws"):                         # binary JASCO source
        return True
    stem = os.path.splitext(name)[0]
    return any(stem.endswith(suf) for suf in ORIENTATION_SCAN_SUFFIXES)


def _looks_processed(name: str) -> bool:
    """True if the filename carries a processed-spectrum marker (the computed
    g-value token). A CSV that looks processed but isn't in the manifest is a
    genuine miss; one that doesn't is incidental and left unflagged."""
    low = name.lower()
    return any(marker in low for marker in PROCESSED_FILE_MARKERS)


def _split_paren(text: str) -> tuple[str, str]:
    """'C-PFBT (100)' -> ('C-PFBT', '100'); no parens -> (text, '')."""
    m = re.match(r"^\s*(.*?)\s*\(\s*(.*?)\s*\)\s*$", text or "")
    if m:
        return m.group(1), m.group(2)
    return (text or "").strip(), ""


def _join_paren(a: str, b: str) -> str:
    a, b = _fmt(a).strip(), _fmt(b).strip()
    return f"{a} ({b})" if a and b else a


class _Entry:
    """One table row: a manifest row (raw dict + live validation state) or an
    orphan CSV. `raw` is the editable source of truth; status/reasons/parsed
    are recomputed from it by ImportWindow._revalidate.

    `dirty` tracks an in-memory edit not yet written back to manifest.csv;
    `staged` tracks an ingest into SQLite. The two are independent -- a row
    can be edited (dirty) then confirmed (staged) while the manifest on disk
    is still stale (dirty stays until an explicit Save to Manifest)."""

    def __init__(self, raw: dict, *, is_orphan: bool = False):
        self.raw = raw
        self.is_orphan = is_orphan
        self.status = "orphan" if is_orphan else "error"
        self.reasons: list[str] = (
            ["CSV present in folder but not referenced by any manifest row"]
            if is_orphan else [])
        self.parsed = None
        self.csv_path: str | None = None
        self.staged = False
        self.dirty = False


def _status_text(e: _Entry) -> str:
    """Status-cell label: badge word plus an unsaved-edit dot and a staged
    check when each applies."""
    if e.is_orphan:
        return "orphan"
    parts = [e.status]
    if e.dirty:
        parts.append("●")            # unsaved edit (not yet in manifest.csv)
    if e.staged:
        parts.append("✓ staged")
    return " ".join(parts)


# Display-column specs: (header, getter(entry) -> str,
#                        setter(raw, text) or None).
# Combined columns (poly+chir, conc/solvent, anneal, dopant+conc) edit via a
# light text syntax that the setter splits back into the underlying manifest
# fields; anything unparseable lands raw in the primary field so validate_row
# flags it instead of the edit being silently dropped.

def _set_poly(name_key: str, chir_key: str):
    def setter(raw, text):
        raw[name_key], raw[chir_key] = _split_paren(text)
    return setter


def _set_conc_solv(raw, text):
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*$", text or "")
    if m:
        raw["conc_mg_ml"], raw["solvent"] = m.group(1), m.group(2)
    else:
        raw["conc_mg_ml"] = (text or "").strip()


def _set_anneal(raw, text):
    nums = re.findall(r"\d+(?:\.\d+)?", text or "")
    raw["anneal_T_C"] = nums[0] if nums else ""
    raw["anneal_min"] = nums[1] if len(nums) > 1 else ""


def _set_dopant(raw, text):
    raw["dopant"], raw["dopant_conc_mg_ml"] = _split_paren(text)


def _set_field(key: str):
    def setter(raw, text):
        raw[key] = (text or "").strip()
    return setter


_COLUMNS = [
    ("status", _status_text, None),
    ("series", lambda e: _fmt(e.raw.get("series")), _set_field("series")),
    ("poly1 (+chir)",
     lambda e: _join_paren(e.raw.get("poly1"), e.raw.get("poly1_chir")),
     _set_poly("poly1", "poly1_chir")),
    ("poly2 (+chir)",
     lambda e: _join_paren(e.raw.get("poly2"), e.raw.get("poly2_chir")),
     _set_poly("poly2", "poly2_chir")),
    ("ratio", lambda e: _fmt(e.raw.get("ratio")), _set_field("ratio")),
    ("conc / solvent",
     lambda e: (f"{_fmt(e.raw.get('conc_mg_ml'))} "
                f"{_fmt(e.raw.get('solvent'))}").strip(),
     _set_conc_solv),
    ("speed (mm/s)", lambda e: _fmt(e.raw.get("speed_mm_s")),
     _set_field("speed_mm_s")),
    ("state", lambda e: _fmt(e.raw.get("state")), _set_field("state")),
    ("anneal (°C / min)",
     lambda e: " / ".join(x for x in (_fmt(e.raw.get("anneal_T_C")),
                                      _fmt(e.raw.get("anneal_min"))) if x),
     _set_anneal),
    ("dopant (+conc)",
     lambda e: _join_paren(e.raw.get("dopant"),
                           e.raw.get("dopant_conc_mg_ml")),
     _set_dopant),
    ("peak_gval", lambda e: _fmt(e.raw.get("peak_gval")),
     _set_field("peak_gval")),
    ("peak_wl", lambda e: _fmt(e.raw.get("peak_wl_nm")),
     _set_field("peak_wl_nm")),
    # filename is the JOIN KEY -- read-only (setter None) so it can never be
    # altered in the window and break the row<->CSV link.
    ("filename", lambda e: _fmt(e.raw.get("filename")), None),
]
_STATUS_COL = 0


class _ManifestModel(QAbstractTableModel):
    """Thin adapter over ImportWindow's entry list. Edits write back into
    entry.raw via the column setter, then the window revalidates the entry
    so the badge recolors live."""

    def __init__(self, window: "ImportWindow"):
        super().__init__(window)
        self.window = window

    # entries live on the window so Confirm/Reload logic owns them.
    @property
    def entries(self) -> list[_Entry]:
        return self.window.entries

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.entries)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(_COLUMNS)

    def headerData(self, section, orientation, role):
        if (orientation == Qt.Orientation.Horizontal
                and role == Qt.ItemDataRole.DisplayRole):
            return _COLUMNS[section][0]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        e = self.entries[index.row()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            return _COLUMNS[index.column()][1](e)
        if (role == Qt.ItemDataRole.BackgroundRole
                and index.column() == _STATUS_COL):
            color = _BADGE.get("orphan" if e.is_orphan else e.status)
            return QBrush(color) if color else None
        if role == Qt.ItemDataRole.ToolTipRole and e.reasons:
            return "\n".join(e.reasons)
        return None

    def flags(self, index):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not index.isValid():
            return base
        e = self.entries[index.row()]
        setter = _COLUMNS[index.column()][2]
        if setter is not None and not e.is_orphan:
            base |= Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        e = self.entries[index.row()]
        setter = _COLUMNS[index.column()][2]
        if setter is None or e.is_orphan:
            return False
        setter(e.raw, str(value))
        e.dirty = True            # in-memory edit not yet saved to manifest.csv
        e.staged = False          # edited after staging -> needs re-confirm
        self.window._revalidate(e)
        self.row_changed(index.row())
        return True

    def row_changed(self, row: int):
        self.dataChanged.emit(self.index(row, 0),
                              self.index(row, len(_COLUMNS) - 1))

    def refresh_all(self):
        """Repaint every cell in place (no model reset, so selection and the
        reasons panel survive). Used after a Save clears the dirty markers."""
        n = self.rowCount()
        if n:
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(n - 1, len(_COLUMNS) - 1))

    def reset(self):
        self.beginResetModel()
        self.endResetModel()


class _FlaggedProxy(QSortFilterProxyModel):
    """'Show only flagged' filter: needs_review + error rows (orphans stay
    visible too -- they are warnings by definition)."""

    only_flagged = False

    def filterAcceptsRow(self, source_row, source_parent):
        if not self.only_flagged:
            return True
        e = self.sourceModel().entries[source_row]
        return e.is_orphan or e.status in ("needs_review", "error")

    def set_only_flagged(self, on: bool):
        self.only_flagged = bool(on)
        self.invalidateFilter()


class ImportWindow(QDialog):
    """Manifest import + verify-the-parse window. Non-modal.

    `batch_id_factory` is main_window.new_batch_id, passed in so this module
    never imports main_window; each Confirm mints one batch covering exactly
    the rows confirmed together, mirroring one Browse. Emits `ingested`
    (the batch_id) after rows land in SQLite so the caller can refresh."""

    ingested = pyqtSignal(str)

    def __init__(self, db, *, batch_id_factory, log=print, parent=None):
        super().__init__(parent)
        self.db = db
        self.log = log
        self._batch_id_factory = batch_id_factory
        self.entries: list[_Entry] = []
        self._manifest_path: str | None = None
        self._base_folder: str | None = None
        self._suppressed = 0          # on-disk CSVs skipped by the orphan check

        self.setWindowTitle("Import Manifest")
        self.resize(1280, 640)
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)

        # --- top bar: source + reload + filter --------------------------
        top = QHBoxLayout()
        open_file = QPushButton("Open Manifest...")
        open_file.clicked.connect(self.on_open_manifest)
        open_folder = QPushButton("Open Folder...")
        open_folder.setToolTip(
            "Pick a folder containing manifest.csv plus the referenced "
            "data CSVs. (You can also drag-drop a folder or a manifest.csv "
            "anywhere onto this window.)")
        open_folder.clicked.connect(self.on_open_folder)
        self.reload_btn = QPushButton("Reload Manifest")
        self.reload_btn.setToolTip(
            "Re-read the manifest from disk. The fix path for bad parses is "
            "re-running Cowork and reloading -- not hand-editing every cell.")
        self.reload_btn.clicked.connect(self.on_reload)
        self.reload_btn.setEnabled(False)
        self.src_label = QLabel("No manifest loaded — drop one here.")
        self.src_label.setStyleSheet("color:#555;")
        self.flagged_chk = QCheckBox("Show only needs-review / error rows")
        self.flagged_chk.toggled.connect(self._on_flag_filter)
        top.addWidget(open_file)
        top.addWidget(open_folder)
        top.addWidget(self.reload_btn)
        top.addSpacing(12)
        top.addWidget(self.src_label, stretch=1)
        top.addWidget(self.flagged_chk)
        root.addLayout(top)

        # --- table + reasons side panel ----------------------------------
        self.model = _ManifestModel(self)
        self.proxy = _FlaggedProxy(self)
        self.proxy.setSourceModel(self.model)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        self.view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.view.horizontalHeader().setStretchLastSection(True)
        self.view.selectionModel().selectionChanged.connect(
            self._on_selection_changed)
        self.model.dataChanged.connect(
            lambda *_: self._after_model_change())

        side = QWidget()
        side_v = QVBoxLayout(side)
        side_v.setContentsMargins(0, 0, 0, 0)
        side_v.addWidget(QLabel("Reasons for the selected row:"))
        self.reasons_list = QListWidget()
        self.reasons_list.setWordWrap(True)
        side_v.addWidget(self.reasons_list, stretch=1)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.view)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([980, 280])
        root.addWidget(split, stretch=1)

        # --- bottom bar: counts + actions --------------------------------
        bottom = QHBoxLayout()
        self.counts_label = QLabel("")
        bottom.addWidget(self.counts_label, stretch=1)
        self.save_btn = QPushButton("Save to Manifest")
        self.save_btn.setToolTip(
            "Write the current rows back to the manifest.csv they were loaded "
            "from (atomic overwrite). Enabled only when there are unsaved "
            "edits. Note: re-running Cowork on the folder will regenerate the "
            "manifest and overwrite hand-edits.")
        self.save_btn.clicked.connect(self.on_save_manifest)
        self.save_btn.setEnabled(False)
        bottom.addWidget(self.save_btn)
        self.confirm_btn = QPushButton("Confirm Selected")
        self.confirm_btn.setStyleSheet("font-weight:bold;")
        self.confirm_btn.setToolTip(
            "Stage the selected rows into the local database (arrays load "
            "now). Enabled only while no selected row is an error/orphan; "
            "needs-review rows may be confirmed and carry their flag "
            "forward.")
        self.confirm_btn.clicked.connect(self.on_confirm_selected)
        self.confirm_btn.setEnabled(False)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bottom.addWidget(self.confirm_btn)
        bottom.addWidget(close_btn)
        root.addLayout(bottom)

    # ---- drag & drop ------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.load_path(path)
                return                      # first usable drop wins

    # ---- loading ------------------------------------------------------------
    def on_open_manifest(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select manifest.csv", "", "CSV files (*.csv)")
        if path:
            self.load_path(path)

    def on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder containing manifest.csv")
        if folder:
            self.load_path(folder)

    def on_reload(self):
        if self._manifest_path:
            self.load_path(self._manifest_path)

    def load_path(self, path: str):
        """Accept a manifest.csv or a folder containing one; (re)build the
        entry list. Every failure lands as a message, never a crash."""
        if os.path.isdir(path):
            candidate = os.path.join(path, "manifest.csv")
            if not os.path.exists(candidate):
                QMessageBox.warning(
                    self, "No manifest found",
                    f"The folder does not contain a manifest.csv:\n{path}")
                return
            self._base_folder = path
            self._manifest_path = candidate
        else:
            self._manifest_path = path
            self._base_folder = os.path.dirname(path)

        try:
            rows = manifest.load_manifest(self._manifest_path)
        except Exception as e:
            self.log(f"Manifest load failed: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            QMessageBox.warning(
                self, "Manifest load failed",
                f"Could not read the manifest:\n{type(e).__name__}: {e}")
            return

        self.entries = [_Entry(r) for r in rows]
        for e in self.entries:
            self._revalidate(e)
        self.entries.extend(self._find_orphans())

        self.model.reset()
        self.view.resizeColumnsToContents()
        self.reload_btn.setEnabled(True)
        self.src_label.setText(self._manifest_path)
        self.src_label.setStyleSheet("")
        self._after_model_change()
        orphan_n = sum(1 for e in self.entries if e.is_orphan)
        tail = f"; {orphan_n} orphan(s)" if orphan_n else ""
        if self._suppressed:
            tail += (f"; {self._suppressed} non-manifest file(s) ignored "
                     f"(raw/orientation/.jws)")
        self.log(f"Manifest loaded: {len(rows)} row(s) from "
                 f"{self._manifest_path}{tail}.")

    def _find_orphans(self) -> list[_Entry]:
        """Genuine orphans: CSVs on disk that LOOK like processed spectral
        files (carry a g-value marker) yet no manifest row references them --
        a real miss worth a human's attention.

        Scans the dropped/manifest folder plus every distinct existing
        source_folder. A candidate is flagged ONLY when it is not the manifest
        itself, not already referenced, NOT an expected-non-manifest file
        (AppleDouble / .jws / raw orientation scan -- suppressed via the shared
        config patterns), AND it positively looks processed. Files outside
        that intersection (raw scans, incidental CSVs) are silently ignored so
        the list isn't flooded. `self._suppressed` records how many on-disk
        CSVs were skipped, for the load log."""
        referenced = {
            os.path.basename((e.raw.get("filename") or "")).lower()
            for e in self.entries if not e.is_orphan}
        manifest_base = (os.path.basename(self._manifest_path).lower()
                         if self._manifest_path else "manifest.csv")
        folders = []
        if self._base_folder:
            folders.append(self._base_folder)
        for e in self.entries:
            sf = (e.raw.get("source_folder") or "").strip()
            if sf and os.path.isdir(sf):
                folders.append(sf)

        orphans, seen, suppressed = [], set(), 0
        for folder in folders:
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            for fn in names:
                if not fn.lower().endswith(".csv"):
                    continue
                if fn.lower() == manifest_base or fn.lower() in referenced:
                    continue
                key = canon_path(os.path.join(folder, fn))
                if key in seen:
                    continue
                seen.add(key)
                # Suppress expected non-manifest files; among the rest, only a
                # processed-looking CSV is a genuine miss. Everything else is
                # incidental and not flagged.
                if (_is_expected_non_manifest_file(fn)
                        or not _looks_processed(fn)):
                    suppressed += 1
                    continue
                entry = _Entry({"filename": fn}, is_orphan=True)
                entry.csv_path = os.path.join(folder, fn)
                orphans.append(entry)
        self._suppressed = suppressed
        return orphans

    # ---- live validation ----------------------------------------------------
    def _revalidate(self, e: _Entry):
        """Re-run metadata validation + the bare file-presence check for one
        entry. No arrays are read (os.path.exists only)."""
        if e.is_orphan:
            return
        status, reasons, parsed = manifest.validate_row(e.raw)
        e.status, e.reasons, e.parsed = status, list(reasons), parsed

        e.csv_path = None
        fn = (e.raw.get("filename") or "").strip()
        if fn:
            candidates = []
            sf = (e.raw.get("source_folder") or "").strip()
            if sf:
                candidates.append(os.path.join(sf, fn))
            if self._base_folder:
                candidates.append(os.path.join(self._base_folder, fn))
            existing = next(
                (c for c in candidates if self._exists(c)), None)
            e.csv_path = existing or (candidates[0] if candidates else None)
            if existing is None:
                e.status = "error"
                e.reasons.append(f"missing file: {fn}")

    @staticmethod
    def _exists(path: str) -> bool:
        try:
            return os.path.exists(path)
        except OSError:
            return False

    # ---- UI state -----------------------------------------------------------
    def _on_flag_filter(self, on: bool):
        self.proxy.set_only_flagged(on)
        self._sync_confirm_enabled()

    def _after_model_change(self):
        self._update_counts()
        self._sync_confirm_enabled()
        self._sync_dirty_state()
        self._show_reasons_for_current()
        # An edit can move a row in/out of the flagged-only filter.
        if self.proxy.only_flagged:
            self.proxy.invalidateFilter()

    def _any_dirty(self) -> bool:
        return any(e.dirty for e in self.entries if not e.is_orphan)

    def _sync_dirty_state(self):
        """Reflect unsaved-edit state in the UI: enable Save only when there
        are unsaved edits (and a manifest to write to), and mark the window
        title."""
        dirty = self._any_dirty()
        self.save_btn.setEnabled(dirty and self._manifest_path is not None)
        self.setWindowTitle(
            "Import Manifest — unsaved edits *" if dirty
            else "Import Manifest")

    def _on_selection_changed(self, *_):
        self._sync_confirm_enabled()
        self._show_reasons_for_current()

    def _selected_entries(self) -> list[_Entry]:
        rows = self.view.selectionModel().selectedRows()
        out = []
        for idx in sorted(rows, key=lambda i: i.row()):
            src = self.proxy.mapToSource(idx)
            if 0 <= src.row() < len(self.entries):
                out.append(self.entries[src.row()])
        return out

    def _sync_confirm_enabled(self):
        sel = self._selected_entries()
        ok = bool(sel) and all(
            (not e.is_orphan) and e.status in ("ok", "needs_review")
            for e in sel)
        self.confirm_btn.setEnabled(ok)

    def _show_reasons_for_current(self):
        self.reasons_list.clear()
        idx = self.view.selectionModel().currentIndex()
        if not idx.isValid():
            return
        src = self.proxy.mapToSource(idx)
        if not (0 <= src.row() < len(self.entries)):
            return
        e = self.entries[src.row()]
        if e.reasons:
            self.reasons_list.addItems(e.reasons)
        else:
            self.reasons_list.addItem("(no issues)")

    def _update_counts(self):
        rows = [e for e in self.entries if not e.is_orphan]
        orphans = len(self.entries) - len(rows)
        ok = sum(1 for e in rows if e.status == "ok")
        review = sum(1 for e in rows if e.status == "needs_review")
        error = sum(1 for e in rows if e.status == "error")
        staged = sum(1 for e in rows if e.staged)
        text = (f"{len(rows)} manifest row(s): {ok} ok, {review} needs "
                f"review, {error} error")
        if orphans:
            text += f"  ·  {orphans} orphan CSV(s)"
        if staged:
            text += f"  ·  {staged} staged"
        self.counts_label.setText(text)

    # ---- Confirm -> ingest ----------------------------------------------------
    def on_confirm_selected(self):
        """Hand the selected parsed rows to the SQLite ingest step. This is
        where arrays load and the machine cross-check runs (database.
        ingest_manifest_record); needs_review rows carry their flags
        forward. One batch_id covers everything confirmed together."""
        entries = self._selected_entries()
        confirmable = [e for e in entries
                       if not e.is_orphan
                       and e.status in ("ok", "needs_review")]
        if not confirmable or len(confirmable) != len(entries):
            self.log("Confirm blocked: selection contains error/orphan rows.")
            return

        batch_id = self._batch_id_factory()
        new = updated = preserved = flagged = failed = 0
        for e in confirmable:
            try:
                meta = manifest.build_metadata(e.parsed, csv_path=e.csv_path)
                carried = (list(e.reasons)
                           if e.status == "needs_review" else [])
                res = ingest_manifest_record(
                    self.db, meta, manifest_row=e.parsed,
                    carried_flags=carried, batch_id=batch_id, log=self.log)
            except Exception as ex:
                res = {"ok": False,
                       "error": f"{type(ex).__name__}: {ex}"}
            if not res.get("ok"):
                failed += 1
                e.status = "error"
                e.reasons.append(f"ingest failed: {res.get('error')}")
                self.log(f"  ingest failed: {e.raw.get('filename')} -> "
                         f"{res.get('error')}")
                continue
            e.staged = True
            if res.get("reasons"):
                flagged += 1
            result = res.get("result")
            if result == "new":
                new += 1
            elif result == "updated":
                updated += 1
            else:
                preserved += 1

        self.model.reset()
        self._after_model_change()
        bits = [f"{new} new", f"{updated} updated",
                f"{preserved} preserved (manually edited)"]
        if flagged:
            bits.append(f"{flagged} flagged needs_review")
        if failed:
            bits.append(f"{failed} failed")
        self.log(f"Manifest ingest (batch {batch_id}): " + ", ".join(bits)
                 + ".")
        if new + updated + preserved:
            self.ingested.emit(batch_id)

    # ---- Save edits back to manifest.csv --------------------------------------
    def on_save_manifest(self):
        """Atomically write the current manifest rows back to the source
        manifest.csv (explicit Save -- never autosave). Only the manifest
        rows are written (orphans are not manifest entries); column order
        follows config.MANIFEST_COLUMNS. On success, dirty flags clear."""
        if not self._manifest_path or not self._any_dirty():
            return
        rows = [e.raw for e in self.entries if not e.is_orphan]
        try:
            manifest.save_manifest(self._manifest_path, rows)
        except Exception as e:
            self.log(f"Save to manifest failed: {type(e).__name__}: {e}")
            self.log(traceback.format_exc())
            QMessageBox.warning(
                self, "Save failed",
                f"Could not write the manifest:\n{type(e).__name__}: {e}\n\n"
                f"The original manifest.csv is unchanged.")
            return
        for e in self.entries:
            e.dirty = False
        self.model.refresh_all()        # drop the ● markers, keep selection
        self._sync_dirty_state()
        self.log(f"Saved {len(rows)} row(s) to {self._manifest_path}.")
