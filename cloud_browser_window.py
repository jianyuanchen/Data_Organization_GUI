"""
Read-only cloud browser. Non-modal QDialog that fetches records from the
MongoDB Atlas collection and plots their EMBEDDED spectra arrays. This is the
download counterpart to the Phase 3 push (promote_records); it never edits or
deletes cloud records -- corrections happen by re-promoting a fixed local
record, so everything here is display-only.

Phase 3b-1: fetch + faithful plotting + basic filters (solvent / system /
film state). The three stacked, x-aligned subplots mirror the verification
window's plot pane, but the curves come straight from the document's
wavelength/cd/g/uv lists -- no CSV, no smoothing, no interpolation; the raw
arrays are drawn as plain lines. Richer cascading filters and peak write-back
are deliberately out of scope here.
"""
from __future__ import annotations

import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QDoubleValidator
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QGroupBox, QSplitter, QMessageBox,
    QSizePolicy, QScrollArea,
)

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from models import VISIBLE_COLUMNS


# Key of each signal's embedded array in the cloud document (set by
# promote_records: wavelength + g + cd + uv numeric lists).
_SIGNAL_DOC_KEY = {"CD": "cd", "g": "g", "UV": "uv"}
_SIGNAL_TITLES = {"CD": "CD (mdeg)", "g": "g-value", "UV": "UV-Vis (abs)"}
_SIGNAL_COLORS = {"CD": "tab:blue", "g": "tab:green", "UV": "tab:red"}
_SIGNAL_ORDER = ("CD", "g", "UV")          # top-to-bottom subplot order
_PLOT_X_RANGE = (300, 700)                  # full wavelength axis, always visible

# Light blue tint for every sidebar row -- a single calm color (no per-status
# palette like the verification sidebar) signals "trusted, read-only cloud
# view" rather than the editable staging vocabulary.
_CLOUD_ROW_TINT = QColor("#e7f1fb")


def _polymers(rec: dict) -> str:
    """'F8BT' for a 1-component record, 'F8BT+PFBT' for a 2-component one."""
    p1 = rec.get("p1_name") or "?"
    p2 = rec.get("p2_name")
    return p1 if (not p2 or p2 == "None") else f"{p1}+{p2}"


def _record_label(rec: dict, index: int) -> str:
    """One-line sidebar label: series + polymers + solvent + film state."""
    series = rec.get("series") or "?"
    solvent = rec.get("solvent") or "?"
    film = rec.get("film_state") or "?"
    return f"{index + 1:>3}. {series} | {_polymers(rec)} | {solvent} | {film}"


def _origin_label(rec: dict) -> str:
    """Legend identifier for the Origin overlay. Cloud records have no
    filename speed token, so we build one from series + polymers and fall
    back to record_id if both are missing.
    """
    series = rec.get("series") or ""
    polymers = _polymers(rec)
    ident = " ".join(x for x in (series, polymers) if x and x != "?").strip()
    return ident or str(rec.get("record_id") or "cloud-record")


# Wavelength floor for the Origin path. Mirrors the file pipeline, which keeps
# rows down to 300 nm (ROW_LIMIT) and drops the machine junk below; the graph
# x-range starts at 300 too. Embedded arrays promoted from CSVs are already
# cleaned this way, so the mask is a safe no-op for current data and a real
# clean for any future full-range document.
_WL_FLOOR_NM = 300


def _cleaned_arrays(rec: dict):
    """Return (wavelength, g, cd, uv) Python lists for the Origin path with
    sub-300 nm rows dropped, or None if the document has no usable spectra.

    Trims all four arrays to a common length first (defensive against a
    truncated document), then masks on wavelength >= 300. Order matches
    plotting.read_csv_columns: [wavelength, g, cd, uv].
    """
    wl = np.asarray(rec.get("wavelength") or [], dtype=float)
    g = np.asarray(rec.get("g") or [], dtype=float)
    cd = np.asarray(rec.get("cd") or [], dtype=float)
    uv = np.asarray(rec.get("uv") or [], dtype=float)
    n = min(len(wl), len(g), len(cd), len(uv))
    if n == 0:
        return None
    wl, g, cd, uv = wl[:n], g[:n], cd[:n], uv[:n]
    mask = wl >= _WL_FLOOR_NM
    if not mask.any():
        return None
    return (wl[mask].tolist(), g[mask].tolist(),
            cd[mask].tolist(), uv[mask].tolist())


class CloudBrowserWindow(QDialog):
    """Non-modal, read-only browser over the Atlas collection."""

    def __init__(self, log=print, parent=None):
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("Cloud Browser  (read-only)")
        self.resize(1280, 760)

        # Main window's log sink -- routed through so cloud fetch lines land
        # in the same output pane as everything else.
        self._log = log

        # Records currently displayed (the last fetch result). Each carries
        # its embedded wavelength/cd/g/uv arrays.
        self.records: list[dict] = []
        self.current_index: int = -1
        # Solvent values ever seen, so filtering by solvent never shrinks its
        # own dropdown. Grows across fetches; never removes options.
        self._known_solvents: set[str] = set()
        # Per-record latest computed peak: signal -> (wl, value, kind).
        self._computed_peaks: dict = {"CD": None, "g": None, "UV": None}
        self._band_artists: list = []
        self._marker_artists: dict = {}
        self._readout_artists: dict = {}
        self._active_signal = "g"
        self._has_plot_data = False

        self._build_ui()
        # Initial population: fetch everything so the solvent dropdown and the
        # list reflect what's actually in the cloud right now.
        self.refresh(initial=True)

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        root = QVBoxLayout(self)

        # Read-only banner so it's visually unmistakable this is the trusted
        # cloud view, not the editable local staging table.
        banner = QLabel(
            "Read-only cloud view — records come from MongoDB Atlas. "
            "To correct a record, fix it locally and re-promote.")
        banner.setStyleSheet(
            "background:#d1ecf1;color:#0c5460;padding:6px;border-radius:4px;")
        banner.setWordWrap(True)
        root.addWidget(banner)

        # ---- Filter row ----
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Solvent:"))
        self.f_solvent = QComboBox()
        self.f_solvent.addItem("Any")
        filt.addWidget(self.f_solvent)
        filt.addSpacing(10)
        filt.addWidget(QLabel("System:"))
        self.f_system = QComboBox()
        self.f_system.addItems(["Any", "1-Component", "2-Component"])
        filt.addWidget(self.f_system)
        filt.addSpacing(10)
        filt.addWidget(QLabel("Film state:"))
        self.f_film = QComboBox()
        self.f_film.addItems(["Any", "As Printed", "Annealed"])
        filt.addWidget(self.f_film)
        filt.addSpacing(10)
        self.refresh_btn = QPushButton("Refresh / Apply")
        self.refresh_btn.setToolTip(
            "Re-fetch cloud records using the current filter selections.")
        self.refresh_btn.clicked.connect(lambda: self.refresh())
        filt.addWidget(self.refresh_btn)
        filt.addStretch(1)
        self.count_label = QLabel("0 cloud records")
        self.count_label.setStyleSheet("font-weight:bold;")
        filt.addWidget(self.count_label)
        root.addLayout(filt)

        # ---- Splitter: sidebar | metadata | plot ----
        # Horizontal so the user can drag the dividers to reallocate width
        # between the records list, the metadata, and the plots.
        splitter = QSplitter(Qt.Orientation.Horizontal)

        sidebar_box = QGroupBox("Cloud records")
        sidebar_v = QVBoxLayout(sidebar_box)
        self.sidebar = QListWidget()
        # Multi-select (Ctrl/Shift-click) drives the "Plot Selected in Origin"
        # batch; a plain click still loads one record into the inline view.
        self.sidebar.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)
        # Small minimum so the list can be dragged narrow to give the plots
        # more room; without this the list's wide sizeHint hogs the splitter.
        self.sidebar.setMinimumWidth(150)
        self.sidebar.itemClicked.connect(self._on_sidebar_click)
        sidebar_v.addWidget(self.sidebar)
        # Wrap the tip so its full-width text doesn't pin a large minimum width
        # on the panel (which would stop the splitter from being dragged narrow).
        tip = QLabel(
            "Tip: Ctrl/Shift-click to select several records to plot in Origin.")
        tip.setWordWrap(True)
        sidebar_v.addWidget(tip)
        splitter.addWidget(sidebar_box)

        # ---- Metadata pane (read-only labels) ----
        # Short title so its width doesn't pin a large panel minimum (the
        # read-only nature is already stated in the banner above). This lets
        # the user drag the metadata narrow to widen the plots.
        meta_box = QGroupBox("Record metadata")
        meta_v = QVBoxLayout(meta_box)
        self.record_header = QLabel()
        self.record_header.setTextFormat(Qt.TextFormat.RichText)
        meta_v.addWidget(self.record_header)
        form_host = QWidget()
        form = QFormLayout(form_host)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Non-editable value labels, one per visible column. Selectable so the
        # user can copy a value, but never editable.
        self.field_labels: dict[str, QLabel] = {}
        for col in VISIBLE_COLUMNS:
            lab = QLabel("")
            lab.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            lab.setStyleSheet("color:#222;")
            self.field_labels[col] = lab
            form.addRow(QLabel(col + ":"), lab)
        # Host the form in a scroll area so the metadata panel can be dragged
        # narrow (to give the plots more room) without clipping -- a horizontal
        # scrollbar appears only when it's squeezed below the form's width.
        meta_scroll = QScrollArea()
        meta_scroll.setWidgetResizable(True)
        meta_scroll.setWidget(form_host)
        meta_scroll.setMinimumWidth(140)
        meta_v.addWidget(meta_scroll, stretch=1)
        splitter.addWidget(meta_box)

        # ---- Plot pane: 3 sharex'd subplots + range band + peak readout ----
        # Short title (see metadata note) so the panel can be dragged narrow;
        # the curves come from the records' embedded arrays.
        plot_box = QGroupBox("Per-record plots")
        plot_v = QVBoxLayout(plot_box)

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

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax_cd, self.ax_g, self.ax_uv = self.figure.subplots(
            3, 1, sharex=True)
        self._sig_axes = {"CD": self.ax_cd, "g": self.ax_g, "UV": self.ax_uv}
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        plot_v.addWidget(self.canvas, stretch=1)

        ctl_row = QHBoxLayout()
        ctl_row.addWidget(QLabel("Active:"))
        self.active_combo = QComboBox()
        self.active_combo.addItems(["CD", "g", "UV"])
        self.active_combo.setToolTip(
            "Which subplot Find Max/Min reads. Click a subplot to switch.")
        self.active_combo.currentTextChanged.connect(self._set_active_signal)
        ctl_row.addWidget(self.active_combo)
        ctl_row.addSpacing(10)
        self.find_max_btn = QPushButton("Find Max")
        self.find_max_btn.clicked.connect(lambda: self._find_extreme("max"))
        ctl_row.addWidget(self.find_max_btn)
        self.find_min_btn = QPushButton("Find Min")
        self.find_min_btn.clicked.connect(lambda: self._find_extreme("min"))
        ctl_row.addWidget(self.find_min_btn)
        ctl_row.addStretch(1)
        plot_v.addLayout(ctl_row)

        splitter.addWidget(plot_box)

        # Default split: give the plots the largest share so the CD/g/UV curves
        # open reasonably large, with a compact records list and metadata. The
        # user can drag any divider to override. Stretch factors make the plots
        # pane absorb the most width when the window is enlarged/maximized.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 4)
        splitter.setSizes([230, 300, 920])
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, stretch=1)

        # ---- Origin output row: signal checkboxes + Plot Selected button ----
        # Mirrors the main area's Output Plots. The inline matplotlib view
        # above is untouched -- Origin is an additional, publication-quality
        # output sourced from the same embedded arrays.
        origin_row = QHBoxLayout()
        origin_row.addWidget(QLabel("Origin output:"))
        self.chk_cd = QCheckBox("Wavelength vs CD")
        self.chk_cd.setChecked(True)                 # CD checked by default
        self.chk_g = QCheckBox("Wavelength vs g-value")
        self.chk_uv = QCheckBox("Wavelength vs UV-Vis")
        for chk in (self.chk_cd, self.chk_g, self.chk_uv):
            origin_row.addWidget(chk)
        origin_row.addSpacing(10)
        self.plot_origin_btn = QPushButton("Plot Selected in Origin")
        self.plot_origin_btn.setStyleSheet("font-weight:bold;")
        self.plot_origin_btn.setToolTip(
            "Plot the selected cloud record(s) in Origin: one consolidated "
            "overlay graph per checked signal, styled like the main area's "
            "Generate Plots but sourced from the embedded cloud arrays.")
        self.plot_origin_btn.clicked.connect(self.on_plot_in_origin)
        origin_row.addWidget(self.plot_origin_btn)
        origin_row.addStretch(1)
        root.addLayout(origin_row)

        bottom = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.on_prev)
        bottom.addWidget(self.prev_btn)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.on_next)
        bottom.addWidget(self.next_btn)
        bottom.addStretch(1)
        self.close_btn = QPushButton("Close")
        self.close_btn.setStyleSheet("padding:6px 14px;font-weight:bold;")
        self.close_btn.clicked.connect(self.accept)
        bottom.addWidget(self.close_btn)
        root.addLayout(bottom)

    # ----------------------------------------------------------- filters ----
    def _build_query(self) -> dict:
        """Translate the three filter dropdowns into a MongoDB filter dict.

        Returns {} for an all-Any selection (fetch everything). Built as a
        plain equality dict so fetch_records can extend it later with the
        richer cascading criteria without changing this contract.
        """
        query: dict = {}
        solvent = self.f_solvent.currentText()
        if solvent and solvent != "Any":
            query["solvent"] = solvent
        system = self.f_system.currentText()
        if system == "1-Component":
            query["n_components"] = 1
        elif system == "2-Component":
            query["n_components"] = 2
        film = self.f_film.currentText()
        if film == "As Printed":
            query["film_state"] = "AP"
        elif film == "Annealed":
            query["film_state"] = "AN"
        return query

    def refresh(self, initial: bool = False):
        """Re-fetch from the cloud with the current filters and rebuild the
        list. Shows a brief Loading... state; the network call runs on the
        main thread (it's quick and guarded), so processEvents keeps the UI
        from looking frozen while it blocks.
        """
        query = {} if initial else self._build_query()

        self.count_label.setText("Loading…")
        self.refresh_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            # Lazy import: keep the dialog importable even if pymongo / config
            # aren't available, so opening it can still show a clear message.
            from mongo_db import fetch_records
            records = fetch_records(query, log=self._log)
        except Exception as e:
            self._log(f"Cloud browser fetch error: {type(e).__name__}: {e}")
            records = []
        finally:
            self.refresh_btn.setEnabled(True)

        self.records = records
        self.current_index = 0 if records else -1

        # Grow the solvent dropdown from whatever solvents we've now seen, so
        # the option list never shrinks just because a filter narrowed the
        # current result set. Selection is preserved.
        for rec in records:
            s = rec.get("solvent")
            if s:
                self._known_solvents.add(str(s))
        self._rebuild_solvent_options()

        self._rebuild_sidebar()
        self.count_label.setText(f"{len(records)} cloud records")

        if records:
            self._load_record(0)
        else:
            self._show_empty_state()

    def _rebuild_solvent_options(self):
        """Repopulate the solvent dropdown from the accumulated known set,
        preserving the current selection.
        """
        prev = self.f_solvent.currentText()
        self.f_solvent.blockSignals(True)
        self.f_solvent.clear()
        self.f_solvent.addItem("Any")
        for s in sorted(self._known_solvents):
            self.f_solvent.addItem(s)
        idx = self.f_solvent.findText(prev)
        self.f_solvent.setCurrentIndex(idx if idx >= 0 else 0)
        self.f_solvent.blockSignals(False)

    # ----------------------------------------------------------- sidebar ----
    def _rebuild_sidebar(self):
        self.sidebar.clear()
        for i, rec in enumerate(self.records):
            item = QListWidgetItem(_record_label(rec, i))
            item.setBackground(QBrush(_CLOUD_ROW_TINT))
            self.sidebar.addItem(item)
        if 0 <= self.current_index < self.sidebar.count():
            self.sidebar.setCurrentRow(self.current_index)

    def _on_sidebar_click(self, item: QListWidgetItem):
        self._load_record(self.sidebar.row(item))

    # ------------------------------------------------------- record load ----
    def _show_empty_state(self):
        self.record_header.setText(
            "<b>No cloud records.</b> "
            "Cloud may be unconfigured/unreachable, or no records match the "
            "current filters — see the log.")
        for lab in self.field_labels.values():
            lab.setText("")
        for w in (self.prev_btn, self.next_btn):
            w.setEnabled(False)
        self._show_no_data("(no records)")

    def _load_record(self, index: int):
        if not (0 <= index < len(self.records)):
            return
        self.current_index = index
        rec = self.records[index]
        self.prev_btn.setEnabled(index > 0)
        self.next_btn.setEnabled(index < len(self.records) - 1)
        rid = rec.get("record_id") or "(no id)"
        self.record_header.setText(
            f"<b>Record {index + 1} of {len(self.records)}</b> "
            f"&nbsp;|&nbsp; record_id: <code>{rid}</code>")
        for col, lab in self.field_labels.items():
            val = rec.get(col)
            lab.setText("" if val is None else str(val))
        self.sidebar.setCurrentRow(index)
        self._refresh_plots()

    def on_prev(self):
        if self.current_index > 0:
            self._load_record(self.current_index - 1)

    def on_next(self):
        if self.current_index < len(self.records) - 1:
            self._load_record(self.current_index + 1)

    # --------------------------------------------------- Origin output ------
    def on_plot_in_origin(self):
        """Plot the selected cloud record(s) in Origin -- one consolidated
        overlay graph per checked signal, sourced from the embedded arrays.

        Read-only: nothing in the cloud (or the local DB) is touched. Runs on
        the main thread because originpro / COM is not reliably thread-safe.
        """
        rows = sorted(self.sidebar.row(it)
                      for it in self.sidebar.selectedItems())
        records = [self.records[r] for r in rows
                   if 0 <= r < len(self.records)]
        if not records:
            self._log("Select one or more cloud records first.")
            return
        signals = [name for chk, name in
                   [(self.chk_cd, "CD"), (self.chk_g, "G-value"),
                    (self.chk_uv, "UV-Vis")] if chk.isChecked()]
        if not signals:
            self._log("Select at least one plot type.")
            return

        # Lazy import, distinguishing the graceful "originpro missing" case
        # from any other load failure -- same pattern as the main window's
        # plotting handlers, so Origin being absent is a clear log line, not
        # a crash.
        try:
            from plotting import (
                PlotData, build_plots_from_data, clear_quantities,
                quantities_for)
        except ImportError as e:
            if getattr(e, "name", None) == "originpro":
                self._log("OriginPro is not installed -- plotting unavailable.")
            else:
                self._log(f"Failed to import plotting module: "
                          f"{type(e).__name__}: {e}")
                self._log(traceback.format_exc())
            return
        except Exception as e:
            self._log(f"Plotting module load failed: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
            return

        # Build in-memory data records: clean each document's arrays (drop
        # <300 nm) and label the curve by series + polymers.
        data = []
        for rec in records:
            label = _origin_label(rec)
            arrays = _cleaned_arrays(rec)
            if arrays is None:
                self._log(f"  SKIPPED (no embedded spectra): {label}")
                continue
            wl, g, cd, uv = arrays
            data.append(PlotData(
                label=label, wavelength=wl, g=g, cd=cd, uv=uv,
                log_name=label))
        if not data:
            self._log(
                "No selected cloud records have embedded spectra to plot.")
            return

        self.plot_origin_btn.setEnabled(False)
        old_text = self.plot_origin_btn.text()
        self.plot_origin_btn.setText("Generating…")
        self._log(f"Plotting {len(signals)} signal(s) for {len(data)} cloud "
                  f"record(s) in Origin…")
        QApplication.processEvents()
        try:
            # Clear all three signal windows first so unchecked ones don't
            # linger from a prior run, then build only the checked ones --
            # same sequence as the main area's Generate Plots.
            try:
                clear_quantities(
                    quantities_for(["CD", "G-value", "UV-Vis"]),
                    log=self._log)
            except Exception as e:
                self._log(f"Pre-clear failed: {type(e).__name__}: {e}")
                self._log(traceback.format_exc())
            build_plots_from_data(
                data, quantities_for(signals), log=self._log)
            self._log("Done.")
        except Exception as e:
            self._log(
                f"Origin plot generation failed: {type(e).__name__}: {e}")
            self._log(traceback.format_exc())
        finally:
            self.plot_origin_btn.setText(old_text)
            self.plot_origin_btn.setEnabled(True)

    # ------------------------------------------------------- plot pane ------
    def _signal_arrays(self, rec: dict, sig: str):
        """Return (wavelength, y) float arrays for one signal, or None if the
        document is missing/short on either array. No resampling -- the raw
        stored lists are used verbatim.
        """
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

    def _refresh_plots(self):
        self._computed_peaks = {"CD": None, "g": None, "UV": None}
        self._marker_artists = {}
        self._readout_artists = {}
        self._band_artists = []

        for ax in self._sig_axes.values():
            ax.clear()

        rec = self.records[self.current_index] if self.current_index >= 0 else None
        if not rec:
            self._show_no_data("(no record)")
            return

        plotted_any = False
        for sig in _SIGNAL_ORDER:
            ax = self._sig_axes[sig]
            arrays = self._signal_arrays(rec, sig)
            if arrays is None:
                continue
            wl, y = arrays
            # Plain line plot of the raw array -- no smoothing/interpolation.
            ax.plot(wl, y, color=_SIGNAL_COLORS[sig], linewidth=1.4)
            plotted_any = True

        if not plotted_any:
            self._show_no_data("(no embedded spectra in this record)")
            return

        self._has_plot_data = True
        self._set_peak_controls_enabled(True)
        self._draw_axes_chrome()
        self._draw_highlight_band()
        self.canvas.draw_idle()

    def _show_no_data(self, msg: str):
        self._has_plot_data = False
        self._set_peak_controls_enabled(False)
        for ax in self._sig_axes.values():
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, color="#888888",
                    style="italic", fontsize=10)
        self._draw_axes_chrome()
        self.canvas.draw_idle()

    def _draw_axes_chrome(self):
        self.ax_cd.set_ylabel("CD")
        self.ax_g.set_ylabel("g")
        self.ax_uv.set_ylabel("UV")
        self.ax_uv.set_xlabel("Wavelength (nm)")
        self.ax_cd.set_xlim(*_PLOT_X_RANGE)   # sharex propagates to g + uv
        for ax in self._sig_axes.values():
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        self._update_active_highlight()

    def _update_active_highlight(self):
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
        self.active_combo.blockSignals(True)
        self.active_combo.setCurrentText(sig)
        self.active_combo.blockSignals(False)
        self._update_active_highlight()

    def _set_peak_controls_enabled(self, enabled: bool):
        self.find_max_btn.setEnabled(enabled)
        self.find_min_btn.setEnabled(enabled)

    # --- highlight band ----------------------------------------------------
    def _on_range_changed(self, _text: str):
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
                ax.axvspan(lo, hi, alpha=0.18, color="#ffd24d", zorder=0))

    # --- find max / min within the band -----------------------------------
    def _find_extreme(self, kind: str):
        """Locate argmax/argmin of the active signal within the highlighted
        band and mark + read it out. Read-only: nothing is persisted (cloud
        records aren't editable here).
        """
        if not self._has_plot_data:
            return
        band = self._parse_range()
        if band is None:
            QMessageBox.information(
                self, "Set wavelength range first",
                "Enter Range min and Range max (in nm) before searching for "
                "a peak. Find Max / Find Min operate only within the band.")
            return
        lo, hi = band
        rec = self.records[self.current_index]
        arrays = self._signal_arrays(rec, self._active_signal)
        if arrays is None:
            QMessageBox.information(
                self, "No data",
                f"This record has no {self._active_signal} array.")
            return
        wl, y = arrays
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
        txt = f"{kind}: {val:.4g}, {wl:.0f} nm"
        handle = ax.text(
            0.98, 0.04, txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="#222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.85, edgecolor="#aaaaaa"))
        self._readout_artists[sig] = handle
