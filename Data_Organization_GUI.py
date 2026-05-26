"""
CD Data Automation GUI
======================
Parses metadata from strictly-named CSV files, stores it in a SQLite database
(the source of truth), lets you filter via a cascading GUI, and dispatches the
selected scans to OriginPro for batch plotting.

Filename convention (underscore-separated):
    Series _ Poly1 _ Poly2 _ Ratio _ ConcSolvent _ State _ Speed [_ Temp if AN] _ RotIn _ RotOut

    R1_S-F8BT_C-PFBT100_50x50_20CB_AN_v0p005_T160_0_0
    R3_F8BT_None_100_20Tol_AP_v0p005_0_0

Stack: PyQt6, pandas, sqlite3 (stdlib), pywin32 (OriginPro COM, Windows only).
Run with uv:
    uv run python cd_gui.py
pyproject deps:  pyqt6  pandas  pywin32
"""

from __future__ import annotations

import os
import re
import sqlite3
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QRadioButton, QButtonGroup,
    QCheckBox, QTableWidget, QTableWidgetItem, QProgressBar, QTextEdit,
    QFileDialog, QGroupBox, QFrame, QHeaderView,
)

DB_PATH = "cd_metadata.db"
DEFAULT_ANNEAL_TIME = 10          # minutes; not stored in filename
WAVELENGTH_FLOOR = 300            # drop rows where wavelength <= this (data ends ~row 801)
SOLVENTS = ["CB", "DCB", "Tol"]   # controlled vocabulary


# ----------------------------------------------------------------------------
# 1. Filename parsing  ->  metadata
# ----------------------------------------------------------------------------

@dataclass
class Meta:
    csv_path: str
    series: str
    p1_name: str
    p1_backbone: str
    p1_chirality: str            # achiral | main-chain | side-chain | unknown
    p1_hand: Optional[str]       # R | S | None
    p1_pct: Optional[int]        # side-chain chiral %
    p2_name: str                 # "None" for single component
    p2_backbone: Optional[str]
    p2_chirality: Optional[str]
    p2_hand: Optional[str]
    p2_pct: Optional[int]
    n_components: int
    config: str                  # 1-comp | achiral+achiral | chiral+achiral | chiral+chiral
    ratio: str
    conc: int                    # mg/mL
    solvent: str
    film_state: str              # AP | AN
    speed_mm_s: float
    anneal_temp: Optional[int]
    anneal_time: Optional[int]   # default tag, not from filename
    rot_in: int
    rot_out: int


def classify_polymer(token: str):
    """Return (backbone, chirality, hand, pct) for a polymer token."""
    if token == "None":
        return (None, None, None, None)
    # main-chain chiral, e.g. R-F8BT / S-F8BT
    m = re.match(r"^([RS])-(.+)$", token)
    if m:
        return (m.group(2), "main-chain", m.group(1), None)
    # side-chain chiral, e.g. C-PFBT100 / C-PFBT50
    m = re.match(r"^C-([A-Za-z0-9]+?)(\d+)$", token)
    if m:
        return (m.group(1), "side-chain", None, int(m.group(2)))
    # bare achiral, e.g. F8BT
    return (token, "achiral", None, None)


def _derive_config(p1_chir, p2_chir, n_components):
    if n_components == 1:
        return "1-comp"
    def bucket(c):
        return "achiral" if c == "achiral" else "chiral"
    parts = sorted([bucket(p1_chir), bucket(p2_chir)])  # canonical order
    return f"{parts[0]}+{parts[1]}"


def parse_filename(path: str) -> Meta:
    """Parse a CSV path into Meta. Raises ValueError on malformed names."""
    stem = Path(path).stem
    f = stem.split("_")
    if len(f) < 9:
        raise ValueError(f"Too few fields ({len(f)}) in '{stem}'")

    # Rotation is always the last two tokens; parse remaining fields from front.
    rot_out = int(f.pop())
    rot_in = int(f.pop())

    series, p1, p2, ratio, conc_solv, state, speed = f[0:7]
    temp_tok = f[7] if len(f) > 7 else None

    # conc + solvent, e.g. 20CB
    m = re.match(r"^(\d+)([A-Za-z]+)$", conc_solv)
    if not m:
        raise ValueError(f"Bad conc/solvent token '{conc_solv}'")
    conc, solvent = int(m.group(1)), m.group(2)

    if state not in ("AP", "AN"):
        raise ValueError(f"Bad film state '{state}' (expected AP/AN)")

    # speed: v0p005 -> 0.005
    if not speed.startswith("v"):
        raise ValueError(f"Bad speed token '{speed}'")
    speed_val = float(speed[1:].replace("p", "."))

    anneal_temp = None
    if temp_tok is not None:
        if not temp_tok.startswith("T"):
            raise ValueError(f"Bad temp token '{temp_tok}'")
        anneal_temp = int(temp_tok[1:])
    if state == "AN" and anneal_temp is None:
        raise ValueError("Annealed film missing T### token")

    p1b, p1c, p1h, p1p = classify_polymer(p1)
    p2b, p2c, p2h, p2p = classify_polymer(p2)
    n = 1 if p2 == "None" else 2

    return Meta(
        csv_path=path, series=series,
        p1_name=p1, p1_backbone=p1b, p1_chirality=p1c, p1_hand=p1h, p1_pct=p1p,
        p2_name=p2, p2_backbone=p2b, p2_chirality=p2c, p2_hand=p2h, p2_pct=p2p,
        n_components=n, config=_derive_config(p1c, p2c, n),
        ratio=ratio, conc=conc, solvent=solvent, film_state=state,
        speed_mm_s=speed_val, anneal_temp=anneal_temp,
        anneal_time=(DEFAULT_ANNEAL_TIME if state == "AN" else None),
        rot_in=rot_in, rot_out=rot_out,
    )


# ----------------------------------------------------------------------------
# 2. SQLite layer  (source of truth)
# ----------------------------------------------------------------------------

COLUMNS = list(Meta.__annotations__.keys())


class DB:
    def __init__(self, path=DB_PATH):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        cols = ", ".join(f"{c} TEXT" if c in ("csv_path", "series", "p1_name",
            "p1_backbone", "p1_chirality", "p1_hand", "p2_name", "p2_backbone",
            "p2_chirality", "p2_hand", "config", "ratio", "solvent", "film_state")
            else f"{c} REAL" if c == "speed_mm_s"
            else f"{c} INTEGER" for c in COLUMNS)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS scans ({cols}, "
                          f"PRIMARY KEY(csv_path))")
        self.conn.commit()

    def upsert(self, m: Meta):
        d = asdict(m)
        placeholders = ", ".join("?" for _ in COLUMNS)
        self.conn.execute(
            f"INSERT OR REPLACE INTO scans ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders})", [d[c] for c in COLUMNS])
        self.conn.commit()

    def update_cell(self, csv_path: str, column: str, value):
        if column not in COLUMNS or column == "csv_path":
            return
        self.conn.execute(f"UPDATE scans SET {column}=? WHERE csv_path=?",
                          (value, csv_path))
        self.conn.commit()

    def query(self, where: str = "", params: tuple = ()):
        sql = "SELECT * FROM scans"
        if where:
            sql += " WHERE " + where
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def distinct(self, column: str):
        return [r[0] for r in self.conn.execute(
            f"SELECT DISTINCT {column} FROM scans ORDER BY {column}").fetchall()
            if r[0] is not None]


# ----------------------------------------------------------------------------
# 3. OriginPro plotting worker (runs in a QThread)
# ----------------------------------------------------------------------------

# Map logical signals to likely CSV column names (case-insensitive, first match).
COLUMN_ALIASES = {
    "wavelength": ["wavelength", "wl", "nm"],
    "cd": ["cd", "ellipticity", "mdeg"],
    "gvalue": ["gvalue", "g-value", "g_value", "gabs", "g"],
    "uvvis": ["uvvis", "uv-vis", "absorbance", "abs", "uv"],
}
PLOT_SIGNALS = {"CD": "cd", "G-value": "gvalue", "UV-Vis": "uvvis"}


def _find_col(df: pd.DataFrame, key: str):
    lower = {c.lower().strip(): c for c in df.columns}
    for alias in COLUMN_ALIASES[key]:
        if alias in lower:
            return lower[alias]
    return None


class OriginWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(self, rows: list[dict], signals: list[str]):
        super().__init__()
        self.rows = rows
        self.signals = signals  # e.g. ["CD", "G-value"]

    def run(self):
        try:
            import win32com.client
            self.log.emit("Connecting to OriginPro (Origin.ApplicationSI)...")
            origin = win32com.client.Dispatch("Origin.ApplicationSI")
            origin.Execute("doc -mc 1;")  # show Origin
        except Exception as e:
            self.log.emit(f"ERROR: could not connect to Origin: {e}")
            return

        # One master worksheet + graph per requested signal.
        books = {}
        for sig in self.signals:
            book = origin.CreatePage(2, f"{sig.replace('-', '')}_data", "Origin")
            books[sig] = {"book": book, "next_col": 0, "series": []}
            self.log.emit(f"Created workbook for {sig}: {book}")

        total = len(self.rows)
        for i, row in enumerate(self.rows):
            path = row["csv_path"]
            label = Path(path).stem
            try:
                df = pd.read_csv(path)
            except Exception as e:
                self.log.emit(f"  skip {label}: read error {e}")
                self.progress.emit(int((i + 1) / total * 100))
                continue

            wl_col = _find_col(df, "wavelength")
            if wl_col is None:
                self.log.emit(f"  skip {label}: no wavelength column")
                self.progress.emit(int((i + 1) / total * 100))
                continue

            # CRITICAL data cleaning: keep wavelength > 300 nm only.
            df = df[df[wl_col] > WAVELENGTH_FLOOR].reset_index(drop=True)

            for sig in self.signals:
                ycol = _find_col(df, PLOT_SIGNALS[sig])
                if ycol is None:
                    continue
                b = books[sig]
                ws = origin.FindWorksheet(b["book"])
                c0 = b["next_col"]
                # push X (wavelength) and Y (signal) as two new columns
                data = df[[wl_col, ycol]].to_numpy().tolist()
                origin.PutWorksheet(b["book"], data, 0, c0)
                # name the Y column so the legend is readable
                origin.Execute(
                    f'win -a {b["book"]}; wks.col{c0 + 2}.comment$ = "{label}";')
                b["series"].append((c0, c0 + 1))
                b["next_col"] += 2

            self.log.emit(f"  loaded {label}")
            self.progress.emit(int((i + 1) / total * 100))

        # Build one line graph per signal containing every series.
        for sig, b in books.items():
            if not b["series"]:
                continue
            gname = origin.CreatePage(3, f"{sig.replace('-', '')}_plot", "LINE")
            for (xc, yc) in b["series"]:
                origin.Execute(
                    f'win -a {gname}; plotxy iy:={b["book"]}!({xc + 1},{yc + 1}) '
                    f'plot:=200 ogl:=<active>;')
            self.log.emit(f"Built graph {gname} for {sig} "
                          f"({len(b['series'])} series)")

        self.log.emit("Done.")
        self.finished_ok.emit()


# ----------------------------------------------------------------------------
# 4. GUI
# ----------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CD Data Automation")
        self.resize(1100, 760)
        self.db = DB()
        self.worker = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        layout.addWidget(self._build_top_bar())
        layout.addWidget(self._build_staging_table(), stretch=3)
        mid = QHBoxLayout()
        mid.addWidget(self._build_filter_panel(), stretch=2)
        mid.addWidget(self._build_output_panel(), stretch=1)
        layout.addLayout(mid)
        layout.addWidget(self._build_execution_area(), stretch=2)

        self.refresh_table()
        self.refresh_filter_options()

    # --- top bar -----------------------------------------------------------
    def _build_top_bar(self):
        box = QGroupBox("Data Source")
        h = QHBoxLayout(box)
        self.path_field = QLineEdit()
        self.path_field.setReadOnly(True)
        self.path_field.setPlaceholderText("No folder selected")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.on_browse)
        self.origin_status = QLabel("  Origin: not connected  ")
        self.origin_status.setStyleSheet(
            "background:#c0392b;color:white;border-radius:4px;")
        connect = QPushButton("Connect to Origin")
        connect.clicked.connect(self.on_connect_origin)
        h.addWidget(QLabel("Folder:"))
        h.addWidget(self.path_field, stretch=1)
        h.addWidget(browse)
        h.addSpacing(20)
        h.addWidget(connect)
        h.addWidget(self.origin_status)
        return box

    # --- staging table -----------------------------------------------------
    def _build_staging_table(self):
        box = QGroupBox("Staging Area  (double-click any cell to correct a "
                        "parsed value; edits are saved to the database)")
        v = QVBoxLayout(box)
        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive)
        self.table.itemChanged.connect(self.on_cell_edited)
        v.addWidget(self.table)
        return box

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

        g.addWidget(QLabel("Film state:"), 3, 0)
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
        g.addWidget(sw, 3, 1)

        self.lbl_temp = QLabel("Anneal T (°C):")
        g.addWidget(self.lbl_temp, 4, 0)
        self.f_temp = QComboBox()
        g.addWidget(self.f_temp, 4, 1)

        apply_btn = QPushButton("Apply Filters")
        apply_btn.clicked.connect(self.refresh_table)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.on_clear_filters)
        g.addWidget(apply_btn, 5, 0)
        g.addWidget(clear_btn, 5, 1)

        self._toggle_conditional()
        return box

    def _toggle_conditional(self, *_):
        is_two = self.comp_group.checkedId() == 2
        self.lbl_config.setEnabled(is_two)
        self.f_config.setEnabled(is_two)
        is_annealed = self.state_group.checkedId() == 2
        self.lbl_temp.setEnabled(is_annealed)
        self.f_temp.setEnabled(is_annealed)

    # --- output panel ------------------------------------------------------
    def _build_output_panel(self):
        box = QGroupBox("Output Plots")
        v = QVBoxLayout(box)
        self.chk_cd = QCheckBox("Wavelength vs. CD"); self.chk_cd.setChecked(True)
        self.chk_g = QCheckBox("Wavelength vs. G-value")
        self.chk_uv = QCheckBox("Wavelength vs. UV-Vis")
        self.chk_mm = QCheckBox("Mueller Matrix  (coming soon)")
        self.chk_mm.setEnabled(False)
        for w in (self.chk_cd, self.chk_g, self.chk_uv, self.chk_mm):
            v.addWidget(w)
        v.addStretch(1)
        return box

    # --- execution area ----------------------------------------------------
    def _build_execution_area(self):
        box = QGroupBox("Execution")
        v = QVBoxLayout(box)
        self.run_btn = QPushButton("Run Processing")
        self.run_btn.clicked.connect(self.on_run)
        self.progress = QProgressBar()
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        v.addWidget(self.run_btn)
        v.addWidget(self.progress)
        v.addWidget(self.log_box)
        return box

    # ----- behavior --------------------------------------------------------
    def log(self, msg: str):
        self.log_box.append(msg)

    def on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select CSV folder")
        if not folder:
            return
        self.path_field.setText(folder)
        ok = err = 0
        for fn in os.listdir(folder):
            if not fn.lower().endswith(".csv"):
                continue
            full = os.path.join(folder, fn)
            try:
                self.db.upsert(parse_filename(full))
                ok += 1
            except Exception as e:
                err += 1
                self.log(f"  parse fail: {fn}  ->  {e}")
        self.log(f"Ingested {ok} file(s), {err} unparsed.")
        self.refresh_filter_options()
        self.refresh_table()

    def on_connect_origin(self):
        try:
            import win32com.client
            win32com.client.Dispatch("Origin.ApplicationSI")
            self.origin_status.setText("  Origin: connected  ")
            self.origin_status.setStyleSheet(
                "background:#27ae60;color:white;border-radius:4px;")
            self.log("OriginPro connection OK.")
        except Exception as e:
            self.origin_status.setText("  Origin: failed  ")
            self.origin_status.setStyleSheet(
                "background:#c0392b;color:white;border-radius:4px;")
            self.log(f"Origin connect failed: {e}")

    def refresh_filter_options(self):
        self.f_solvent.blockSignals(True)
        self.f_solvent.clear()
        self.f_solvent.addItem("Any")
        self.f_solvent.addItems(self.db.distinct("solvent"))
        self.f_solvent.blockSignals(False)
        self.f_temp.clear()
        self.f_temp.addItem("Any")
        self.f_temp.addItems([str(t) for t in self.db.distinct("anneal_temp")])

    def _build_where(self):
        clauses, params = [], []
        if self.f_solvent.currentText() != "Any":
            clauses.append("solvent=?"); params.append(self.f_solvent.currentText())
        cid = self.comp_group.checkedId()
        if cid == 1:
            clauses.append("n_components=1")
        elif cid == 2:
            clauses.append("n_components=2")
            if self.f_config.currentText() != "Any":
                clauses.append("config=?"); params.append(self.f_config.currentText())
        sid = self.state_group.checkedId()
        if sid == 1:
            clauses.append("film_state='AP'")
        elif sid == 2:
            clauses.append("film_state='AN'")
            if self.f_temp.currentText() != "Any":
                clauses.append("anneal_temp=?"); params.append(int(self.f_temp.currentText()))
        return " AND ".join(clauses), tuple(params)

    def on_clear_filters(self):
        self.f_solvent.setCurrentIndex(0)
        self.comp_group.button(0).setChecked(True)
        self.state_group.button(0).setChecked(True)
        self.f_config.setCurrentIndex(0)
        self.f_temp.setCurrentIndex(0)
        self._toggle_conditional()
        self.refresh_table()

    def refresh_table(self):
        where, params = self._build_where()
        self.current_rows = self.db.query(where, params)
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.current_rows))
        for r, row in enumerate(self.current_rows):
            for c, col in enumerate(COLUMNS):
                val = "" if row[col] is None else str(row[col])
                item = QTableWidgetItem(val)
                if col == "csv_path":
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)
        self.table.blockSignals(False)
        self.log(f"Showing {len(self.current_rows)} scan(s).")

    def on_cell_edited(self, item: QTableWidgetItem):
        row = item.row()
        col = COLUMNS[item.column()]
        csv_path = self.current_rows[row]["csv_path"]
        self.db.update_cell(csv_path, col, item.text())
        self.current_rows[row][col] = item.text()
        self.log(f"Updated {col} for {Path(csv_path).name}")

    def on_run(self):
        signals = [name for chk, name in
                   [(self.chk_cd, "CD"), (self.chk_g, "G-value"),
                    (self.chk_uv, "UV-Vis")] if chk.isChecked()]
        if not signals:
            self.log("Select at least one plot type.")
            return
        if not getattr(self, "current_rows", None):
            self.log("No scans match the current filters.")
            return
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.worker = OriginWorker(self.current_rows, signals)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished_ok.connect(lambda: self.run_btn.setEnabled(True))
        self.worker.start()


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()