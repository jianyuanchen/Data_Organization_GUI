"""
OriginPro CD Scan Automation -- consolidated layout, with formatting.

For each quantity (g-value, CD, UV-Vis) builds one workbook with the scans
side by side  [ A(X)=Wavelength | B=scan1 | C=scan2 | ... ]  and overlays
every scan in a single grouped, styled graph. Print speed (parsed from each
filename) goes in the column's Comments row. Origin is left open at the end.

Run from VS Code:  pip install originpro
"""

import os
import re
import sys
import glob
import math
from dataclasses import dataclass, field
import originpro as op

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_DIR  = r"D:\OneDrive\Desktop\Diao_Group\Scripts\Origin_Graph_Python_Automation\test_data"            # folder of CSVs (relative to run dir, or a full path)
ROW_LIMIT = 801                     # keep rows down to 300 nm; drop machine junk below
PLOT_LINE = 200                     # LabTalk plot type id for a line
WAVELENGTH_LNAME = "wavelength (nm)"

# --- FORMAT: global defaults applied to every graph -------------------------
# Any key here can be overridden per-quantity (see QUANTITIES below).
FORMAT = {
    "line_width":     2500,         # internal set -w units (NOT points); 2500 ~ thick line
    "legend_border":  False,        # True = keep the box around the legend
    "frame":          "box",        # "box" = full frame, "L" = left + bottom only
    "font_size":      24,           # pt, axis titles + tick labels
    "x_range":        (300, 700),   # (from, to) in nm
    "x_major_ticks":  5,            # TOTAL number of major ticks on the X axis
    "x_minor_ticks":  1,            # number of small ticks between each pair of X major ticks
    "y_major_ticks":  5,            # TOTAL number of major ticks on the Y axis
    "y_minor_ticks":  1,            # number of small ticks between each pair of Y major ticks
    "y_symmetric":    True,         # symmetric about zero (-M, +M); False = 0..M
    "ticks_all_sides": False,       # False = ticks only on bottom + left (box still drawn)
}


@dataclass
class Quantity:
    src_col: int                    # 0-based column index in each CSV (A=0, B=1, C=2, D=3)
    col_lname: str                  # Long Name for the consolidated columns
    book_lname: str                 # workbook name
    graph_lname: str                # graph name
    overrides: dict = field(default_factory=dict)   # FORMAT keys to override for this graph
    wks: object = None              # consolidation worksheet (set at runtime)
    gp:  object = None              # overlay graph page    (set at runtime)
    data_absmax: float = 0.0        # largest |y| seen, for the symmetric Y range (runtime)


# Trim this list to just the g-value entry if that's all you need.
# `overrides` only needs the keys that differ from FORMAT; everything else is inherited.
# g-value & CD span +/- values  -> symmetric Y;  UV-Vis is one-sided  -> not symmetric.
QUANTITIES = [
    Quantity(1, "g-value (abs)", "g-value", "Master g-value",
             overrides={"y_symmetric": True}),
    Quantity(2, "CD (mdeg)",     "CD",      "Master CD",
             overrides={"y_symmetric": True}),
    Quantity(3, "UV-Vis (abs)",  "UV-Vis",  "Master UV-Vis",
             overrides={"y_symmetric": False}),
]


# Cleanly release Origin if the script crashes.
if op and op.oext:
    sys.excepthook = lambda *a: (op.exit(), sys.__excepthook__(*a))


# ===========================================================================
# HELPERS
# ===========================================================================
def nice_ceil(x: float) -> float:
    """Smallest 'nice' number (1, 2, 2.5, or 5 x 10^k) that is >= x.

    Gives clean tick labels: e.g. 0.178 -> 0.2, 7381 -> 10000, 0.308 -> 0.5.
    """
    if x <= 0:
        return 0.0
    base = 10.0 ** math.floor(math.log10(x))      # power-of-ten below x
    for mult in (1, 2, 2.5, 5, 10):
        if x <= mult * base * (1 + 1e-9):
            return mult * base
    return 10 * base


def parse_speed(stem: str) -> str:
    """'...50_50_0-005mms_160C...' -> '0.005mms'  (hyphen = decimal point)."""
    m = re.search(r"(\d+(?:-\d+)?)mms", stem, re.IGNORECASE)
    return m.group(1).replace("-", ".") + "mms" if m else ""


def read_csv_columns(path):
    """Return [wavelengths, g, CD, UV] as four lists, numeric rows only, capped at ROW_LIMIT."""
    rows = []
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            parts = line.replace("\t", ",").split(",")
            try:                                  # keep only real data rows;
                rows.append([float(parts[i]) for i in range(4)])
            except (ValueError, IndexError):      # header / trailing-junk lines fail here
                continue
            if len(rows) == ROW_LIMIT:
                break
    return [list(col) for col in zip(*rows)] if rows else []


def overlay(q: Quantity):
    """Plot every scan column, group for auto-coloring, rescale, label legend by speed."""
    layer = q.gp[0]
    for c in range(1, q.wks.cols):                # cols 1..N are the scans
        layer.add_plot(q.wks, c, 0, type=PLOT_LINE)
    layer.group()                                 # auto-assign a color per curve
    layer.rescale()
    op.lt_exec(f'win -a {q.gp.name}; legendupdate legend:="@LC";')


def style_graph(q: Quantity):
    """Apply the merged FORMAT (global + this quantity's overrides) to one graph."""
    fmt   = {**FORMAT, **q.overrides}             # overrides win on conflicting keys
    layer = q.gp[0]

    # --- axis ranges (reliable: originpro layer methods) --------------------
    x0, x1 = fmt["x_range"]
    x_step = (x1 - x0) / (fmt["x_major_ticks"] - 1)
    layer.set_xlim(x0, x1, x_step)

    # --- Y axis: nice round step, then snap the max TIGHT to the data -------
    # Pick a "nice" step (clean round labels) sized for ~n ticks, then set the
    # axis max to the smallest multiple of that step that still covers the data.
    # Snapping (rather than step*(n-1)) avoids extra empty headroom -- e.g. a
    # UV-Vis peak of 2.58 lands on a max of 3, not 4.
    n = fmt["y_major_ticks"]
    if fmt["y_symmetric"]:
        step = nice_ceil(q.data_absmax / ((n - 1) / 2))
        M = math.ceil(q.data_absmax / step) * step
        y0, y1 = -M, M
    else:
        step = nice_ceil(q.data_absmax / (n - 1))
        M = math.ceil(q.data_absmax / step) * step
        y0, y1 = 0, M
    layer.set_ylim(y0, y1, step)

    # --- ticks + frame + cosmetics (LabTalk, OriginPro 2026 property names) --
    # Scope every command to THIS graph's layer 1 via a range declaration, so it
    # doesn't depend on which window happens to be active. `gl` is that range;
    # `set ... -w` still needs the active window, so we activate it as well.
    g = q.gp.name
    op.lt_exec(f"win -a {g};")

    op.lt_exec(
        f"range gl = [{g}]1!;"
        # minor ticks: COUNT between each pair of major ticks
        f"gl.x.minorTicks = {fmt['x_minor_ticks']}; gl.y.minorTicks = {fmt['y_minor_ticks']};"
        # frame: showAxes 1 = bottom/left only, 3 = full box
        f"gl.x.showAxes = {3 if fmt['frame'] == 'box' else 1};"
        f"gl.y.showAxes = {3 if fmt['frame'] == 'box' else 1};"
        # tick-label font size, in points (layer.axis.label.pt)
        f"gl.x.label.pt = {fmt['font_size']}; gl.y.label.pt = {fmt['font_size']};"
        # axis-title font size: XB / YL are the bottom-X and left-Y title label
        # objects on the active layer; .fsize is their point size.
        f"xb.fsize = {fmt['font_size']}; yl.fsize = {fmt['font_size']};"
    )

    # ticks only on bottom + left: keep the box lines but clear top (x2) / right (y2)
    # tick marks. layer.<axis>.ticks bitmask: 0=none,1=major in,2=major out,4=minor in,8=minor out.
    if not fmt["ticks_all_sides"]:
        op.lt_exec(f"range gl = [{g}]1!; gl.x2.ticks = 0; gl.y2.ticks = 0;")

    # line width on every curve. set -w uses an internal scale (NOT points):
    # value 2500 was tested in the Script Window to give a thick line.
    # The proven command is `set %C -w 2500` (%C = active dataset). The earlier
    # `set %(name,c)` addressing did NOT take effect, so instead we make each
    # plot the active dataset (layer -i!N) and run the working `set %C` form.
    for c in range(1, q.wks.cols):
        op.lt_exec(f"win -a {g}; layer -i!{c}; set %C -w {fmt['line_width']};")

    # legend border: confirmed working in OriginPro 2026. The property is
    # `legend.background` (0 = None, 1 = Box, 2 = Shadow, 3 = Marble) -- NOT
    # .frame/.border. Activate the window in the same call so it targets this graph.
    op.lt_exec(f"win -a {g}; legend.background = {1 if fmt['legend_border'] else 0};")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    if op.oext:
        op.set_show(True)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        print(f"No .csv files found in: {os.path.abspath(DATA_DIR)}")
        return
    print(f"Found {len(files)} CSV file(s).")

    for q in QUANTITIES:                          # create the books + empty graphs
        q.wks = op.new_book("w", lname=q.book_lname)[0]
        q.gp  = op.new_graph(lname=q.graph_lname, template="line")

    col, wrote_x = 1, False                        # col = next free column; X written once
    for path in files:
        cols = read_csv_columns(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        if not cols:
            print(f"  SKIPPED (no numeric data): {stem}")
            continue

        speed = parse_speed(stem)
        for q in QUANTITIES:
            if not wrote_x:
                q.wks.from_list(0, cols[0], WAVELENGTH_LNAME, axis="X")
            ydata = cols[q.src_col]
            q.wks.from_list(col, ydata, q.col_lname, comments=speed, axis="Y")
            q.data_absmax = max(q.data_absmax, max(abs(v) for v in ydata))  # for symmetric Y

        wrote_x = True
        print(f"  added: {stem}  ({len(cols[0])} rows, {speed or 'n/a'})")
        col += 1

    for q in QUANTITIES:
        overlay(q)
        style_graph(q)

    print(f"\nDone. {col - 1} scan(s) consolidated. Origin left open for review.")


if __name__ == "__main__":
    main()