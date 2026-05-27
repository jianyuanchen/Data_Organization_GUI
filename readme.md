# CD Data Automation GUI

A desktop application for organizing, filtering, and plotting circular dichroism (CD) spectroscopy data from meniscus-guided coated conjugated-polymer films. It parses experiment metadata directly from CSV filenames, stores it in a local database that serves as the single source of truth, lets you filter the dataset through a cascading interface, and dispatches the selected scans to OriginPro for automated, consistently-styled batch graphing.

**Stack:** Python · PyQt6 (GUI) · SQLite (metadata store) · OriginPro via the `originpro` package (plotting). Runs locally on Windows; managed with `uv`.

---

## Filename Convention

All metadata is encoded in the CSV filename, underscore-separated. Hyphens appear only inside polymer names.

```
Series _ Poly1 _ Poly2 _ Ratio _ ConcSolvent _ Speed _ State [_ Temp if AN] _ Gval _ Wavelength

annealed:    R1_C-PFBT100_S-F8BT_50x50_20CB_v0p005_AN_T160_gval=0p047_500nm
as-printed:  R3_F8BT_None_100_20Tol_v0p005_AP_gval=0p042_493nm
```

| Field | Meaning | Notes |
|---|---|---|
| Series | Experiment id (R1, R2…) | |
| Poly1 | First polymer | C-PFBT with chiral side-chain %: `C-PFBT100`, `C-PFBT50` |
| Poly2 | Second polymer, or `None` | `F8BT` (achiral) / `R-F8BT` / `S-F8BT` (main-chain chiral) |
| Ratio | Solution mass ratio | `50x50`, or `100` for single-component |
| ConcSolvent | Concentration + solvent | `20CB` = 20 mg/mL in chlorobenzene; also `DCB`, `Tol` |
| Speed | Blade speed (mm/s) | `v0p005` = 0.005 mm/s (`p` = decimal point) |
| State | Film state | `AP` (as printed) / `AN` (annealed) |
| Temp | Anneal temperature (°C) | Present only when `AN`, e.g. `T160` |
| Gval | Peak g-value | `gval=0p047` = 0.047 |
| Wavelength | Peak wavelength | `500nm` |

Anneal *time* is not encoded in the filename — it is stored as a database tag (default 10 min).

---

## Implemented Features

### Data ingestion & parsing
- **Filename parser** decodes every field above into structured metadata, deriving polymer backbone, chirality type (achiral / main-chain / side-chain), handedness (R/S), and side-chain percentage. The two-component configuration (achiral+achiral / chiral+achiral / chiral+chiral) is *derived* from the polymers rather than stored separately, so it can never fall out of sync.
- **Folder ingestion** parses every CSV in a chosen folder and writes the metadata to SQLite.

### Database as source of truth
- Parsed metadata lives in a local **SQLite** database (`cd_metadata.db`), keyed by file path.
- **Canonical path keying** normalizes every file path (slash style, case, relative vs. absolute) to a single form, so the same file can never produce duplicate rows regardless of how its path is spelled.
- **One-time de-duplication migration** runs on startup to collapse any pre-existing duplicate-path rows, preferring manually-edited versions.

### Staging table & editing
- All parsed metadata is shown in a spreadsheet-style **staging table**.
- **Read-only by default.** An **Edit** button unlocks editing; while editing it transforms into **Save**, **Cancel**, and **Save & Exit**.
- Edits are **staged in memory** (pending changes tinted yellow) and only written to the database on Save — nothing is altered by accident. Cancel discards; a confirmation guard prevents losing unsaved edits when changing filters or browsing.
- A transient green **toast notification** (plus a permanent log line) confirms each successful save.
- Manually-edited rows are flagged (`edited=1`) and **preserved across re-ingestion** — re-browsing a folder updates un-edited rows from their filenames but never overwrites your corrections. (The database is the source of truth for edited rows; filename and stored values may legitimately diverge.)

### Cleanup & sync
- **Orphan pruning** removes database rows whose underlying file no longer exists on disk — automatically after browsing, and on demand via a **Prune Missing** button (full-database sweep). Rows whose files still exist are never pruned.

### Filtering
- Cascading **filter panel**: solvent, system complexity (1- vs 2-component), two-component configuration, film state, and anneal temperature.
- **Specific-polymer filtering** (data-driven from the loaded dataset): pick exact polymers per system. Single-component selects one polymer; two-component offers two slots with **unordered matching** (a C-PFBT100 + R-F8BT pair matches regardless of slot order). Polymer and configuration filters combine as independent AND conditions.
- Conditional UI: filter controls show/hide based on the system selection so only relevant options appear.

### OriginPro plotting
- **Generate Plots** sends the currently-filtered scans to OriginPro, building one consolidated workbook and one styled overlay graph **per selected signal** (CD, g-value, UV-Vis), with all scans side-by-side and color-grouped. Print speed (from the filename) labels each curve in the legend.
- Data is cleaned before plotting (rows at/below 300 nm dropped).
- **Per-signal toggles**: each plot type has a checkbox plus an **x** button that removes that plot from Origin and unchecks it. Generate syncs to the checkboxes, so unchecked signals are cleared. Repeat runs **rebuild cleanly** — no accumulating duplicate workbooks or graphs.
- Consistent graph styling (axis ranges, symmetric vs. one-sided Y, tick counts, fonts, line widths, legend) applied automatically.

### Origin connection management
- **Connect to Origin** button attaches to Origin (launching it if needed), verifies the connection, and reports the version — with a color-coded status indicator.
- On startup the app **detects** an already-running Origin without launching one (Origin is heavy, so it only opens on explicit Connect or when plotting). Plotting works whether or not Connect was clicked.

### Robustness
- The GUI launches and runs even if OriginPro isn't installed (all Origin calls are lazily imported and guarded) — parsing, filtering, and editing never depend on Origin.

---

## Planned / Future Directions

### Near-term reliability hardening *(prioritized)*
- **Database transaction safety** *(highest priority)* — ensure all multi-row operations (Save, dedup, prune) are fully transactional and roll back cleanly on failure or DB lock, so the store can never be left in a half-written state.
- **Bad/partial CSV handling** — gracefully skip-and-log empty, in-progress, or malformed files (fewer than four columns, zero usable rows) without aborting a plot run.
- **Semantic validation** — flag (not reject) filenames that parse successfully but are physically inconsistent (e.g. a chiral+chiral label with no handedness/percentage), so bad metadata is caught before it enters analyses.

### Analysis features
- **Coverage heatmap** — a visual map of which polymer / solvent / condition combinations have already been tested vs. which remain unexplored, to guide future experiments toward maximal coverage. (Directly serves the original goal of identifying untried permutations.)
- **Mueller matrix** plotting (UI placeholder already present).

### Possible enhancements
- Visible indicator for manually-edited rows in the staging table.
- "Reset row" action to clear an edit and re-sync a row to its filename.
- Background-threaded plotting if Origin runs grow large enough that the brief UI freeze becomes noticeable (currently main-thread for COM stability).
- Linter cleanup (cosmetic type/style warnings).

---

## Notes for Testing
- Designed and validated against a 3-file test set; **next step is a larger real dataset** to stress-test parsing coverage, filtering, and plotting at scale.
- Database file: `cd_metadata.db` (delete to reset to a clean state).
- Run: `uv run python Data_Organization_GUI.py`
