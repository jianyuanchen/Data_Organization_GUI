# CD Data Automation GUI

A desktop application for organizing, filtering, and plotting circular dichroism (CD) spectroscopy data from meniscus-guided coated conjugated-polymer films. It parses experiment metadata directly from CSV filenames, stores it in a local database that serves as the single source of truth, lets you filter the dataset through a cascading interface, and dispatches the selected scans to OriginPro for automated, consistently-styled batch graphing.

**Stack:** Python · PyQt6 (GUI) · SQLite (metadata store) · OriginPro via the `originpro` package (plotting). Runs locally on Windows; managed with `uv`.

---

## Requirements

- **Windows only.** The plotting backend talks to OriginPro through COM via `pywin32`; neither is available on macOS or Linux.
- **OriginPro must be installed locally** for plotting to work. The GUI itself launches without OriginPro — parsing, filtering, the staging table, and editing all work — but **Generate Plots** requires a licensed local OriginPro install.
- **Python 3.12 or newer.** Developed against 3.14.5; 3.12 / 3.13 also work. (All four dependencies have wheels for 3.12–3.14.)
- **[uv](https://docs.astral.sh/uv/)** for environment management and dependency installation. See the uv site for install instructions on Windows (`winget install --id=astral-sh.uv` or the PowerShell installer linked there).

---

## Installation

```powershell
# 1. Clone the repository
git clone <repo-url>
cd Data_Organization_GUI

# 2. Create the virtual environment and install all dependencies from pyproject.toml
uv sync
```

`uv sync` reads `pyproject.toml`, creates a `.venv/` in the project folder, and installs `pyqt6`, `pandas`, `pywin32`, and `originpro` into it. No manual `pip install` step is needed.

If `uv sync` fails on `originpro` or `pywin32`, you're almost certainly not on Windows — see the Requirements section above.

---

## Running

```powershell
uv run python main.py
```

`uv run` activates the project's virtual environment for that single command, so you don't need to activate it yourself. The entry point is `main.py`.

---

## First Run

- The app opens with **no folder loaded** — the staging table is empty.
- Click **Browse...** to point at a folder of correctly-named CSVs (see the Filename Convention below). Every parseable file is ingested into the local SQLite database (`cd_metadata.db`, created automatically in the project folder).
- The **Origin: not connected** indicator is normal at startup — OriginPro is heavy, so the app does not launch it until you click **Connect to Origin** or **Generate Plots**. Parsing, filtering, and editing all work whether or not Origin is connected.

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

## How the Cloud Database Works

The app uses two storage tiers that work together:

- **Local (your machine).** A SQLite file (`cd_metadata.db`) sitting in the project folder. This is your personal working copy: everything you browse, edit, and review lives here. It's private to your machine and never shared automatically.
- **Cloud (the lab's shared store).** A MongoDB Atlas database that holds the lab-wide, verified dataset. Atlas is a managed cloud database service — you don't run a server yourself, you just connect to it over the internet.

The flow is one-way and deliberate. You parse files, review them in the verification window, and **Confirm** the ones you're sure about — all locally. Then, when you're ready, you click **Promote Selected to Cloud** or **Promote Batch to Cloud** to upload only the confirmed records to Atlas. Records that are still pending, rejected, marked needs-work, or unparsed are never uploaded. The cloud is the *trusted store*; anything that lands there has been reviewed by a human.

Each cloud record is **self-contained**. It holds all the metadata (polymer, solvent, ratio, etc.), the embedded spectral arrays (wavelength / CD / g-value / UV-Vis as numeric lists), provenance fields (who added it, when it was verified, when it was promoted), and a stable unique `record_id`. Because the spectra are embedded directly in the document, anyone with database access can retrieve a record and plot it without needing the original CSV file.

Re-uploading the same record does not create a duplicate. The upload is keyed on `record_id`, so promoting a record a second time **updates** the existing cloud document instead of inserting a new one — useful when you correct a peak locally and want the cloud copy refreshed.

If the cloud is unreachable or not configured yet, the app still runs fully: you can parse, review, confirm, edit, and plot locally without ever connecting to Atlas. Cloud actions just report a clear error in the log instead of working.

---

## Getting Database Access (for lab members)

The lab admin (the project maintainer) controls who can read and write to the shared database. New lab members get a personal database user from the admin — you do **not** sign up for Atlas yourself, and you do **not** get Atlas console / admin access. Once the admin issues your credentials, you connect by pasting a connection string into a local `.env` file.

### A. For the admin — creating a user for a lab member

Done once per lab member, all in the MongoDB Atlas web console:

1. In Atlas, go to **Security → Database Access** and click **Add New Database User**.
2. Set a username and password for that member. Record the password securely (a password manager, an encrypted vault) and share it with the member through a secure channel — a password-manager share link, a 1Password / Bitwarden item, or in person. **Never** email, Slack, or chat the password in plaintext.
3. Under **Database User Privileges**, assign the built-in role **"Read and write to any database"** — *not* `atlasAdmin`. This lets the member upload (insert), read (download), and update records, which is everything the app needs. It does **not** let them manage other users, change cluster or database settings, drop the database, delete data wholesale, or see billing / admin info — those stay with the admin only.
4. In Atlas, go to **Security → Network Access** and add the member's public IP address to the IP Access List (or follow the lab's network policy, e.g. a campus subnet entry). Without this, their machine cannot reach the cluster regardless of credentials.

Then hand the member: their username, their password (through a secure channel), and the connection-string template — with `<username>`, `<password>`, `<cluster>`, and `<app>` left as placeholders. They'll fill the credentials in themselves.

### B. For the lab member — connecting once you have credentials

1. Install the project as described in the **Installation** section above (install `uv`, clone the repo, run `uv sync`).
2. In the project folder, copy `.env.example` to a new file named `.env`:

   ```powershell
   Copy-Item .env.example .env
   ```

   `.env` is gitignored and will never be committed. The `.env.example` file is just a template with placeholders — safe to commit, no secrets in it.

3. Open `.env` in any text editor and fill in `MONGODB_URI` with the connection string the admin gave you, replacing `<username>` and `<password>` with your own credentials. The format is:

   ```
   MONGODB_URI="mongodb+srv://<username>:<password>@<cluster>/?appName=<app>"
   MONGODB_DB="cd_automation"
   MONGODB_COLLECTION="samples"
   ```

   Leave `MONGODB_DB` and `MONGODB_COLLECTION` as they are unless the admin tells you otherwise.

4. Launch the app (`uv run python main.py`) and click **Test Cloud Connection** in the top bar. A green **Cloud: connected** indicator with a `Connected to cd_automation.samples` log line means you're set. A red indicator means something is wrong — most often a typo in the password, an IP that hasn't been allowlisted, or a network issue. Share the log line with the admin if you can't figure it out.

### A note on the connection string

The `MONGODB_URI` value contains a live password. Treat `.env` like a private key:

- Keep it on your machine only. Never commit it, never paste it into a chat or email, never share your screen with it open.
- If you suspect the password has been exposed in any way — a screenshot, an accidental paste, a shared screen — tell the admin immediately so they can rotate it. A rotated password is the only safe response to exposure; "it was only visible for a second" is not.

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
- Run: `uv run python main.py`
