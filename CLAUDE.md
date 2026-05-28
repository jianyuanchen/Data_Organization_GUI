# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Data Organization GUI** — a Python desktop application for the Diao Group that automates data organization tasks using OriginLab Origin as the backend. The GUI drives Origin via its Python API (`originpro`).

## Environment Setup

**Python version**: 3.14.5, managed via **uv**.

```powershell
# Create/activate the virtual environment
uv venv
.venv\Scripts\Activate.ps1

# Install dependencies
uv pip install originpro
```

The `.venv` is already present in the repository root. To run scripts directly without activating:

```powershell
.venv\Scripts\python.exe main.py
```

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `originpro` | 1.1.15 | Python API for OriginLab Origin (COM-based automation) |
| `OriginExt` | 1.2.5 | OriginLab Origin extension support |

**`originpro` requires OriginLab Origin to be installed on the machine.** The package connects to a running or launched Origin instance via COM. Import pattern:

```python
import originpro as op
```

Origin must be licensed and installed; `op.set_show(True/False)` controls whether the Origin window is visible during automation.

## Architecture

Flat module layout (no subpackages); imports flow `main` -> `main_window` -> `{database, parser, plotting}` -> `models`.

- **`main.py`** — entry point. Builds the QApplication, instantiates `MainWindow`, and runs the event loop.
- **`main_window.py`** — PyQt6 `MainWindow` and every GUI handler (filters, staging table, edit-mode state, Origin connect, plotting dispatch).
- **`database.py`** — SQLite layer: schema + forward-only migrations, dedup pass, record_id backfill, prune, and the three upsert flavors.
- **`parser.py`** — filename -> `Meta` parser.
- **`models.py`** — pure data layer: `Meta` dataclass, `classify_polymer`, `canon_path`, and column constants (`COLUMNS`, `VISIBLE_COLUMNS`). No file/DB/Qt deps.
- **`plotting.py`** — OriginPro automation. `build_plots`, `clear_quantities`, `quantities_for`, and the `Quantity` dataclass. Lazy-imported by `main_window` so the GUI launches without `originpro` installed.
- **`originpro` API** — all Origin interactions (reading/writing worksheets, graphs, projects) go through this library rather than direct COM calls.

## Running

```powershell
# Run the main application
.venv\Scripts\python.exe main.py

# Quick sanity check
.venv\Scripts\python.exe hello_world.py
```
