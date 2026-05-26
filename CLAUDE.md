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
.venv\Scripts\python.exe Data_Organization_GUI.py
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

The project is in early development. The intended architecture is:

- **`Data_Organization_GUI.py`** — main entry point; will contain the GUI application (likely `tkinter` or similar) that provides controls for selecting data sources, specifying organization rules, and triggering Origin automation.
- **`originpro` API** — all Origin interactions (reading/writing worksheets, graphs, projects) go through this library rather than direct COM calls.

## Running

```powershell
# Run the main application
.venv\Scripts\python.exe Data_Organization_GUI.py

# Quick sanity check
.venv\Scripts\python.exe hello_world.py
```
