"""
Entry point. `python main.py` (or `uv run python main.py`) launches the GUI.
"""
from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    # Defer until after the first paint so the window appears instantly even if
    # the COM probe takes a moment. Origin is heavy; we must never boot it just
    # to open the GUI -- startup_origin_check uses launch=False.
    QTimer.singleShot(0, win.startup_origin_check)
    app.exec()


if __name__ == "__main__":
    main()
