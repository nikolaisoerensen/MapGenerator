#!/usr/bin/env python3
"""
Path: MapGenerator/main.py

MapGenerator - Hauptprogramm
"""

import sys
import os
from PyQt5.QtWidgets import QApplication

# Füge das Projektverzeichnis zum Python-Pfad hinzu
#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_menu import MainMenuWindow

def main():
    app = QApplication(sys.argv)

    app.setQuitOnLastWindowClosed(False)

    window = MainMenuWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()