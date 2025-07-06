#!/usr/bin/env python3
"""
Path: MapGenerator/gui/navigation_widget.py
__init__.py existiert in "gui"

Navigation Widget für einheitliche Navigation zwischen Tabs
"""

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy, QVBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal


class NavigationWidget(QWidget):
    """Einheitliches Navigation Widget für alle Tabs"""

    # Signale für Navigation
    prev_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    quick_gen_clicked = pyqtSignal()
    exit_clicked = pyqtSignal()

    def __init__(self, show_prev=True, show_next=True, prev_text="Zurück", next_text="Weiter"):
        super().__init__()
        self.init_ui(show_prev, show_next, prev_text, next_text)

    def init_ui(self, show_prev, show_next, prev_text, next_text):
        layout = QVBoxLayout()

        # Navigation Buttons (Zurück/Weiter)
        nav_layout = QHBoxLayout()
        nav_widget = QWidget()

        if show_prev:
            self.prev_btn = QPushButton(prev_text)
            self.prev_btn.setStyleSheet(
                "QPushButton { background-color: #9E9E9E; color: white; font-weight: bold; padding: 10px; }")
            self.prev_btn.clicked.connect(self.prev_clicked.emit)
            nav_layout.addWidget(self.prev_btn)

        if show_next:
            self.next_btn = QPushButton(next_text)
            self.next_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
            self.next_btn.clicked.connect(self.next_clicked.emit)
            nav_layout.addWidget(self.next_btn)

        nav_widget.setLayout(nav_layout)
        layout.addWidget(nav_widget)

        # Weitere Action Buttons
        self.quick_gen_btn = QPushButton("Schnellgenerierung")
        self.quick_gen_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        self.quick_gen_btn.clicked.connect(self.quick_gen_clicked.emit)

        self.exit_btn = QPushButton("Beenden")
        self.exit_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        self.exit_btn.clicked.connect(self.exit_clicked.emit)

        layout.addWidget(self.quick_gen_btn)
        layout.addWidget(self.exit_btn)

        self.setLayout(layout)