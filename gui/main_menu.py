#!/usr/bin/env python3
"""
Path: MapGenerator/gui/main_menu.py
__init__.py existiert in "gui"

World Generator - Hauptmenü
Entry Point für die Anwendung
"""

import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QSpacerItem,
                             QSizePolicy, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor


class MainMenuWindow(QMainWindow):
    """Hauptmenü der World Generator Anwendung"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator")
        self.setGeometry(200, 200, 1500, 1000)
        self.setMinimumSize(1500, 1000)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QVBoxLayout()

        # Titel
        title_label = QLabel("World Generator")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 36px;
                font-weight: bold;
                color: #2c3e50;
                margin: 30px;
            }
        """)

        # Untertitel
        subtitle_label = QLabel("Erstelle deine eigene Fantasywelt")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: #7f8c8d;
                margin-bottom: 40px;
            }
        """)

        # Button Container
        button_container = QWidget()
        button_layout = QVBoxLayout()
        button_container.setMaximumWidth(300)
        button_container.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
                border-radius: 10px;
                padding: 20px;
            }
        """)

        # Buttons
        self.new_map_btn = QPushButton("Neue Karte")
        self.new_map_btn.setStyleSheet(self.get_button_style("#27ae60"))
        self.new_map_btn.clicked.connect(self.new_map)

        self.load_game_btn = QPushButton("Spiel laden")
        self.load_game_btn.setStyleSheet(self.get_button_style("#95a5a6"))
        self.load_game_btn.clicked.connect(self.load_game)
        self.load_game_btn.setEnabled(False)  # Nicht funktionierend

        self.settings_btn = QPushButton("Einstellungen")
        self.settings_btn.setStyleSheet(self.get_button_style("#95a5a6"))
        self.settings_btn.clicked.connect(self.settings)
        self.settings_btn.setEnabled(False)  # Nicht funktionierend


        self.exit_btn = QPushButton("Verlassen")
        self.exit_btn.setStyleSheet(self.get_button_style("#e74c3c"))
        self.exit_btn.clicked.connect(self.exit_app)

        # Buttons zu Layout hinzufügen
        button_layout.addWidget(self.new_map_btn)
        button_layout.addWidget(self.load_game_btn)
        button_layout.addWidget(self.settings_btn)
        button_layout.addWidget(self.exit_btn)
        button_container.setLayout(button_layout)

        # Spacer für zentrierte Ausrichtung
        top_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        bottom_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Layout zusammensetzen
        main_layout.addItem(top_spacer)
        main_layout.addWidget(title_label)
        main_layout.addWidget(subtitle_label)

        # Button Container zentrieren
        button_container_layout = QHBoxLayout()
        button_container_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        button_container_layout.addWidget(button_container)
        button_container_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(button_container_layout)
        main_layout.addItem(bottom_spacer)

        central_widget.setLayout(main_layout)

        # Hintergrund-Styling
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #bdc3c7, stop:1 #ecf0f1);
            }
        """)

    def get_button_style(self, color):
        """Einheitlicher Button-Style"""
        return f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                padding: 15px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                margin: 5px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color, 0.2)};
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #7f8c8d;
            }}
        """

    def darken_color(self, hex_color, factor=0.1):
        """Verdunkelt eine Hex-Farbe"""
        # Vereinfachte Verdunkelung - für Produktionscode würde man QColor verwenden
        return hex_color.replace('#', '#AA')  # Placeholder

    def new_map(self):
        """Startet den Terrain Editor"""
        print("Starte Terrain Editor...")

        # Zeige Loading-Feedback
        self.setEnabled(False)
        self.setWindowTitle("World Generator - Lade Terrain Editor...")

        # Verarbeite Events damit UI responsive bleibt
        from PyQt5.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            # Pre-importiere matplotlib um Cache zu nutzen
            import matplotlib
            matplotlib.use('Qt5Agg')  # Backend früh setzen

            from gui.tabs.terrain_tab import TerrainWindow
            from gui.world_state import WorldState

            world_state = WorldState()
            world_state.set_window_geometry(self.geometry())

            self.terrain_window = TerrainWindow()

            # Übernehme Größe und zeige
            self.terrain_window.setGeometry(self.geometry())
            self.terrain_window.show()
            self.close()

        except Exception as e:
            self.setEnabled(True)
            self.setWindowTitle("World Generator")
            print(f"Fehler beim Starten des Terrain Editors: {e}")
            import traceback
            traceback.print_exc()

    def load_game(self):
        """Lädt ein gespeichertes Spiel (nicht implementiert)"""
        print("Spiel laden - nicht implementiert")

    def settings(self):
        """Öffnet die Einstellungen (nicht implementiert)"""
        print("Einstellungen - nicht implementiert")

    def exit_app(self):
        """Beendet die Anwendung"""
        QApplication.quit()

    def resizeEvent(self, event):
        """Behält Proportionen beim Resize bei"""
        super().resizeEvent(event)