#!/usr/bin/env python3
"""
Angepasste Main Menu für Core-Integration
Vereinfachte Version die direkt mit terrain_tab.py arbeitet
"""

import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QSpacerItem,
                             QSizePolicy, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor


class MainMenuWindow(QMainWindow):
    """Hauptmenü der World Generator Anwendung - Core-Integration Version"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("World Generator - Core Integration")
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
        subtitle_label = QLabel("Erstelle deine eigene Fantasywelt mit Core-Integration")
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
        button_container.setMaximumWidth(350)
        button_container.setStyleSheet("""
            QWidget {
                background-color: #ecf0f1;
                border-radius: 10px;
                padding: 20px;
            }
        """)

        # Buttons
        self.new_map_btn = QPushButton("Neue Karte (Terrain)")
        self.new_map_btn.setStyleSheet(self.get_button_style("#27ae60"))
        self.new_map_btn.clicked.connect(self.new_map)

        self.test_integration_btn = QPushButton("Integration Testen")
        self.test_integration_btn.setStyleSheet(self.get_button_style("#3498db"))
        self.test_integration_btn.clicked.connect(self.test_integration)

        self.load_game_btn = QPushButton("Spiel laden")
        self.load_game_btn.setStyleSheet(self.get_button_style("#95a5a6"))
        self.load_game_btn.clicked.connect(self.load_game)
        self.load_game_btn.setEnabled(False)  # Nicht implementiert

        self.settings_btn = QPushButton("Einstellungen")
        self.settings_btn.setStyleSheet(self.get_button_style("#95a5a6"))
        self.settings_btn.clicked.connect(self.settings)
        self.settings_btn.setEnabled(False)  # Nicht implementiert

        self.exit_btn = QPushButton("Verlassen")
        self.exit_btn.setStyleSheet(self.get_button_style("#e74c3c"))
        self.exit_btn.clicked.connect(self.exit_app)

        # Buttons zu Layout hinzufügen
        button_layout.addWidget(self.new_map_btn)
        button_layout.addWidget(self.test_integration_btn)
        button_layout.addWidget(self.load_game_btn)
        button_layout.addWidget(self.settings_btn)
        button_layout.addWidget(self.exit_btn)
        button_container.setLayout(button_layout)

        # Status-Info Label
        self.status_label = QLabel("Bereit für Core-Integration")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #27ae60;
                background-color: rgba(39, 174, 96, 0.1);
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
            }
        """)

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
        main_layout.addWidget(self.status_label)
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
                transform: translateY(-1px);
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color, 0.2)};
                transform: translateY(1px);
            }}
            QPushButton:disabled {{
                background-color: #bdc3c7;
                color: #7f8c8d;
            }}
        """

    def darken_color(self, hex_color, factor=0.1):
        """Vereinfachte Farbverdunkelung"""
        # Für Demo-Zwecke vereinfacht
        darker_colors = {
            '#27ae60': '#229954',
            '#3498db': '#2980b9',
            '#e74c3c': '#c0392b',
            '#95a5a6': '#7f8c8d'
        }
        return darker_colors.get(hex_color, hex_color)

    def new_map(self):
        """Startet den Terrain Editor mit Core-Integration"""
        print("Starte Terrain Editor mit Core-Integration...")

        # Zeige Loading-Feedback
        self.new_map_btn.setEnabled(False)
        self.new_map_btn.setText("Lade...")
        self.status_label.setText("Lade Terrain Editor...")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #f39c12;
                background-color: rgba(243, 156, 18, 0.1);
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
            }
        """)

        # Verarbeite Events damit UI responsive bleibt
        QApplication.processEvents()

        # Pre-importiere matplotlib um Cache zu nutzen
        import matplotlib
        matplotlib.use('Qt5Agg')  # Backend früh setzen

        # Teste Core-Module Import
        try:
            from core import BaseTerrainGenerator
            from gui_old.managers.parameter_manager import WorldParameterManager
            print("✓ Core-Module erfolgreich importiert")
            self.status_label.setText("Core-Module geladen...")
            QApplication.processEvents()
        except ImportError as e:
            raise Exception(f"Core-Module fehlen: {e}")

        # Importiere Terrain Tab
        try:
            from gui_old.tabs.terrain_tab import TerrainWindow
            print("✓ Terrain Tab erfolgreich importiert")
            self.status_label.setText("GUI-Module geladen...")
            QApplication.processEvents()
        except ImportError:
            # Fallback: Versuche direkten Import
            import sys
            import os

            # Füge aktuelles Verzeichnis zum Path hinzu
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            from gui_old.tabs.terrain_tab import TerrainWindow
            print("Terrain Tab mit Fallback importiert")

        # Erstelle und zeige Terrain Window
        self.terrain_window = TerrainWindow()
        self.terrain_window.setGeometry(self.geometry())
        self.terrain_window.show()

        print("✓ Terrain Editor erfolgreich gestartet")
        self.close()

    def test_integration(self):
        """Testet die Core-Integration"""
        print("Teste Core-Integration...")

        self.test_integration_btn.setEnabled(False)
        self.test_integration_btn.setText("Teste...")
        self.status_label.setText("Führe Integration-Tests durch...")
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #3498db;
                background-color: rgba(52, 152, 219, 0.1);
                border-radius: 5px;
                padding: 10px;
                margin: 10px;
            }
        """)

        QApplication.processEvents()

        try:
            # Führe Integration-Tests durch
            import sys
            import os

            # Füge aktuelles Verzeichnis zum Path hinzu
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Test Core-Module
            from core import BaseTerrainGenerator, validate_terrain_parameters
            from gui_old.managers.parameter_manager import WorldParameterManager

            # Schnell-Test
            generator = BaseTerrainGenerator(64, 64, 42)
            test_params = {
                'size': 64,
                'height': 100,
                'octaves': 4,
                'frequency': 0.015,
                'persistence': 0.6,
                'lacunarity': 2.0,
                'redistribute_power': 1.2,
                'seed': 12345
            }

            heightmap = generator.generate_heightmap(**test_params)
            is_valid, corrected, warnings = validate_terrain_parameters(**test_params)

            world_state = WorldParameterManager()
            world_state.set_terrain_params(test_params)

            # Test erfolgreich
            self.status_label.setText("✓ Integration-Tests erfolgreich!")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #27ae60;
                    background-color: rgba(39, 174, 96, 0.1);
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px;
                }
            """)

            print("✓ Integration-Tests erfolgreich!")
            print(f"  - Heightmap generiert: {heightmap.shape}")
            print(f"  - Parameter validiert: {len(warnings)} Warnungen")
            print(f"  - WorldState funktioniert")

        except Exception as e:
            print(f"✗ Integration-Tests fehlgeschlagen: {e}")
            self.status_label.setText(f"✗ Test fehlgeschlagen: {str(e)[:30]}...")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 12px;
                    color: #e74c3c;
                    background-color: rgba(231, 76, 60, 0.1);
                    border-radius: 5px;
                    padding: 10px;
                    margin: 10px;
                }
            """)

        finally:
            self.test_integration_btn.setEnabled(True)
            self.test_integration_btn.setText("Integration Testen")

    def show_error_details(self, error):
        """Zeigt detaillierte Fehler-Information"""
        from PyQt5.QtWidgets import QMessageBox

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Integration Fehler")
        msg.setText("Fehler beim Starten des Terrain Editors")
        msg.setDetailedText(f"""
Fehler: {str(error)}

Mögliche Lösungen:
1. Stelle sicher, dass alle Dependencies installiert sind:
   - opensimplex: pip install opensimplex
   - PyQt5: pip install PyQt5
   - matplotlib: pip install matplotlib
   - numpy: pip install numpy
   - scipy: pip install scipy

2. Prüfe die Dateistruktur:
   - terrain_generator.py im Root-Verzeichnis
   - world_state.py im Root-Verzeichnis
   - terrain_tab.py im gui/tabs/ Verzeichnis

3. Führe Integration-Tests aus:
   - Klicke "Integration Testen" für detaillierte Diagnose

4. Prüfe Python-Path und Imports
        """)
        msg.exec_()

    def load_game(self):
        """Lädt ein gespeichertes Spiel (nicht implementiert)"""
        print("Spiel laden - noch nicht implementiert")

    def settings(self):
        """Öffnet die Einstellungen (nicht implementiert)"""
        print("Einstellungen - noch nicht implementiert")

    def exit_app(self):
        """Beendet die Anwendung"""
        QApplication.quit()

    def resizeEvent(self, event):
        """Behält Proportionen beim Resize bei"""
        super().resizeEvent(event)


# Hauptfunktion für direkten Start
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Setze Application-Properties
    app.setApplicationName("World Generator")
    app.setApplicationVersion("1.0 - Core Integration")

    # Hauptfenster erstellen
    window = MainMenuWindow()
    window.show()

    sys.exit(app.exec_())