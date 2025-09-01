"""
Path: gui/main_menu.py

Funktionsweise: Einfaches Hauptmenü mit 4 Buttons - GEÄNDERT für Signal-Navigation
- Map-Editor Button sendet Signal statt direkte Loading-Erstellung
- Manager von main.py übernehmen statt eigene Erstellung
- Backward-Compatibility für direkten MainMenu-Start
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import logging

from gui.widgets.widgets import BaseButton, GradientBackgroundWidget
from gui.config.gui_default import WindowSettings, LayoutSettings

def get_error_decorators():
    """
    Funktionsweise: Lazy Loading von Error Handler Decorators mit Fallback
    Aufgabe: Lädt Error Handler nur wenn verfügbar, sonst No-op Decorators
    Return: Tuple von Decorator-Funktionen (initialization_handler, ui_navigation_handler)
    """
    try:
        from gui.error_handler import initialization_handler, ui_navigation_handler
        return initialization_handler, ui_navigation_handler
    except ImportError:
        # Graceful Fallback - No-op Decorators wenn Error Handler nicht verfügbar
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator

# Decorators global laden
initialization_handler, ui_navigation_handler = get_error_decorators()

class MainMenuWindow(QMainWindow):
    """
    Funktionsweise: Einfaches Hauptmenü-Fenster als Entry-Point der Anwendung
    Aufgabe: 4 Buttons für Navigation - Map-Editor, Laden, Settings, Beenden
    Kommunikation: map_editor_requested Signal für Navigation zu main.py
    """

    # NEUES SIGNAL FÜR MAIN.PY INTEGRATION
    map_editor_requested = pyqtSignal()

    def __init__(self, shader_manager=None, parameter_manager=None, generation_orchestrator=None,
                 data_manager=None, navigation_manager=None):
        """
        Funktionsweise: Initialisiert MainMenu mit optionalen Manager-References
        Aufgabe: UI Setup ohne eigene Manager-Erstellung (kommt von main.py)
        Parameter: data_manager, navigation_manager von main.py (optional für Backward-Compatibility)
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)

        self.data_manager = data_manager
        self.navigation_manager = navigation_manager
        self.generation_orchestrator = generation_orchestrator
        self.shader_manager = shader_manager
        self.parameter_manager = parameter_manager

        # Window Setup
        self.setup_window()

        # UI Setup
        self.setup_ui()

        self.logger.info("Main Menu initialized")

    def setup_window(self):
        """
        Funktionsweise: Konfiguriert Hauptfenster-Eigenschaften mit gui_default
        Aufgabe: Größe, Position, Titel aus WindowSettings
        """
        # Window Properties
        self.setWindowTitle("Map Generator - Main Menu")

        # Window Size aus gui_default.py
        main_menu_settings = WindowSettings.MAIN_MENU
        width = main_menu_settings.get("width", 800)
        height = main_menu_settings.get("height", 600)
        min_width = main_menu_settings.get("min_width", 600)
        min_height = main_menu_settings.get("min_height", 400)

        self.resize(width, height)
        self.setMinimumSize(min_width, min_height)

        # Center Window
        self.center_window()

        # Window Flags
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)

    def center_window(self):
        """
        Funktionsweise: Zentriert Fenster auf dem Bildschirm
        Aufgabe: Berechnet Bildschirmmitte und positioniert Fenster dort
        """
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)

    def setup_ui(self):
        """
        Funktionsweise: Erstellt einfache UI für Hauptmenü
        Aufgabe: Gradient-Background, Titel, 4 Buttons
        """
        # Central Widget mit Gradient Background
        central_widget = GradientBackgroundWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(LayoutSettings.MARGIN * 3, LayoutSettings.MARGIN * 3,
                                       LayoutSettings.MARGIN * 3, LayoutSettings.MARGIN * 3)
        main_layout.setSpacing(LayoutSettings.MARGIN * 2)

        # Logo/Title Section
        title_section = self.create_title_section()
        main_layout.addWidget(title_section)

        # Button Section
        button_section = self.create_button_section()
        main_layout.addWidget(button_section)

        # Spacer für bessere Verteilung
        main_layout.addStretch()

        central_widget.setLayout(main_layout)

    def create_title_section(self) -> QWidget:
        """
        Funktionsweise: Erstellt Title/Logo Sektion
        Return: QWidget mit Logo und Titel
        """
        section = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # Application Title
        title_label = QLabel("MAP GENERATOR")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 48px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(title_label)

        # Subtitle
        subtitle_label = QLabel("World Generation Suite")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                color: #34495e;
                font-style: italic;
            }
        """)
        layout.addWidget(subtitle_label)

        section.setLayout(layout)
        return section

    def create_button_section(self) -> QWidget:
        """
        Funktionsweise: Erstellt Button-Sektion mit 4 Hauptbuttons
        Return: QWidget mit allen Hauptmenü-Buttons (Styling aus gui_default)
        """
        section = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(LayoutSettings.PADDING)

        # Button Container für maximale Breite
        button_container = QWidget()
        button_container.setMaximumWidth(300)
        button_layout = QVBoxLayout()
        button_layout.setSpacing(LayoutSettings.PADDING)

        # Map-Editor Button (funktional)
        self.map_editor_button = BaseButton("Map-Editor", "primary")
        self.map_editor_button.setMinimumHeight(
            LayoutSettings.BUTTON_HEIGHT + 15)  # Etwas höher für Hauptbutton
        self.map_editor_button.clicked.connect(self.start_map_editor)
        button_layout.addWidget(self.map_editor_button)

        # Load Button (placeholder)
        self.load_button = BaseButton("Laden", "secondary")
        self.load_button.setMinimumHeight(LayoutSettings.BUTTON_HEIGHT + 10)
        self.load_button.clicked.connect(self.show_placeholder_message)
        button_layout.addWidget(self.load_button)

        # Settings Button (placeholder)
        self.settings_button = BaseButton("Settings", "secondary")
        self.settings_button.setMinimumHeight(LayoutSettings.BUTTON_HEIGHT + 10)
        self.settings_button.clicked.connect(self.show_placeholder_message)
        button_layout.addWidget(self.settings_button)

        # Exit Button
        self.exit_button = BaseButton("Beenden", "danger")
        self.exit_button.setMinimumHeight(LayoutSettings.BUTTON_HEIGHT + 10)
        self.exit_button.clicked.connect(self.exit_application)
        button_layout.addWidget(self.exit_button)

        button_container.setLayout(button_layout)

        # Container in Section Layout
        container_layout = QHBoxLayout()
        container_layout.addStretch()
        container_layout.addWidget(button_container)
        container_layout.addStretch()

        layout.addLayout(container_layout)
        section.setLayout(layout)
        return section

    def start_map_editor(self):
        """
        Funktionsweise: Sendet Signal für Map-Editor Start (keine direkte Aktion)
        Aufgabe: Informiert main.py über Map-Editor Request
        """
        try:
            self.logger.info("Map Editor requested")

            # Signal an main.py senden
            self.map_editor_requested.emit()

        except Exception as e:
            self.show_error_message("Error", f"Failed to start Map Editor:\n{str(e)}")
            self.logger.error(f"Map Editor request failed: {e}")

    def show_placeholder_message(self):
        """
        Funktionsweise: Zeigt Placeholder-Message für nicht implementierte Features
        Aufgabe: User-freundliche Info für Laden/Settings Buttons
        """
        sender = self.sender()
        button_text = sender.text() if sender else "Feature"

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(f"{button_text} - In Development")
        msg_box.setText(f"{button_text} ist noch nicht implementiert.")

        feature_info = {
            "Laden": "Hier können Sie gespeicherte Welten laden.\nFormate: JSON, PNG-Sets",
            "Settings": "Hier können Sie Anwendungseinstellungen ändern.\nSprache, Performance, Pfade"
        }

        info_text = feature_info.get(button_text,
                                     "Diese Funktion wird in einer zukünftigen Version verfügbar sein.")
        msg_box.setInformativeText(info_text)

        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

        self.logger.info(f"Placeholder message shown for '{button_text}'")

    def exit_application(self):
        """
        Funktionsweise: Beendet Anwendung direkt ohne Bestätigung
        Aufgabe: Einfaches QApplication.quit()
        """
        self.logger.info("Application exit initiated")
        QApplication.quit()

    def show_error_message(self, title: str, message: str):
        """
        Funktionsweise: Zeigt standardisierte Error-Message
        Parameter: title (str), message (str) - Titel und Nachricht des Fehlers
        Aufgabe: Einheitliche Error-Darstellung
        """
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def closeEvent(self, event):
        """
        Funktionsweise: Beendet Anwendung bei X-Button Klick
        Aufgabe: Direkte Beendigung ohne Bestätigung
        Parameter: event (QCloseEvent)
        """
        self.exit_application()
        event.accept()