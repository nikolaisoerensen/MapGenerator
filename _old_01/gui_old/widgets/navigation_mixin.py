#!/usr/bin/env python3
"""
Path: MapGenerator/gui/widgets/navigation_mixin.py
__init__.py existiert in "widgets"

Navigation Mixin für einheitliche Tab-Navigation
Eliminiert Code-Duplikation bei Navigation-Logik
"""

from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtCore import QRect, Qt


class NavigationMixin:
    """
    Funktionsweise: Mixin-Klasse für einheitliche Navigation zwischen Tabs
    - Wird zu Control Panel Klassen hinzugefügt
    - Stellt standard Navigation-Methoden bereit
    - Reduziert Code-Duplikation erheblich
    - Benötigt world_state Attribut in der Klasse
    """

    def setup_navigation(self, layout, show_prev=True, show_next=True,
                         prev_text="Zurück", next_text="Weiter"):
        """
        Funktionsweise: Fügt Navigation-Buttons zum Layout hinzu
        - Erstellt einheitliche Button-Struktur
        - Verbindet Buttons mit Standard-Navigation-Methoden
        - Fallback für NavigationWidget wenn nicht verfügbar

        Args:
            layout: Qt Layout wo Buttons eingefügt werden
            show_prev: Zeige Zurück-Button
            show_next: Zeige Weiter-Button
            prev_text: Text für Zurück-Button
            next_text: Text für Weiter-Button
        """
        try:
            # Versuche NavigationWidget zu verwenden
            from gui_old.navigation_widget import NavigationWidget
            self.navigation = NavigationWidget(show_prev, show_next, prev_text, next_text)
            self.navigation.prev_clicked.connect(self.prev_menu)
            self.navigation.next_clicked.connect(self.next_menu)
            self.navigation.quick_gen_clicked.connect(self.quick_generate)
            self.navigation.exit_clicked.connect(self.exit_app)
            layout.addWidget(self.navigation)

        except ImportError:
            # Fallback: Erstelle Buttons manuell
            self._create_fallback_navigation(layout, show_prev, show_next,
                                             prev_text, next_text)

        # Restart Button (einheitlich für alle Tabs)
        restart_layout = QVBoxLayout()
        self.restart_btn = QPushButton("Neu Starten")
        self.restart_btn.setStyleSheet(
            "QPushButton { background-color: #FF5722; color: white; font-weight: bold; padding: 10px; }")
        self.restart_btn.clicked.connect(self.restart_generation)
        restart_layout.addWidget(self.restart_btn)
        layout.addLayout(restart_layout)

    def _create_fallback_navigation(self, layout, show_prev, show_next, prev_text, next_text):
        """
        Funktionsweise: Erstellt Navigation-Buttons als Fallback
        - Wird aufgerufen wenn NavigationWidget nicht verfügbar
        - Repliziert die NavigationWidget Funktionalität
        """
        button_layout = QVBoxLayout()

        # Navigation Buttons (Zurück/Weiter)
        if show_prev or show_next:
            nav_layout = QHBoxLayout()
            nav_widget = QWidget()

            if show_prev:
                self.prev_btn = QPushButton(prev_text)
                self.prev_btn.setStyleSheet(
                    "QPushButton { background-color: #9E9E9E; color: white; font-weight: bold; padding: 10px; }")
                self.prev_btn.clicked.connect(self.prev_menu)
                nav_layout.addWidget(self.prev_btn)

            if show_next:
                self.next_btn = QPushButton(next_text)
                self.next_btn.setStyleSheet(
                    "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
                self.next_btn.clicked.connect(self.next_menu)
                nav_layout.addWidget(self.next_btn)

            nav_widget.setLayout(nav_layout)
            button_layout.addWidget(nav_widget)

        # Action Buttons
        self.quick_gen_btn = QPushButton("Schnellgenerierung")
        self.quick_gen_btn.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; }")
        self.quick_gen_btn.clicked.connect(self.quick_generate)

        self.exit_btn = QPushButton("Beenden")
        self.exit_btn.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        self.exit_btn.clicked.connect(self.exit_app)

        button_layout.addWidget(self.quick_gen_btn)
        button_layout.addWidget(self.exit_btn)
        layout.addLayout(button_layout)

    def navigate_to_tab(self, tab_module, tab_class):
        """
        Funktionsweise: Generische Tab-Navigation
        - Speichert aktuellen Zustand
        - Wechselt zu neuem Tab
        - Erhält Fenster-Geometrie

        Args:
            tab_module (str): Import-Pfad des Tab-Moduls
            tab_class (str): Name der Tab-Klasse
        """
        try:
            current_geometry = self.window().geometry()

            # Module importieren
            module = __import__(tab_module, fromlist=[tab_class])
            window_class = getattr(module, tab_class)

            # Neues Fenster erstellen
            new_window = window_class()

            # WICHTIG: Window-Hierarchie umstellen
            new_window.setAttribute(Qt.WA_QuitOnClose, True)  # Neues Haupt-Window
            self.window().setAttribute(Qt.WA_QuitOnClose, False)  # Altes nicht mehr wichtig

            new_window.setGeometry(current_geometry)
            new_window.show()

            # Jetzt sicher schließen
            self.window().close()

        except Exception as e:
            print(f"Navigation Fehler: {e}")

    def restart_generation(self):
        """
        Funktionsweise: Geht zurück zum Terrain Tab (Anfang)
        - Standard-Implementierung für alle Tabs
        - Kann von Subklassen überschrieben werden
        """
        print("Wechsle zum Terrain Menü (Neustart)")
        try:
            if hasattr(self, 'world_state') and self.world_state:
                self.world_state.ui_state.set_window_geometry(self.window().geometry())
        except:
            pass  # Ignoriere Geometry-Fehler

        self.navigate_to_tab('gui.tabs.terrain_tab', 'TerrainWindow')

    def exit_app(self):
        """
        Funktionsweise: Beendet die Anwendung
        - Standard-Implementierung für alle Tabs
        """
        QApplication.quit()

    def quick_generate(self):
        """
        Funktionsweise: Schnellgenerierung
        - Standard-Implementierung sammelt aktuelle Parameter
        - Kann von Subklassen erweitert werden
        """
        print("Schnellgenerierung gestartet!")
        if hasattr(self, 'get_parameters'):
            params = self.get_parameters()
            print("Parameter:", params)

    # Diese Methoden müssen von jeder Tab-Klasse implementiert werden
    def next_menu(self):
        """Einfacher nächster Tab"""
        current_class = self.__class__.__name__

        if current_class == 'TerrainWindow':
            self.navigate_to_tab('gui.tabs.geology_tab', 'GeologyWindow')
        elif current_class == 'GeologyWindow':
            self.navigate_to_tab('gui.tabs.settlement_tab', 'SettlementWindow')

    def prev_menu(self):
        """Muss in Subklasse implementiert werden"""
        raise NotImplementedError("prev_menu() muss implementiert werden")


class TabNavigationHelper:
    """
    Funktionsweise: Helper-Klasse für Tab-spezifische Navigation
    - Definiert die Navigation-Kette zwischen Tabs
    - Stellt Utility-Methoden für häufige Navigations-Operationen bereit
    """

    # Tab-Reihenfolge definieren
    TAB_SEQUENCE = [
        ('gui.tabs.terrain_tab', 'TerrainWindow'),
        ('gui.tabs.geology_tab', 'GeologyWindow'),
        ('gui.tabs.settlement_tab', 'SettlementWindow'),
        ('gui.tabs.weather_tab', 'WeatherWindow'),
        ('gui.tabs.water_tab', 'WaterWindow'),
        ('gui.tabs.biome_tab', 'BiomeWindow')
    ]

    @classmethod
    def get_next_tab(cls, current_tab_class):
        """
        Funktionsweise: Ermittelt den nächsten Tab in der Sequenz
        Args:
            current_tab_class (str): Name der aktuellen Tab-Klasse
        Returns:
            tuple: (module, class) des nächsten Tabs oder None
        """
        for i, (module, tab_class) in enumerate(cls.TAB_SEQUENCE):
            if tab_class == current_tab_class:
                if i + 1 < len(cls.TAB_SEQUENCE):
                    return cls.TAB_SEQUENCE[i + 1]
                return None  # Letzter Tab
        return None

    @classmethod
    def get_prev_tab(cls, current_tab_class):
        """
        Funktionsweise: Ermittelt den vorherigen Tab in der Sequenz
        Args:
            current_tab_class (str): Name der aktuellen Tab-Klasse
        Returns:
            tuple: (module, class) des vorherigen Tabs oder None
        """
        for i, (module, tab_class) in enumerate(cls.TAB_SEQUENCE):
            if tab_class == current_tab_class:
                if i > 0:
                    return cls.TAB_SEQUENCE[i - 1]
                return None  # Erster Tab
        return None

    @classmethod
    def go_to_main_menu(cls, current_window, world_state=None):
        """
        Funktionsweise: Kehrt zum Hauptmenü zurück
        - Kann von jedem Tab aufgerufen werden
        """
        try:
            # Speichere Geometrie
            current_geometry = current_window.geometry()

            # Importiere und erstelle Hauptmenü
            from gui_old.main_menu import MainMenuWindow
            main_menu = MainMenuWindow()

            # Setze Geometrie
            main_menu.setGeometry(current_geometry)
            main_menu.show()

            # Schließe aktuelles Fenster sicher
            current_window.close()

        except Exception as e:
            print(f"Fehler beim Wechsel zum Hauptmenü: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: App beenden
            from PyQt5.QtWidgets import QApplication
            QApplication.quit()