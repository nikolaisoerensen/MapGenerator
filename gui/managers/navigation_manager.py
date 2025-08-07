"""
Path: gui/managers/navigation_manager.py

Funktionsweise: Zentrale Tab-Navigation und Parameter-Persistierung
- Tab-Reihenfolge: main_menu → loading → terrain ⇆ geology ⇆ weather ⇆ water ⇆ biome ⇆ settlement ⇆ overview
- Automatische Parameter-Speicherung vor Tab-Wechsel
- Window-Geometrie Persistierung
- Graceful Cleanup und Resource-Management
- Validation vor Navigation (alle Dependencies erfüllt?)

Kommunikationskanäle:
- Signals: tab_changed, parameters_saved, validation_failed
- Config: gui_default.py für Window-Settings
- Data: Koordination mit data_manager für Parameter-Transfer
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
from gui.config.gui_default import WindowSettings


class NavigationManager(QObject):
    """
    Funktionsweise: Zentrale Verwaltung der Tab-Navigation zwischen allen Map-Editor Komponenten
    Aufgabe: Koordiniert Tab-Wechsel, Parameter-Persistierung und Dependency-Validation
    """

    # Signals für Cross-Tab Communication
    tab_changed = pyqtSignal(str, str)  # (from_tab, to_tab)
    parameters_saved = pyqtSignal(str)  # (tab_name)
    validation_failed = pyqtSignal(str, str)  # (tab_name, error_message)

    def __init__(self, data_manager):
        """
        Funktionsweise: Initialisiert NavigationManager mit Tab-Reihenfolge und Dependencies
        Aufgabe: Setzt Standard-Tab-Reihenfolge und verknüpft mit data_manager
        """
        super().__init__()
        self.data_manager = data_manager
        self.current_tab = "main_menu"

        # Tab-Reihenfolge wie in Dokumentation definiert
        self.tab_order = [
            "main_menu",
            "terrain",
            "geology",
            "weather",
            "water",
            "biome",
            "settlement",
            "overview"
        ]

        # Dependency-Mapping für Validation
        self.tab_dependencies = {
            "main_menu": [],
            "terrain": [],
            "geology": ["terrain"],
            "weather": ["terrain"],
            "water": ["terrain", "geology", "weather"],
            "biome": ["terrain", "weather", "water"],
            "settlement": ["terrain", "water", "biome"],
            "overview": ["terrain", "geology", "settlement", "weather", "water", "biome"]
        }

        # Window-Geometrie Cache
        self.window_geometries = {}

    def navigate_to_tab(self, target_tab):
        """
        Funktionsweise: Navigiert zu angegebenem Tab nach Validation und Parameter-Speicherung
        Aufgabe: Prüft Dependencies, speichert Parameter und wechselt Tab
        Parameter: target_tab (str) - Name des Ziel-Tabs
        """
        if not self._validate_dependencies(target_tab):
            return False

        if not self._save_current_parameters():
            return False

        old_tab = self.current_tab
        self.current_tab = target_tab

        self.tab_changed.emit(old_tab, target_tab)
        return True

    def navigate_next(self):
        """
        Funktionsweise: Navigiert zum nächsten Tab in der definierten Reihenfolge
        Aufgabe: Automatische Vorwärts-Navigation durch Tab-Sequenz
        """
        current_index = self.tab_order.index(self.current_tab)
        if current_index < len(self.tab_order) - 1:
            next_tab = self.tab_order[current_index + 1]
            return self.navigate_to_tab(next_tab)
        return False

    def navigate_previous(self):
        """
        Funktionsweise: Navigiert zum vorherigen Tab in der definierten Reihenfolge
        Aufgabe: Automatische Rückwärts-Navigation durch Tab-Sequenz
        """
        current_index = self.tab_order.index(self.current_tab)
        if current_index > 0:
            prev_tab = self.tab_order[current_index - 1]
            return self.navigate_to_tab(prev_tab)
        return False

    def _validate_dependencies(self, target_tab):
        """
        Funktionsweise: Prüft ob alle Dependencies für Ziel-Tab erfüllt sind
        Aufgabe: Validation basierend auf verfügbaren Daten im data_manager
        Parameter: target_tab (str) - Tab der validiert werden soll
        """
        required_deps = self.tab_dependencies.get(target_tab, [])

        for dependency in required_deps:
            if not self._check_tab_completion(dependency):
                error_msg = f"Tab '{dependency}' must be completed before accessing '{target_tab}'"
                self.validation_failed.emit(target_tab, error_msg)
                self._show_validation_error(error_msg)
                return False

        return True

    def _check_tab_completion(self, tab_name):
        """
        Funktionsweise: Prüft ob ein Tab alle erforderlichen Outputs generiert hat
        Aufgabe: Überprüft data_manager auf vorhandene Daten für gegebenen Tab
        Parameter: tab_name (str) - Name des zu prüfenden Tabs
        """
        required_outputs = {
            "terrain": ["heightmap", "slopemap", "shadowmap"],
            "geology": ["rock_map", "hardness_map"],
            "weather": ["wind_map", "precip_map", "humid_map", "temp_map"],
            "water": ["water_map", "flow_map", "soil_moist_map", "water_biomes_map"],
            "biome": ["biome_map", "super_biome_mask"],
            "settlement": ["settlement_list", "landmark_list", "roadsite_list", "plot_map", "civ_map"]
        }

        outputs = required_outputs.get(tab_name, [])
        for output in outputs:
            if not self.data_manager.has_data(output):
                return False

        return True

    def _save_current_parameters(self):
        """
        Funktionsweise: Speichert Parameter des aktuellen Tabs vor Navigation
        Aufgabe: Automatische Parameter-Persistierung über data_manager
        """
        try:
            # Parameter-Speicherung wird vom jeweiligen Tab selbst durchgeführt
            # NavigationManager triggert nur das Signal
            self.parameters_saved.emit(self.current_tab)
            return True
        except Exception as e:
            error_msg = f"Failed to save parameters for tab '{self.current_tab}': {str(e)}"
            self.validation_failed.emit(self.current_tab, error_msg)
            return False

    def _show_validation_error(self, message):
        """
        Funktionsweise: Zeigt Validation-Fehler in MessageBox an
        Aufgabe: User-friendly Error-Display für Navigation-Probleme
        Parameter: message (str) - Fehlermeldung für Benutzer
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Navigation Error")
        msg_box.setText(message)
        msg_box.exec_()

    def save_window_geometry(self, tab_name, geometry):
        """
        Funktionsweise: Speichert Window-Geometrie für gegebenen Tab
        Aufgabe: Window-Geometrie Persistierung zwischen Sessions
        Parameter: tab_name (str), geometry (QRect) - Tab-Name und Fenster-Geometrie
        """
        self.window_geometries[tab_name] = geometry

    def restore_window_geometry(self, tab_name):
        """
        Funktionsweise: Stellt gespeicherte Window-Geometrie für Tab wieder her
        Aufgabe: Lädt gespeicherte Fenster-Position und -Größe
        Parameter: tab_name (str) - Tab dessen Geometrie wiederhergestellt werden soll
        Returns: QRect oder None falls keine Geometrie gespeichert
        """
        return self.window_geometries.get(tab_name, None)

    def get_default_window_size(self, window_type="MAP_EDITOR"):
        """
        Funktionsweise: Gibt Standard-Fenstergröße aus gui_default.py zurück
        Aufgabe: Lädt Default-Geometrie für verschiedene Window-Typen
        Parameter: window_type (str) - Typ des Fensters (MAIN_MENU oder MAP_EDITOR)
        Returns: Dict mit width/height Werten
        """
        if window_type == "MAIN_MENU":
            return WindowSettings.MAIN_MENU
        else:
            return WindowSettings.MAP_EDITOR

    def cleanup(self):
        """
        Funktionsweise: Graceful Cleanup bei Anwendungs-Beendigung
        Aufgabe: Speichert finale Parameter und räumt Ressourcen auf
        """
        try:
            self._save_current_parameters()
            # Weitere Cleanup-Operationen können hier hinzugefügt werden
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")

    def get_current_tab(self):
        """
        Funktionsweise: Gibt aktuell aktiven Tab zurück
        Aufgabe: Tab-Status-Abfrage für andere Komponenten
        Returns: str - Name des aktuellen Tabs
        """
        return self.current_tab

    def get_tab_index(self, tab_name):
        """
        Funktionsweise: Gibt Index des Tabs in der Navigation-Reihenfolge zurück
        Aufgabe: Position-Berechnung für Progress-Anzeige
        Parameter: tab_name (str) - Name des Tabs
        Returns: int - Index in tab_order oder -1 falls nicht gefunden
        """
        try:
            return self.tab_order.index(tab_name)
        except ValueError:
            return -1