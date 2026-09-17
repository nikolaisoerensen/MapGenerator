"""
Path: managers/navigation_manager.py

Funktionsweise: Zentrale Tab-Navigation und Parameter-Persistierung
- Tab-Reihenfolge: main_menu → terrain ⇆ geology ⇆ erosion ⇆ weather ⇆ water ⇆ biome ⇆ settlement ⇆ overview
- Automatische Parameter-Speicherung vor Tab-Wechsel
- Window-Geometrie Persistierung
- Graceful Cleanup und Resource-Management
- Validation vor Navigation (alle Dependencies erfüllt?)

Kommunikationskanäle:
- Signals: tab_changed, parameters_saved, validation_failed
- Config: gui_default.py für Window-Settings
- Data: Koordination mit data_lod_manager für Parameter-Transfer
"""
import logging

from PyQt6.QtCore import QObject, pyqtSignal
from gui.config.gui_default import WindowSettings


# KANONISCHE TAB-REIHENFOLGE (Ticket #56, 2026-09-17).
#
# Diese Liste ist die Vorwaerts/Zurueck-Reihenfolge der GENERATOR-Pipeline
# (main_menu + die Reiter, durch die sich der Nutzer sequentiell durcharbeitet,
# plus die abschliessende Overview). Vorher stand sie an FUENF Stellen separat:
# hier, in gui/map_editor.py (dort aber als LEERE Liste, siehe Kommentar an
# self.tab_order dort - die wird dynamisch aus tab_configs gefuellt und ist
# deshalb bewusst NICHT an diese Konstante gekoppelt, siehe dort) und
# DREIMAL in gui/widgets/widgets.py (NavigationPanel.update_navigation_buttons/
# go_previous/go_next) - wobei zwei der drei Kopien "erosion" schlicht
# vergessen hatten (Kopierfehler, keine Absicht).
#
# ENTSCHEIDUNG "erosion": GEHOERT dazu. gui/tabs/erosion_tab.py::ErosionTab
# ist ein echter, registrierter Reiter (siehe tab_configs in
# gui/map_editor.py._setup_tabs(), Eintrag ("erosion", "Erosion", ErosionTab,
# ...) zwischen "geology" und "weather") - kein Platzhalter, kein totes Feature.
# EROSION_AKTIV (gui/config/value_default.py) schaltet nur die BERECHNUNG ab,
# nicht den Reiter selbst; der Tab bleibt sichtbar und navigierbar. Die zwei
# Kopien ohne "erosion" in widgets.py waren damit ein Bug, keine bewusste
# Auslassung - diese Konstante behebt ihn, indem es nur noch eine Wahrheit
# gibt.
GENERATOR_TAB_ORDER = [
    "main_menu",
    "terrain",
    "geology",
    # Erosion sitzt zwischen Geology und Weather - dieselbe Reihenfolge
    # wie in der Pipeline (siehe CALCULATOR_GRAPH).
    "erosion",
    "weather",
    "water",
    "biome",
    "settlement",
    "overview",
]


class NavigationManager(QObject):
    """
    Funktionsweise: Zentrale Verwaltung der Tab-Navigation zwischen allen Map-Editor Komponenten
    Aufgabe: Koordiniert Tab-Wechsel, Parameter-Persistierung
    """

    # Signals für Cross-Tab Communication
    tab_changed = pyqtSignal(str, str)  # (from_tab, to_tab)
    parameters_saved = pyqtSignal(str)  # (tab_name)

    def __init__(self, data_lod_manager):
        """
        Funktionsweise: Initialisiert NavigationManager mit Tab-Reihenfolge
        Aufgabe: Setzt Standard-Tab-Reihenfolge für die reine Navigation
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.data_lod_manager = data_lod_manager
        self.current_tab = "main_menu"

        # Tab-Reihenfolge wie in Dokumentation definiert.
        # Kopie (nicht Referenz) der Modul-Konstante GENERATOR_TAB_ORDER, damit
        # eine Instanz ihre eigene Liste besitzt, falls sie zur Laufzeit
        # veraendert wird (siehe z.B. Tests). Die Konstante selbst ist die
        # kanonische Quelle, siehe Kommentar dort.
        self.tab_order = list(GENERATOR_TAB_ORDER)

        # Window-Geometrie Cache
        self.window_geometries = {}

    def navigate_to_tab(self, target_tab):
        """
        Funktionsweise: Navigiert zu angegebenem Tab und speichert vorher die Parameter
        Aufgabe: Speichert Parameter des aktuellen Tabs und wechselt zum Ziel-Tab.
                 Es findet keine Dependency-Prüfung statt: alle Tabs sind jederzeit erreichbar,
                 da die Berechnungen ohnehin fortlaufend im Hintergrund aufgefüllt werden.
        Parameter: target_tab (str) - Name des Ziel-Tabs
        """
        if not self._save_current_parameters():
            return False

        old_tab = self.current_tab
        self.current_tab = target_tab

        self.tab_changed.emit(old_tab, target_tab)
        return True


    def _current_tab_index(self):
        """
        Funktionsweise: Ermittelt den Index des aktuellen Tabs in der Tab-Reihenfolge
        Aufgabe: Kapselt die Index-Suche und schützt vor einem unbekannten current_tab,
                 damit navigate_next/navigate_previous nie mit einem ValueError abbrechen
        Return: Index in tab_order oder None, wenn current_tab nicht in der Reihenfolge liegt
        """
        if self.current_tab not in self.tab_order:
            self.logger.warning(
                f"Aktueller Tab '{self.current_tab}' nicht in tab_order, Navigation abgebrochen")
            return None
        return self.tab_order.index(self.current_tab)

    def navigate_next(self):
        """
        Funktionsweise: Navigiert zum nächsten Tab in der definierten Reihenfolge
        Aufgabe: Automatische Vorwärts-Navigation durch Tab-Sequenz
        """
        current_index = self._current_tab_index()
        if current_index is None:
            return False
        if current_index < len(self.tab_order) - 1:
            next_tab = self.tab_order[current_index + 1]
            return self.navigate_to_tab(next_tab)
        return False

    def navigate_previous(self):
        """
        Funktionsweise: Navigiert zum vorherigen Tab in der definierten Reihenfolge
        Aufgabe: Automatische Rückwärts-Navigation durch Tab-Sequenz
        """
        current_index = self._current_tab_index()
        if current_index is None:
            return False
        if current_index > 0:
            prev_tab = self.tab_order[current_index - 1]
            return self.navigate_to_tab(prev_tab)
        return False

    def _save_current_parameters(self):
        """
        Funktionsweise: Speichert Parameter des aktuellen Tabs vor Navigation
        Aufgabe: Automatische Parameter-Persistierung über data_lod_manager
        """
        try:
            # Parameter-Speicherung wird vom jeweiligen Tab selbst durchgeführt
            # NavigationManager triggert nur das Signal
            self.parameters_saved.emit(self.current_tab)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save parameters for tab '{self.current_tab}': {e}")
            return False

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
