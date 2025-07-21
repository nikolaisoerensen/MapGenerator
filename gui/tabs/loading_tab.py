"""
Path: gui/tabs/loading_tab.py

Funktionsweise: Einfacher Ladebildschirm mit sequenzieller Generator-Berechnung
- Ladebalken mit 6 Stufen für alle Generator-Tabs
- Text unter Ladebalken mit aktueller Aufgabe
- Beenden-Button für User-Abbruch
- LOD64 (64x64 Auflösung) für alle Berechnungen
- Auto-Close nach 3 Sekunden → Navigation zu terrain_tab
- Error-Handling mit PyCharm-Integration für Debugging
- DataManager-Speicherung für alle berechneten Werte
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import logging
import traceback

def get_loading_error_decorators():
    """
    Funktionsweise: Lazy Loading von Loading Tab Error Decorators
    Aufgabe: Lädt Core-Generation und Data-Management Decorators
    Return: Tuple von Decorator-Funktionen
    """
    try:
        from gui.error_handler import core_generation_handler, data_management_handler
        return core_generation_handler, data_management_handler
    except ImportError:
        def noop_decorator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        return noop_decorator, noop_decorator

core_generation_handler, data_management_handler = get_loading_error_decorators()


class GeneratorProgressBar(QWidget):
    """
    Funktionsweise: Einfacher Progress-Display für 6 Generator-Schritte
    Aufgabe: Ladebalken, aktueller Generator-Text und Status-Anzeige
    """

    def __init__(self):
        super().__init__()
        self.generator_names = ["Terrain", "Geology", "Settlement", "Weather", "Water", "Biome"]
        self.current_step = 0
        self.setup_ui()

    def setup_ui(self):
        """
        Funktionsweise: Erstellt einfache UI für Progress-Display
        Aufgabe: Titel, Ladebalken, aktueller Text, Status
        """
        layout = QVBoxLayout()
        layout.setSpacing(20)

        # Haupttitel
        self.title_label = QLabel("Generiere Welt...")
        self.title_label.setAlignment(Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)

        # Haupt-Progress-Bar
        self.main_progress = QProgressBar()
        self.main_progress.setRange(0, len(self.generator_names))
        self.main_progress.setValue(0)
        self.main_progress.setTextVisible(True)
        self.main_progress.setFormat("%v / %m Generatoren")
        layout.addWidget(self.main_progress)

        # Aktueller Generator Text
        self.current_task_label = QLabel("Bereit zum Start...")
        self.current_task_label.setAlignment(Qt.AlignCenter)
        font = self.current_task_label.font()
        font.setPointSize(12)
        self.current_task_label.setFont(font)
        layout.addWidget(self.current_task_label)

        # Status-Label
        self.status_label = QLabel("LOD64 (64x64 Auflösung)")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def set_current_generator(self, generator_name: str, step: int):
        """
        Funktionsweise: Setzt aktuellen Generator und aktualisiert Display
        Parameter: generator_name (str), step (int) - Aktueller Generator und Schritt-Nummer
        """
        self.current_step = step
        self.current_task_label.setText(f"Berechne {generator_name}...")
        self.main_progress.setValue(step)

    def mark_generator_complete(self, generator_name: str, success: bool):
        """
        Funktionsweise: Markiert Generator als abgeschlossen
        Parameter: generator_name (str), success (bool) - Generator und Erfolgs-Status
        """
        if success:
            self.current_task_label.setText(f"{generator_name} ✅ abgeschlossen")
            self.main_progress.setValue(self.current_step + 1)
        else:
            self.current_task_label.setText(f"{generator_name} ❌ fehlgeschlagen")
            self.status_label.setText("Generation abgebrochen")

    def mark_all_complete(self):
        """
        Funktionsweise: Markiert alle Generatoren als erfolgreich abgeschlossen
        Aufgabe: Finale Status-Anzeige
        """
        self.title_label.setText("🎉 Welt erfolgreich generiert!")
        self.current_task_label.setText("Alle Generatoren abgeschlossen")
        self.status_label.setText("Wechsel zu Terrain Editor in 3 Sekunden...")
        self.main_progress.setValue(len(self.generator_names))


class LoadingTab(QDialog):
    """
    Funktionsweise: Einfacher Loading-Dialog mit sequenzieller Generator-Ausführung
    Aufgabe: Führt alle 6 Generatoren nacheinander aus und speichert in DataManager
    """

    def __init__(self, data_manager=None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        self.generation_successful = False

        # Setup UI
        self.setup_ui()

        # Auto-start sofort
        QTimer.singleShot(100, self.start_generation)

    def setup_ui(self):
        """
        Funktionsweise: Erstellt UI für Loading-Dialog
        Aufgabe: Progress-Bar und Beenden-Button
        """
        self.setWindowTitle("Welt wird generiert...")
        self.setModal(True)
        self.setFixedSize(400, 200)

        layout = QVBoxLayout()

        # Progress-Bar Widget
        self.progress_widget = GeneratorProgressBar()
        layout.addWidget(self.progress_widget)

        # Spacer
        layout.addStretch()

        # Beenden-Button
        self.quit_button = QPushButton("Beenden")
        self.quit_button.clicked.connect(self.handle_quit_request)
        layout.addWidget(self.quit_button)

        self.setLayout(layout)

        # Window-Close-Verhalten
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint)

    @core_generation_handler("world_generation")
    def start_generation(self):
        """
        Funktionsweise: Startet sequenzielle Generation aller 6 Generatoren mit Error-Protection
        Aufgabe: Führt Generator-Kette durch mit LOD64 und DataManager-Speicherung
        Besonderheit: Error Handler schützt vor Generation-Fehlern und Memory-Issues
        """
        self.logger.info("Starte Welt-Generierung mit LOD64...")
        self.quit_button.setText("Abbrechen")

        # Generator-Liste mit Import-Pfaden
        generators = [
            ("Terrain", "core.terrain_generator"),
            ("Geology", "core.geology_generator"),
            ("Weather", "core.weather_generator"),
            ("Water", "core.water_generator"),
            ("Biome", "core.biome_generator"),
            ("Settlement", "core.settlement_generator")
        ]

        # Führe jeden Generator aus
        for step, (generator_name, import_path) in enumerate(generators):
            try:
                # UI Update
                self.progress_widget.set_current_generator(generator_name, step)
                QApplication.processEvents()  # UI aktualisieren

                # Generator ausführen
                success = self.execute_generator(generator_name, import_path)

                if success:
                    self.progress_widget.mark_generator_complete(generator_name, True)
                    self.logger.info(f"Generator {generator_name} erfolgreich abgeschlossen")
                else:
                    self.progress_widget.mark_generator_complete(generator_name, False)
                    self.logger.error(f"Generator {generator_name} fehlgeschlagen")
                    return  # Abbruch bei Fehler

                # Kurze Pause zwischen Generatoren
                QTimer.singleShot(500, lambda: None)
                QApplication.processEvents()

            except Exception as e:
                # Detaillierter Error für PyCharm
                error_msg = f"Fehler in Generator {generator_name}: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())  # Full Stack-Trace für PyCharm

                self.progress_widget.mark_generator_complete(generator_name, False)
                self.show_error_message(generator_name, str(e))
                return

        # Alle Generatoren erfolgreich
        self.generation_successful = True
        self.progress_widget.mark_all_complete()
        self.quit_button.setText("Fertig")

        # Auto-Close nach 3 Sekunden
        QTimer.singleShot(3000, self.complete_generation)

    @data_management_handler("generator_execution")
    def execute_generator(self, generator_name: str, import_path: str) -> bool:
        """
        Funktionsweise: Führt einen einzelnen Generator aus mit Error-Protection
        Parameter: generator_name (str), import_path (str) - Generator-Info
        Return: bool - True wenn erfolgreich
        Besonderheit: Error Handler schützt vor Data-Management und Import-Fehlern
        """
        try:
            self.logger.debug(f"Importiere Generator: {import_path}")

            # Spezielle Behandlung für Terrain-Generator
            if generator_name == "Terrain":
                return self._execute_terrain_generator()
            elif generator_name == "Geology":
                return self._execute_geology_generator()
            elif generator_name == "Weather":
                return self._execute_weather_generator()
            elif generator_name == "Water":
                return self._execute_water_generator()
            elif generator_name == "Biome":
                return self._execute_biome_generator()
            elif generator_name == "Settlement":
                return self._execute_settlement_generator()
            else:
                self.logger.warning(f"Unbekannter Generator: {generator_name}")
                return False

        except ImportError as e:
            self.logger.error(f"Import-Fehler für {generator_name}: {e}")
            raise ImportError(f"Generator {generator_name} konnte nicht importiert werden: {e}")

        except AttributeError as e:
            self.logger.error(f"Generator-Klasse nicht gefunden in {generator_name}: {e}")
            raise AttributeError(f"Generator-Klasse in {generator_name} nicht gefunden: {e}")

        except Exception as e:
            self.logger.error(f"Ausführungs-Fehler in {generator_name}: {e}")
            raise Exception(f"Generator {generator_name} Ausführung fehlgeschlagen: {e}")

    def _execute_terrain_generator(self) -> bool:
        """
        Funktionsweise: Führt Terrain-Generator mit BaseGenerator-API aus
        Aufgabe: Terrain-Generation mit einheitlicher BaseGenerator-API
        Return: bool - True wenn erfolgreich
        """
        try:
            # Terrain-Generator importieren
            from core.terrain_generator import BaseTerrainGenerator

            # Generator erstellen
            generator = BaseTerrainGenerator(map_seed=42)  # Standard-Seed

            # Progress-Callback für Loading-Tab
            def terrain_progress(step_name, progress_percent, detail_message):
                # Progress an Loading-Tab weiterleiten
                if self.progress_callback:
                    self.progress_callback(step_name, progress_percent, detail_message)

            # Terrain mit neuer BaseGenerator-API generieren
            terrain_data = generator.generate(
                lod="LOD64",
                progress=terrain_progress,
                data_manager=self.data_manager
                # Keine Parameter = Standard-Parameter aus value_default.py
            )

            self.logger.debug("Terrain-Generation erfolgreich abgeschlossen")
            return True

        except Exception as e:
            self.logger.error(f"Terrain-Generator Fehler: {e}")
            raise

    def _execute_geology_generator(self) -> bool:
        """
        Funktionsweise: Führt Geology-Generator mit BaseGenerator-API aus
        Aufgabe: Echte Geology-Generation mit DataManager-Integration
        Return: bool - True wenn erfolgreich
        """
        try:
            # Geology-Generator importieren
            from core.geology_generator import GeologyGenerator

            # Generator erstellen
            generator = GeologyGenerator(map_seed=42)  # Standard-Seed

            # Progress-Callback für Loading-Tab
            def geology_progress(step_name, progress_percent, detail_message):
                # Progress an Loading-Tab weiterleiten
                if hasattr(self, 'progress_widget'):
                    self.progress_widget.current_task_label.setText(detail_message)

            # Geology mit neuer BaseGenerator-API generieren
            result = generator.generate(
                lod="LOD64",
                progress=geology_progress,
                data_manager=self.data_manager
                # Keine Parameter = Standard-Parameter aus value_default.py
            )

            self.logger.debug("Geology-Generation erfolgreich abgeschlossen")
            return True

        except Exception as e:
            self.logger.error(f"Geology-Generator Fehler: {e}")
            raise

    def _execute_weather_generator(self) -> bool:
        """
        Funktionsweise: Führt Weather-Generator mit BaseGenerator-API aus
        Aufgabe: Echte Weather-Generation mit DataManager-Integration und TerrainData-Input
        Return: bool - True wenn erfolgreich
        """
        try:
            # Weather-Generator importieren
            from core.weather_generator import WeatherSystemGenerator

            # Generator erstellen
            generator = WeatherSystemGenerator(map_seed=42)  # Standard-Seed

            # Progress-Callback für Loading-Tab
            def weather_progress(step_name, progress_percent, detail_message):
                # Progress an Loading-Tab weiterleiten
                if hasattr(self, 'progress_widget') and self.progress_widget:
                    self.progress_widget.current_task_label.setText(detail_message)

            # Weather mit neuer BaseGenerator-API generieren
            weather_data = generator.generate(
                lod="LOD64",
                progress=weather_progress,
                data_manager=self.data_manager
                # Keine Parameter = Standard-Parameter aus value_default.py
            )

            self.logger.debug("Weather-Generation erfolgreich abgeschlossen")
            return True

        except Exception as e:
            self.logger.error(f"Weather-Generator Fehler: {e}")
            raise

    def _execute_water_generator(self) -> bool:
        """
        Funktionsweise: Führt Water-Generator mit Standard-Parametern aus (DUMMY)
        Aufgabe: Placeholder für zukünftige Water-Generation
        Return: bool - True (Dummy-Erfolg)
        """
        try:
            from gui.config.value_default import WATER
            import numpy as np

            # Dependencies aus DataManager holen
            heightmap = self.data_manager.get_terrain_data("heightmap")
            precip_map = self.data_manager.get_weather_data("precip_map")
            if heightmap is None or precip_map is None:
                raise Exception("Dependencies für Water-Generator nicht verfügbar")

            # Dummy-Water-Daten erstellen
            map_size = heightmap.shape[0]
            water_map = np.random.uniform(0, 2, (map_size, map_size)).astype(np.float32)
            flow_map = np.random.uniform(0, 5, (map_size, map_size, 2)).astype(np.float32)  # Flow x,y
            soil_moist_map = np.random.uniform(0, 100, (map_size, map_size)).astype(np.float32)
            erosion_map = np.random.uniform(0, 0.5, (map_size, map_size)).astype(np.float32)
            water_biomes_map = np.random.randint(0, 4, (map_size, map_size), dtype=np.uint8)

            # Standard-Parameter
            parameters = {
                'lake_volume_threshold': WATER.LAKE_VOLUME_THRESHOLD["default"],
                'rain_threshold': WATER.RAIN_THRESHOLD["default"],
                'erosion_strength': WATER.EROSION_STRENGTH["default"]
            }

            # Im DataManager speichern
            if self.data_manager:
                self.data_manager.set_water_data("water_map", water_map, parameters)
                self.data_manager.set_water_data("flow_map", flow_map, parameters)
                self.data_manager.set_water_data("soil_moist_map", soil_moist_map, parameters)
                self.data_manager.set_water_data("erosion_map", erosion_map, parameters)
                self.data_manager.set_water_data("water_biomes_map", water_biomes_map, parameters)
                self.logger.debug("Water-Dummy-Daten im DataManager gespeichert")

            return True

        except Exception as e:
            self.logger.error(f"Water-Generator Fehler: {e}")
            raise

    def _execute_biome_generator(self) -> bool:
        """
        Funktionsweise: Führt Biome-Generator mit Standard-Parametern aus (DUMMY)
        Aufgabe: Placeholder für zukünftige Biome-Generation
        Return: bool - True (Dummy-Erfolg)
        """
        try:
            from gui.config.value_default import BIOME
            import numpy as np

            # Dependencies aus DataManager holen
            heightmap = self.data_manager.get_terrain_data("heightmap")
            temp_map = self.data_manager.get_weather_data("temp_map")
            soil_moist_map = self.data_manager.get_water_data("soil_moist_map")
            if heightmap is None or temp_map is None or soil_moist_map is None:
                raise Exception("Dependencies für Biome-Generator nicht verfügbar")

            # Dummy-Biome-Daten erstellen
            map_size = heightmap.shape[0]
            biome_map = np.random.randint(0, 8, (map_size, map_size), dtype=np.uint8)  # 8 Biome-Typen
            biome_map_super = np.random.randint(0, 4, (map_size, map_size), dtype=np.uint8)  # 4 Super-Biome
            super_biome_mask = np.random.choice([True, False], (map_size, map_size))

            # Standard-Parameter
            parameters = {
                'biome_wetness_factor': BIOME.BIOME_WETNESS_FACTOR["default"],
                'biome_temp_factor': BIOME.BIOME_TEMP_FACTOR["default"],
                'sea_level': BIOME.SEA_LEVEL["default"],
                'alpine_level': BIOME.ALPINE_LEVEL["default"]
            }

            # Im DataManager speichern
            if self.data_manager:
                self.data_manager.set_biome_data("biome_map", biome_map, parameters)
                self.data_manager.set_biome_data("biome_map_super", biome_map_super, parameters)
                self.data_manager.set_biome_data("super_biome_mask", super_biome_mask, parameters)
                self.logger.debug("Biome-Dummy-Daten im DataManager gespeichert")

            return True

        except Exception as e:
            self.logger.error(f"Biome-Generator Fehler: {e}")
            raise

    def _execute_settlement_generator(self) -> bool:
        """
        Funktionsweise: Führt Settlement-Generator mit Standard-Parametern aus (DUMMY)
        Aufgabe: Placeholder für zukünftige Settlement-Generation
        Return: bool - True (Dummy-Erfolg)
        """
        try:
            from gui.config.value_default import SETTLEMENT
            import numpy as np

            # Dependencies aus DataManager holen
            heightmap = self.data_manager.get_terrain_data("heightmap")
            water_map = self.data_manager.get_water_data("water_map")
            if heightmap is None or water_map is None:
                raise Exception("Dependencies für Settlement-Generator nicht verfügbar")

            # Dummy-Settlement-Daten erstellen
            map_size = heightmap.shape[0]

            # Listen für Objekte
            settlement_list = [{"x": 32, "y": 32, "type": "village", "population": 500}]  # Dummy-Settlement
            landmark_list = [{"x": 16, "y": 48, "type": "tower", "height": 20}]  # Dummy-Landmark
            roadsite_list = [{"x1": 32, "y1": 32, "x2": 16, "y2": 48}]  # Dummy-Road

            # Maps
            plot_map = np.zeros((map_size, map_size), dtype=np.uint8)
            plot_map[30:35, 30:35] = 1  # Dummy-Settlement-Plot
            civ_map = np.random.uniform(0, 1, (map_size, map_size)).astype(np.float32)

            # Standard-Parameter
            parameters = {
                'settlements': SETTLEMENT.SETTLEMENTS["default"],
                'landmarks': SETTLEMENT.LANDMARKS["default"],
                'roadsites': SETTLEMENT.ROADSITES["default"],
                'plotnodes': SETTLEMENT.PLOTNODES["default"]
            }

            # Im DataManager speichern
            if self.data_manager:
                self.data_manager.set_settlement_data("settlement_list", settlement_list, parameters)
                self.data_manager.set_settlement_data("landmark_list", landmark_list, parameters)
                self.data_manager.set_settlement_data("roadsite_list", roadsite_list, parameters)
                self.data_manager.set_settlement_data("plot_map", plot_map, parameters)
                self.data_manager.set_settlement_data("civ_map", civ_map, parameters)
                self.logger.debug("Settlement-Dummy-Daten im DataManager gespeichert")

            return True

        except Exception as e:
            self.logger.error(f"Settlement-Generator Fehler: {e}")
            raise

    def complete_generation(self):
        """
        Funktionsweise: Schließt Loading ab und wechselt zu terrain_tab
        Aufgabe: Navigation über NavigationManager zu terrain_tab
        """
        if self.generation_successful:
            self.logger.info("Welt-Generierung abgeschlossen, wechsle zu Terrain Editor")

            # Navigation über NavigationManager (falls verfügbar)
            try:
                if hasattr(self.parent(), 'navigation_manager'):
                    self.parent().navigation_manager.navigate_to_tab("terrain")
            except Exception as e:
                self.logger.warning(f"Navigation zu terrain_tab fehlgeschlagen: {e}")

            self.accept()  # Dialog erfolgreich schließen
        else:
            self.reject()  # Dialog mit Fehler schließen

    def handle_quit_request(self):
        """
        Funktionsweise: Behandelt Beenden/Abbrechen-Request
        Aufgabe: Direkte Beendigung ohne Bestätigung
        """
        if self.quit_button.text() == "Fertig":
            self.complete_generation()
        else:
            self.logger.info("Welt-Generierung vom User abgebrochen")
            self.reject()

    def show_error_message(self, generator_name: str, error_message: str):
        """
        Funktionsweise: Zeigt detaillierte Error-Message für PyCharm-Integration
        Parameter: generator_name (str), error_message (str) - Fehler-Information
        Aufgabe: Error-Dialog mit Stack-Trace für Debugging
        """
        # Detaillierte Error-Info für PyCharm
        full_error = f"Generator: {generator_name}\nFehler: {error_message}\n\nStack-Trace siehe PyCharm Console"

        QMessageBox.critical(
            self,
            "Generator-Fehler",
            full_error
        )

    def closeEvent(self, event):
        """
        Funktionsweise: Behandelt Window-Close Event
        Aufgabe: Ruft handle_quit_request auf
        """
        self.handle_quit_request()
        event.accept()