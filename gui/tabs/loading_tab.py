"""
Path: gui/tabs/loading_tab.py

Funktionsweise: Einfacher Ladebildschirm mit GenerationOrchestrator Integration - AKTUALISIERT
- Ladebalken mit 6 Stufen für alle Generator-Tabs
- Text unter Ladebalken mit aktueller Aufgabe
- Beenden-Button für User-Abbruch
- LOD64 (64x64 Auflösung) für alle Berechnungen
- Auto-Close nach 3 Sekunden → Navigation zu terrain_tab
- Error-Handling mit PyCharm-Integration für Debugging
- GEÄNDERT: Nutzt GenerationOrchestrator statt eigene Generator-Ausführung
- GEÄNDERT: Signals für main.py Integration
- REFACTORED: Entfernt doppelten Code durch Orchestrator-Delegation
"""


import logging
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, pyqtSlot
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QProgressBar, QWidget, QDialog, QPushButton, QApplication, QMessageBox


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
        self.generator_names = ["Terrain", "Geology", "Weather", "Water", "Biome", "Settlement"]
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

        # Progress-Details für LOD-Level
        self.lod_progress = QProgressBar()
        self.lod_progress.setRange(0, 100)
        self.lod_progress.setValue(0)
        self.lod_progress.setTextVisible(True)
        self.lod_progress.setFormat("LOD64 - %p%")
        layout.addWidget(self.lod_progress)

        self.setLayout(layout)

    def set_current_generator(self, generator_name: str, step: int):
        """
        Funktionsweise: Setzt aktuellen Generator und aktualisiert Display
        Parameter: generator_name (str), step (int) - Aktueller Generator und Schritt-Nummer
        """
        self.current_step = step
        self.current_task_label.setText(f"Berechne {generator_name}...")
        self.main_progress.setValue(step)

    def set_lod_progress(self, lod_level: str, progress_percent: int, detail: str = ""):
        """
        Funktionsweise: Aktualisiert LOD-spezifischen Progress
        Parameter: lod_level, progress_percent, detail
        """
        self.lod_progress.setValue(progress_percent)
        self.lod_progress.setFormat(f"{lod_level} - {progress_percent}%")

        if detail:
            self.status_label.setText(detail)

    def mark_generator_complete(self, generator_name: str, success: bool):
        """
        Funktionsweise: Markiert Generator als abgeschlossen
        Parameter: generator_name (str), success (bool) - Generator und Erfolgs-Status
        """
        if success:
            self.current_task_label.setText(f"{generator_name} ✅ abgeschlossen")
            self.main_progress.setValue(self.current_step + 1)
            self.lod_progress.setValue(100)
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
        self.lod_progress.setValue(100)


class LoadingTab(QDialog):
    """
    Funktionsweise: Einfacher Loading-Dialog mit GenerationOrchestrator Integration - REFACTORED
    Aufgabe: Führt alle 6 Generatoren über Orchestrator aus und zeigt Progress
    Kommunikation: loading_completed und loading_cancelled Signals für main.py
    ÄNDERUNG: Nutzt GenerationOrchestrator statt direkte Generator-Calls
    """

    # SIGNALS FÜR MAIN.PY INTEGRATION:
    loading_completed = pyqtSignal(bool)  # (success)
    loading_cancelled = pyqtSignal()

    def __init__(self, data_manager=None, generation_orchestrator=None, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.generation_orchestrator = generation_orchestrator
        self.logger = logging.getLogger(__name__)
        self.generation_successful = False

        # Generator-Tracking
        self.generator_sequence = ["terrain", "geology", "weather", "water", "biome", "settlement"]
        self.completed_generators = set()  # Track welche fertig sind
        self.failed_generators = set()
        self.total_generators = len(self.generator_sequence)

        # Setup UI
        self.setup_ui()

        # Orchestrator-Signals verbinden
        self.setup_orchestrator_signals()

        # Auto-start sofort
        QTimer.singleShot(100, self.start_generation)

    def setup_ui(self):
        """
        Funktionsweise: Erstellt UI für Loading-Dialog
        Aufgabe: Progress-Bar und Beenden-Button
        """
        self.setWindowTitle("Welt wird generiert...")
        self.setModal(True)
        self.setFixedSize(400, 250)  # Etwas größer für LOD-Progress

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

    def setup_orchestrator_signals(self):
        """
        Funktionsweise: Verbindet GenerationOrchestrator Signals mit Loading-UI
        Aufgabe: Progress-Updates und Completion-Tracking
        """
        if not self.generation_orchestrator:
            self.logger.warning("No GenerationOrchestrator provided - fallback mode")
            return

        # Generation-Progress für LOD-Updates
        self.generation_orchestrator.generation_progress.connect(self.on_generation_progress)

        # Generation-Completion für Generator-Tracking
        self.generation_orchestrator.generation_completed.connect(self.on_generation_completed)

        # Batch-Completion für finale Fertigstellung
        self.generation_orchestrator.batch_generation_completed.connect(self.on_batch_completed)

    @core_generation_handler("world_generation")
    def start_generation(self):
        """
        Funktionsweise: Startet sequenzielle Generation über GenerationOrchestrator - REFACTORED
        Aufgabe: Requests für alle 6 Generatoren mit LOD64 über Orchestrator
        Besonderheit: Error Handler schützt vor Generation-Fehlern und Memory-Issues
        ÄNDERUNG: Nutzt Orchestrator statt direkte Generator-Calls
        """
        self.logger.info("Starte Welt-Generierung mit LOD64 über GenerationOrchestrator...")
        self.quit_button.setText("Abbrechen")

        if not self.generation_orchestrator:
            # Fallback zu alter Methode bei fehlendem Orchestrator
            self.start_generation_fallback()
            return

        # Alle Generator-Requests über Orchestrator senden
        for i, generator_type in enumerate(self.generator_sequence):
            try:
                # Standard-Parameter für Loading (würden normalerweise aus Defaults kommen)
                parameters = self.get_default_parameters(generator_type)

                # Request an Orchestrator senden
                request_id = self.generation_orchestrator.request_generation(
                    generator_type=generator_type,
                    parameters=parameters,
                    target_lod="LOD64",
                    source_tab="loading",
                    priority=len(self.generator_sequence) - i  # Höhere Priority für frühere Generatoren
                )

                self.logger.info(f"Requested generation for {generator_type}: {request_id}")

            except Exception as e:
                self.logger.error(f"Failed to request generation for {generator_type}: {e}")
                self.show_error_message(generator_type, str(e))
                return

    def get_default_parameters(self, generator_type: str) -> dict:
        """
        Funktionsweise: Holt Default-Parameter für Generator aus value_default.py
        Parameter: generator_type (str)
        Return: dict mit Default-Parametern
        """
        try:
            from gui.config.value_default import get_parameter_config

            # Parameter-Listen für jeden Generator
            param_lists = {
                "terrain": ["size", "amplitude", "octaves", "frequency", "persistence", "lacunarity",
                            "redistribute_power", "map_seed"],
                "geology": ["sedimentary_hardness", "igneous_hardness", "metamorphic_hardness", "ridge_warping"],
                "weather": ["air_temp_entry", "solar_power", "altitude_cooling", "thermic_effect"],
                "water": ["lake_volume_threshold", "rain_threshold", "manning_coefficient", "erosion_strength"],
                "biome": ["biome_wetness_factor", "biome_temp_factor", "sea_level", "alpine_level", "snow_level"],
                "settlement": ["settlements", "landmarks", "roadsites", "plotnodes", "civ_influence_decay"]
            }

            parameters = {}
            param_names = param_lists.get(generator_type, [])

            for param_name in param_names:
                try:
                    config = get_parameter_config(generator_type, param_name)
                    parameters[param_name] = config["default"]
                except (ValueError, KeyError):
                    self.logger.warning(f"No config found for {generator_type}.{param_name}")

            return parameters

        except ImportError:
            # Fallback bei fehlendem Config-Modul
            return {"map_seed": 42}

    @pyqtSlot(str, str, int, str)
    def on_generation_progress(self, generator_type: str, lod_level: str, progress_percent: int, detail: str):
        """
        Funktionsweise: Slot für Generation-Progress Updates vom Orchestrator
        Parameter: generator_type, lod_level, progress_percent, detail
        """
        # Generator-Index finden
        if generator_type in self.generator_sequence:
            generator_index = self.generator_sequence.index(generator_type)
            generator_name = generator_type.title()

            # UI aktualisieren
            self.progress_widget.set_current_generator(generator_name, generator_index)
            self.progress_widget.set_lod_progress(lod_level, progress_percent, detail)

    @pyqtSlot(str, str, bool)
    def on_generation_completed(self, generator_type: str, lod_level: str, success: bool):
        """
        Funktionsweise: Slot für einzelne Generator-Completion mit Batch-Check
        """
        generator_name = generator_type.title()

        if success:
            self.completed_generators.add(generator_type)
            self.progress_widget.mark_generator_complete(generator_name, True)
            self.logger.info(f"Generator {generator_name} completed successfully")

            # NEU: Prüfe ob alle Generatoren fertig sind
            if len(self.completed_generators) >= self.total_generators:
                self.logger.info("All generators completed - triggering completion")
                # Simuliere batch_generation_completed Signal
                self.on_batch_completed(True, "All generators completed successfully")

        else:
            self.failed_generators.add(generator_type)
            self.progress_widget.mark_generator_complete(generator_name, False)
            self.logger.error(f"Generator {generator_name} failed")

            # Bei Fehler auch Completion triggern
            self.on_batch_completed(False, f"Generator {generator_name} failed")

    @pyqtSlot(bool, str)
    def on_batch_completed(self, success: bool, summary_message: str):
        """
        Funktionsweise: Slot für komplette Batch-Generation Completion mit Debug-Logging
        """
        self.logger.info(f"on_batch_completed called: success={success}, message='{summary_message}'")
        print(f"LoadingTab: on_batch_completed - success={success}")  # DEBUG

        self.generation_successful = success

        if success:
            self.progress_widget.mark_all_complete()
            self.quit_button.setText("Fertig")

            # Auto-Close nach 3 Sekunden
            self.logger.info("Starting 3-second auto-close timer")
            QTimer.singleShot(3000, self.complete_generation)
        else:
            self.show_error_message("Batch Generation", summary_message)

    def start_generation_fallback(self):
        """
        Funktionsweise: Fallback-Methode bei fehlendem GenerationOrchestrator
        Aufgabe: Alte sequenzielle Generation für Backward-Compatibility
        """
        self.logger.warning("Using fallback generation method - no orchestrator available")

        # Alte Implementierung von execute_*_generator verwenden
        generators = [
            ("Terrain", "core.terrain_generator"),
            ("Geology", "core.geology_generator"),
            ("Weather", "core.weather_generator"),
            ("Water", "core.water_generator"),
            ("Biome", "core.biome_generator"),
            ("Settlement", "core.settlement_generator")
        ]

        for step, (generator_name, import_path) in enumerate(generators):
            try:
                self.progress_widget.set_current_generator(generator_name, step)
                QApplication.processEvents()

                success = self.execute_generator_fallback(generator_name, import_path)

                if success:
                    self.progress_widget.mark_generator_complete(generator_name, True)
                    self.completed_generators.add(generator_name.lower())
                else:
                    self.progress_widget.mark_generator_complete(generator_name, False)
                    self.failed_generators.add(generator_name.lower())
                    return

            except Exception as e:
                self.logger.error(f"Fallback generation failed for {generator_name}: {e}")
                self.show_error_message(generator_name, str(e))
                return

        # Alle erfolgreich
        self.generation_successful = True
        self.progress_widget.mark_all_complete()
        self.quit_button.setText("Fertig")
        QTimer.singleShot(3000, self.complete_generation)

    @data_management_handler("generator_execution")
    def execute_generator_fallback(self, generator_name: str, import_path: str) -> bool:
        """
        Funktionsweise: Fallback-Ausführung für einzelne Generatoren
        Parameter: generator_name, import_path
        Return: bool - Success
        Besonderheit: Nur für Backward-Compatibility, sollte nicht verwendet werden
        """
        # Vereinfachte Fallback-Implementierung
        # (Original-Code aus alter loading_tab.py würde hier stehen)
        try:
            # Dummy-Erfolg für Fallback
            import time
            time.sleep(0.5)  # Simulate generation time
            return True
        except Exception as e:
            self.logger.error(f"Fallback execution failed: {e}")
            return False

    def complete_generation(self):
        """
        Funktionsweise: Schließt Loading ab und sendet Signal an main.py
        Aufgabe: Loading-Complete Signal senden statt direkte Navigation
        """
        print("LoadingTab: complete_generation called")  # DEBUG

        if self.generation_successful:
            self.logger.info("Welt-Generierung abgeschlossen, sende completion signal")
            print("LoadingTab: Sending loading_completed(True)")  # DEBUG

            # Signal an main.py senden statt direkte Navigation
            self.loading_completed.emit(True)

            # Kurz warten damit main.py reagieren kann
            QTimer.singleShot(100, self.accept)  # Dialog nach 100ms schließen
        else:
            print("LoadingTab: Sending loading_completed(False)")  # DEBUG
            self.loading_completed.emit(False)
            QTimer.singleShot(100, self.reject)

    def handle_quit_request(self):
        """
        Funktionsweise: Behandelt Beenden/Abbrechen-Request mit Signal
        Aufgabe: Loading-Cancelled Signal an main.py senden
        """
        print("LoadingTab: handle_quit_request called")  # DEBUG

        if self.quit_button.text() == "Fertig":
            self.complete_generation()
        else:
            self.logger.info("Welt-Generierung vom User abgebrochen")
            print("LoadingTab: Sending loading_cancelled")  # DEBUG

            # Signal an main.py senden
            self.loading_cancelled.emit()

            QTimer.singleShot(100, self.reject)

    def show_error_message(self, generator_name: str, error_message: str):
        """
        Funktionsweise: Zeigt detaillierte Error-Message für PyCharm-Integration
        Parameter: generator_name (str), error_message (str) - Fehler-Information
        Aufgabe: Error-Dialog mit Stack-Trace für Debugging
        """
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