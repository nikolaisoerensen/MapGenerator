"""
Path: gui/tabs/base_tab.py

BaseMapTab ist die fundamentale Basis-Klasse für alle spezialisierten Map-Editor Tabs.
Sie besitzt KEIN eigenes Fenster-Layout mehr - stattdessen liefert sie drei
eigenständige Top-Level-Widgets (viewport_widget, parameter_widget,
statistics_widget), die das MapEditorWindow-Shell-Layout in seine drei
globalen Spalten (Viewport-Stack / Parameter-Stack / Statistics-Stack) einhängt.

Kernverantwortlichkeiten:
- viewport_widget: 2D/3D Display-Stack mit Toggle-Controls und Fallback-Management
- parameter_widget: Parameter-UI-Controls als Proxy zum ParameterManager
- statistics_widget: Statistics-Anzeige (aktuell noch Platzhalter für die meisten
  Generatoren, siehe create_statistics_controls())
- Display-Update-Koordination ohne eigene Business-Logic

Manager-Integration als Proxy:
- ParameterManager: UI-Parameter ↔ Storage-Parameter Synchronisation
- GenerationOrchestrator: Generation-Requests mit Parameter-Weiterleitung
- DataLODManager: Display-Data-Retrieval für UI-Updates
- NavigationManager: Tab-Navigation und Window-Geometry

Extensibility für Sub-Classes:
- Required: generator_type, required_dependencies, create_parameter_controls()
- Optional: create_visualization_controls(), update_display_mode(), check_input_dependencies()
"""

from PyQt6.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QTimer
)
from PyQt6.QtGui import QFont, QPalette
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QFrame, QSizePolicy, QLabel, QVBoxLayout,
    QScrollArea, QSplitter, QGroupBox, QStackedWidget, QRadioButton,
    QButtonGroup, QProgressBar
)

import logging
import time
import hashlib
import weakref
import numpy as np
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

from gui.widgets.widgets import BaseButton, StatusIndicator, DisplayWrapper, NavigationPanel

# Map Display Import mit Fallback
try:
    from gui.widgets.map_display_2d import MapDisplay2D
    MAP_DISPLAY_2D_AVAILABLE = True
except ImportError:
    MAP_DISPLAY_2D_AVAILABLE = False
    print("Warning: MapDisplay2D not available")

try:
    from gui.widgets.map_display_3d import MapDisplay3DWidget
    MAP_DISPLAY_3D_AVAILABLE = True
except ImportError:
    MAP_DISPLAY_3D_AVAILABLE = False
    print("Warning: MapDisplay3DWidget not available")

def get_error_handler():
    """Zentrale Error-Handler Import mit Fallback"""
    try:
        from gui.utils.error_handler import error_handler
        return error_handler
    except ImportError:
        def noop_decorator(func):
            return func
        return noop_decorator

error_handler = get_error_handler()


@dataclass(frozen=True)
class Overlay:
    """
    Ein Vektor-Overlay als WERT, kein Aufruf (docs/spezifikation/15_ANZEIGE.md Punkt 2).
    Ein Reiter meldet nur Name/Sichtbarkeit/Daten an BaseMapTab._push_overlays();
    WIE gezeichnet wird (2D per overlay_*(), 3D per RGBA-Skin), entscheidet
    ausschliesslich das Register unten - der Reiter kennt den Unterschied
    nicht mehr und kann ihn deshalb nicht mehr falsch treffen.

    `sichtbar=False` ist ein vollwertiger Zustand, kein Weglassen: erst
    dadurch kann das Register eine liegengebliebene 3D-Textur abraeumen
    (siehe `clear_river_overlay`, bislang der einzige Weg dafuer).
    """
    name: str
    sichtbar: bool
    daten: Any = None


# ---------------------------------------------------------------------------
# Overlay-Register (docs/spezifikation/15_ANZEIGE.md Punkt 3): "privat zur implementation
# von BaseMapTab, aber mit eigenen Tests" - deshalb hier als Modulfunktionen
# statt als Methoden, unabhaengig von jedem Reiter pruefbar (siehe
# tests/smoke_test_push_overlays.py), aber nicht als oeffentliches Interface
# gedacht. Haelt je Overlay-Name BEIDE Wege nebeneinander - das ist die
# locality, um die es in der Spezifikation geht. Ein Name ohne Eintrag ist in
# _push_overlays() unten ein Fehler, keine stille Auslassung.
# ---------------------------------------------------------------------------

def _siedlungen_2d(display, overlay: "Overlay") -> None:
    """
    2D-Adapter fuer das Siedlungs-Overlay (Staedte/Landmarken/Roadsites) -
    ruft MapDisplay2D.overlay_settlements() genau wie bisher SettlementTab
    und BiomeTab je einzeln. sichtbar=False zeichnet nichts; die 2D-Anzeige
    baut ihr Bild bei jedem update_display() ohnehin neu auf, ein separates
    Abraeumen braucht es dort nicht (anders als in 3D, siehe _siedlungen_3d).
    """
    if not overlay.sichtbar or not hasattr(display, "overlay_settlements"):
        return
    settlements, landmarks, roadsites = overlay.daten
    if settlements or landmarks or roadsites:
        display.overlay_settlements(settlements, landmarks, roadsites)


def _siedlungen_3d(display, overlay: "Overlay") -> None:
    """
    3D-Adapter: RGBA-Skin per `rasterize_settlements_rgba()` auf das Gelaende,
    derselbe Weg wie `SettlementTab.apply_3d_overlays()` (laut CLAUDE.md das
    Vorbild fuer "in 2D sichtbar -> auch in 3D") - kein neuer GLSL-Code.

    Nutzt `display.heightmap` (von update_heightmap() gesetzt) statt eines
    eigenen DataLODManager-Zugriffs, damit dieser Adapter ohne Reiter-Kontext
    auskommt - `_push_data_to_current_display()` hat die 3D-Heightmap zu
    diesem Zeitpunkt bereits gepusht (laeuft laut Vorbild-Reihenfolge immer
    VOR den Overlays, siehe update_biome_display()).

    sichtbar=False raeumt die Textur ueber set_layer_visibility() ab, ohne
    sie neu zu bauen.

    Fingerabdruck-Cache auf `display` selbst (uebersteht Aufrufe, weil das
    Display-Objekt pro Reiter dauerhaft existiert, siehe `display.heightmap`):
    _push_overlays() laeuft bei JEDEM Haken-/Anzeige-Wechsel neu, auch wenn nur
    Fluesse betroffen sind. Ohne Fingerabdruck baut rasterize_settlements_rgba()
    dann bei jedem irrelevanten Klick ein frisches Array, und
    MapDisplay3D._overlay_cache_pruefen() cacht ueber die Objekt-Identitaet des
    Arrays - ein frisches Array sieht dort immer wie "geaendert" aus, also
    volles Textur-Upload bei jedem Klick (Nachtrag 2026-09-16, urspruenglich in
    BiomeTab.apply_overlays() behoben, jetzt hierher gezogen, weil dieser
    Adapter fuer alle Reiter gilt statt nur fuer Biome).
    """
    if not hasattr(display, "set_layer_visibility"):
        return
    settlements, landmarks, roadsites = overlay.daten
    hat_inhalt = bool(overlay.sichtbar and (settlements or landmarks or roadsites))
    if hat_inhalt and hasattr(display, "update_overlay_data"):
        heightmap = getattr(display, "heightmap", None)
        if heightmap is not None:
            fingerabdruck = (id(settlements), id(landmarks), id(roadsites), heightmap.shape[0])
            if fingerabdruck != getattr(display, "_siedlungen_overlay_fingerabdruck", None):
                from gui.widgets.overlay_rasterizer import rasterize_settlements_rgba
                rgba = rasterize_settlements_rgba(
                    settlements or [], landmarks or [], roadsites or [], [], [],
                    map_size=heightmap.shape[0], resolution=heightmap.shape[0])
                display.update_overlay_data("settlement", "uebersicht", rgba)
                display._siedlungen_overlay_fingerabdruck = fingerabdruck
    display.set_layer_visibility("settlement", "uebersicht", hat_inhalt)


def _fluesse_zeichnen(display, overlay: "Overlay") -> None:
    """
    Gemeinsamer Adapter fuer 2D UND 3D: `overlay_river_generations()` und
    `clear_river_overlay()` existieren unter DEMSELBEN Namen und derselben
    Signatur auf MapDisplay2D und MapDisplay3DWidget (2026-08-24 fuer 3D
    nachgebaut) - hier gibt es die zwei Dialekte aus Punkt 3 nicht, deshalb
    eine Funktion fuer beide Registerseiten statt zwei fast identischer.

    `overlay.daten` ist entweder direkt die Generationskarte (Baeche/Mikro
    immer aus, so ruft es BiomeTab - dort gibt es keinen Mikro-Haken) oder
    ein Tupel `(generation_map, zeige_mikro)` - diese zweite Form nutzte bis
    2026-09-23 RiverTab (Ticket #11, `mikro_checkbox`); dessen eigene Ansicht
    "Flussnetz (Generationen)" ist seither entfernt, BiomeTab ist der einzige
    verbliebene Aufrufer und ruft ausschliesslich die erste Form. Die
    Tupel-Form bleibt hier stehen, weil sie harmlos allgemein ist und ein
    zukuenftiger Aufrufer sie wieder nutzen kann.
    """
    if overlay.sichtbar:
        daten = overlay.daten
        if isinstance(daten, tuple):
            generation_map, zeige_mikro = daten
        else:
            generation_map, zeige_mikro = daten, False
        if generation_map is not None and hasattr(display, "overlay_river_generations"):
            display.overlay_river_generations(np.asarray(generation_map), zeige_mikro=zeige_mikro)
    elif hasattr(display, "clear_river_overlay"):
        # Im 3D bleibt eine einmal gesetzte Textur liegen, bis sie
        # abgeschaltet wird (2D zeichnet ohnehin neu, siehe _siedlungen_2d).
        display.clear_river_overlay()


_OVERLAY_REGISTER: Dict[str, Dict[str, Callable[[Any, "Overlay"], None]]] = {
    "siedlungen": {"2d": _siedlungen_2d, "3d": _siedlungen_3d},
    "fluesse": {"2d": _fluesse_zeichnen, "3d": _fluesse_zeichnen},
}


class BaseMapTab(QWidget):
    """
    Basis-Klasse für alle Map-Editor Tabs mit einheitlicher Manager-Integration
    und standardisiertem 70/30 Layout-System.
    """

    # Outgoing Signals (UI → Managers)
    generate_requested = pyqtSignal(str, dict)  # generator_type, parameters
    parameter_ui_changed = pyqtSignal(str, str, object)  # generator_type, param_name, value
    display_mode_changed = pyqtSignal(str)  # display_mode
    view_switched = pyqtSignal(str)  # "2d"/"3d" - siehe switch_view(), von map_editor.py
    # genutzt, um den 2D/3D-Modus tabübergreifend als globale Präferenz zu behandeln
    # (User-Report: "beim Wechsel des Tabs ist man vom 3D Modus wieder im 2D Modus").

    def __init__(self, data_lod_manager=None, parameter_manager=None,
                 navigation_manager=None, shader_manager=None, generation_orchestrator=None):
        super().__init__()

        # Manager-Referenzen (können None sein für Fallback-Verhalten)
        self.data_lod_manager = data_lod_manager
        self.parameter_manager = parameter_manager
        self.navigation_manager = navigation_manager
        self.shader_manager = shader_manager
        self.generation_orchestrator = generation_orchestrator

        # Core Attributes - müssen von Sub-Classes gesetzt werden
        self.generator_type = getattr(self, 'generator_type', 'unknown')
        self.required_dependencies = getattr(self, 'required_dependencies', [])

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Layout-System
        self.layout_config = LayoutConfiguration()

        # UI Components (werden in setup_ui() initialisiert)
        # viewport_widget/parameter_widget sind die beiden eigenständigen
        # Top-Level-Widgets, die das MapEditorWindow-Shell-Layout in Spalte 2
        # (Viewport-Stack) bzw. Spalte 3 (Parameter-Stack) einhängt. BaseMapTab
        # besitzt selbst kein Splitter-Layout mehr - canvas_container/control_widget
        # bleiben als rückwärtskompatible Aliase erhalten, da Sub-Classes darüber
        # ihre Controls einhängen (self.control_panel.layout().addWidget(...)).
        self.viewport_widget = None
        self.canvas_container = None
        self.parameter_widget = None
        self.control_panel = None
        self.control_widget = None
        self.statistics_widget = None
        self.navigation_panel = None
        self.status_display = None

        # Display-System
        self.display_stack = None
        self.map_display_2d = None
        self.map_display_3d = None
        self.current_view = "2d"
        self.view_2d_button = None
        self.view_3d_button = None
        self.visualization_controls = None

        # Generation-System (vereinfacht)
        self.generation_active = False
        self.auto_simulation_enabled = False

        # Setup-Sequenz
        self.setup_ui()
        self.setup_manager_connections()

    def setup_ui(self):
        """Hauptmethode für UI-Setup mit modularer Struktur"""
        self.logger.debug("Setting up UI components")

        try:
            self._create_viewport_container()
            self._create_canvas_area()
            self._create_parameter_container()
            self._create_control_panel()
            self._create_navigation_panel()
            self._create_status_display()
            self._create_statistics_widget()

            # Sub-Class spezifische Controls
            self.create_parameter_controls()

            # DANACH, NICHT IN JEDEM REITER EINZELN. Die neun Reiter bauen ihre
            # Regler auf drei verschiedene Weisen; die Sperre hier anzusetzen
            # ist die einzige Stelle, an der sie garantiert alle erreicht.
            self._stillgelegte_regler_sperren()

            self.logger.debug("UI setup completed successfully")

        except Exception as e:
            self.logger.error(f"UI setup failed: {e}")
            self._create_fallback_ui(str(e))

    def _create_viewport_container(self):
        """
        Erstellt den eigenständigen Viewport-Container (Spalte 2 Content).
        Wird vom MapEditorWindow-Shell-Layout in einen gemeinsamen
        QStackedWidget eingehängt - BaseMapTab selbst legt kein eigenes
        Splitter/Fenster-Layout mehr an.
        """
        self.viewport_widget = QWidget()
        self.canvas_container = self.viewport_widget  # Backward-kompatibler Alias

    def _create_parameter_container(self):
        """
        Erstellt den eigenständigen Parameter-Container (Spalte 3 Content).
        Die Breite wird vom Shell-Layout (fixe Spalte 3) vorgegeben, daher
        legt sich dieses Widget selbst keine feste Breite mehr fest. Kein
        eigener Rahmen - die Spalte 3 QTabWidget-Chrome übernimmt die
        visuelle Abgrenzung.
        """
        self.parameter_widget = QWidget()
        self.control_widget = self.parameter_widget  # Backward-kompatibler Alias

    def _create_statistics_widget(self):
        """
        KANN von Sub-Classes über create_statistics_controls() befüllt werden.
        Bis Sub-Classes ihre Statistics-Anzeigen aus dem Parameter-Panel
        herauslösen, zeigt dieser Tab einen Platzhalter (Statistics stecken
        aktuell noch mit im Parameter-Panel, siehe create_parameter_controls()).
        """
        self.statistics_widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.create_statistics_controls(layout)
        self.statistics_widget.setLayout(layout)

    def create_statistics_controls(self, layout: QVBoxLayout):
        """
        KANN von Sub-Classes überschrieben werden, um das Statistics-Tab in
        Spalte 3 zu befüllen. Default: Platzhalter-Hinweis.
        """
        placeholder = QLabel("Statistics view not yet separated from Parameter panel for this generator")
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #7f8c8d; padding: 20px;")
        layout.addWidget(placeholder)

    def _create_canvas_area(self):
        """Erstellt Canvas mit 2D/3D Display-Stack"""
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(5, 5, 5, 5)

        # View Toggle Buttons
        button_layout = QHBoxLayout()

        self.view_2d_button = BaseButton("2D View", "primary")
        self.view_2d_button.clicked.connect(lambda: self.switch_view("2d"))
        button_layout.addWidget(self.view_2d_button)

        self.view_3d_button = BaseButton("3D View", "secondary")
        self.view_3d_button.clicked.connect(lambda: self.switch_view("3d"))
        button_layout.addWidget(self.view_3d_button)

        button_layout.addStretch()

        # Visualization Controls (überschreibbar von Sub-Classes)
        self.visualization_controls = self.create_visualization_controls()
        if self.visualization_controls:
            button_layout.addWidget(self.visualization_controls)

        canvas_layout.addLayout(button_layout)

        # Display Stack
        self.display_stack = QStackedWidget()
        self._create_display_widgets()
        canvas_layout.addWidget(self.display_stack)

        self.canvas_container.setLayout(canvas_layout)

    def _create_display_widgets(self):
        """Erstellt 2D und 3D Display-Widgets mit Fallback"""
        # 2D Display
        if MAP_DISPLAY_2D_AVAILABLE:
            try:
                display_2d = MapDisplay2D()
                self.map_display_2d = DisplayWrapper(display_2d)
            except Exception as e:
                self.logger.error(f"2D Display creation failed: {e}")
                self.map_display_2d = self._create_fallback_display("2D Display\n(Creation failed)")
        else:
            self.map_display_2d = self._create_fallback_display("2D Display\n(Not available)")

        self.display_stack.addWidget(self.map_display_2d.display)

        # 3D Display
        if MAP_DISPLAY_3D_AVAILABLE:
            try:
                display_3d = MapDisplay3DWidget()
                self.map_display_3d = DisplayWrapper(display_3d)
            except Exception as e:
                self.logger.error(f"3D Display creation failed: {e}")
                self.map_display_3d = self._create_fallback_display("3D Display\n(Creation failed)")
        else:
            self.map_display_3d = self._create_fallback_display("3D Display\n(Not available)")

        self.display_stack.addWidget(self.map_display_3d.display)

    def _create_fallback_display(self, text: str) -> DisplayWrapper:
        """Erstellt Fallback-Display für fehlerhafte Display-Creation"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; color: #6c757d;")
        return DisplayWrapper(label)

    def _create_control_panel(self):
        """
        Erstellt scrollbares Control Panel.
        ScrollArea, äußeres Layout und Panel-Inhalts-Layout werden als Attribute
        gehalten: die Ownership-Kette bleibt damit unabhängig vom
        Garbage-Collector-Timing stabil, und Sub-Classes erhalten mit
        control_panel_content_layout einen direkten Zugriff auf das
        Inhalts-Layout, der nicht vom Qt-Layout-Lookup abhängt.
        """
        self.control_panel_layout = QVBoxLayout()
        self.control_panel_layout.setContentsMargins(5, 5, 5, 5)

        # Scrollable Area für Parameter
        self.control_scroll_area = QScrollArea()
        self.control_scroll_area.setWidgetResizable(True)
        self.control_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.control_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.control_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Control Panel Content
        self.control_panel = QWidget()
        self.control_panel_content_layout = QVBoxLayout()
        self.control_panel_content_layout.setContentsMargins(5, 5, 5, 5)
        self.control_panel_content_layout.setSpacing(10)
        self.control_panel.setLayout(self.control_panel_content_layout)

        self.control_scroll_area.setWidget(self.control_panel)
        self.control_panel_layout.addWidget(self.control_scroll_area)

        if self.control_widget.layout() is None:
            self.control_widget.setLayout(self.control_panel_layout)
        else:
            self.control_widget.layout().addLayout(self.control_panel_layout)

    def _create_navigation_panel(self):
        """Erstellt Navigation Panel (nicht scrollbar)"""
        try:
            if self.navigation_manager:
                self.navigation_panel = NavigationPanel(self.navigation_manager)
                self.control_widget.layout().addWidget(self.navigation_panel)
        except Exception as e:
            self.logger.error(f"Navigation panel creation failed: {e}")

    def _create_status_display(self):
        """Erstellt Status-Display"""
        try:
            self.status_display = StatusIndicator("System Status")
            self.status_display.set_success("Ready")
            self.control_panel_content_layout.addWidget(self.status_display)
        except Exception as e:
            self.logger.error(f"Status display creation failed: {e}")

    def _create_fallback_ui(self, error_message: str):
        """Erstellt minimale Fallback-UI bei Setup-Fehlern"""
        try:
            fallback_layout = QVBoxLayout()
            error_label = QLabel(f"UI Setup Error:\n{error_message}")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            error_label.setStyleSheet("color: red; padding: 20px; font-size: 12px;")
            fallback_layout.addWidget(error_label)
            self.setLayout(fallback_layout)
        except Exception as fehler:
            # Stiller Ersatzpfad: das ist bereits der Fallback fuer einen
            # gescheiterten Tab-Aufbau - scheitert auch ER, bleibt der Tab
            # komplett leer und niemand erfaehrt, dass ueberhaupt zwei
            # Fehler aufgetreten sind (urspruenglicher Setup-Fehler UND
            # dieser). CLAUDE.md "jeder stille Rueckfall braucht eine laute
            # Logzeile".
            self.logger.error(
                "Auch Fallback-UI fehlgeschlagen (%s) - urspruenglicher "
                "Fehler war: %s. Tab bleibt leer.", fehler, error_message)

    def setup_manager_connections(self):
        """
        Verbindet Signals mit verfügbaren Managern.
        Jeder Manager wird einzeln abgesichert verbunden, damit ein
        fehlgeschlagener Anschluss die übrigen Verbindungen nicht verhindert.
        Fehlgeschlagene Verbindungen werden als INFO geloggt und der Tab
        läuft ohne den betreffenden Manager weiter.
        """
        # ParameterManager Connections
        if self.parameter_manager:
            try:
                self.parameter_manager.parameter_changed.connect(self.on_parameter_changed)
                self.parameter_ui_changed.connect(self.parameter_manager.set_parameter)
            except (AttributeError, TypeError) as e:
                self.logger.info(f"ParameterManager connection not established: {e}")

        # DataLODManager Connections
        if self.data_lod_manager:
            try:
                self.data_lod_manager.data_updated.connect(self.on_data_updated)
            except (AttributeError, TypeError) as e:
                self.logger.info(f"DataLODManager connection not established: {e}")

        # GenerationOrchestrator Connections
        if self.generation_orchestrator:
            try:
                self.generate_requested.connect(self.generation_orchestrator.request_generation)
                self.generation_orchestrator.generation_started.connect(self.on_generation_started)
                self.generation_orchestrator.generation_completed.connect(self.on_generation_completed)
                self.generation_orchestrator.generation_progress.connect(self.on_generation_progress)
            except (AttributeError, TypeError) as e:
                self.logger.info(f"GenerationOrchestrator connection not established: {e}")

    # =============================================================================
    # DISPLAY MANAGEMENT
    # =============================================================================

    @pyqtSlot(str)
    def switch_view(self, view_type: str):
        """Wechselt zwischen 2D und 3D Ansicht"""
        # Altes Display deaktivieren
        current_display = self.get_current_display()
        if current_display and hasattr(current_display, 'set_active'):
            try:
                current_display.set_active(False)
            except Exception as e:
                self.logger.debug(f"Display deactivation warning: {e}")

        # View umschalten
        self.current_view = view_type

        if view_type == "2d":
            self.display_stack.setCurrentIndex(0)
            self._update_button_styles("2d")
        else:
            self.display_stack.setCurrentIndex(1)
            self._update_button_styles("3d")

        # Neues Display aktivieren
        new_display = self.get_current_display()
        if new_display and hasattr(new_display, 'set_active'):
            try:
                new_display.set_active(True)
            except Exception as e:
                self.logger.debug(f"Display activation warning: {e}")

        # Display aktualisieren
        self.update_display_mode()

        # NACHZIEHENDES NEUZEICHNEN FUER DIE 3D-ANSICHT (2026-08-13,
        # docs/OFFENE_PUNKTE.md 6.25). Nutzerbefund: "klicke auf 3D-View (hier
        # wird dann nie automatisch schon 3D geladen oder es wird nicht
        # dargestellt, erst wenn ich mich bewege ist es zu sehen)".
        #
        # `update_heightmap()` ruft am Ende zwar bereits `self.update()` - aber
        # zu diesem Zeitpunkt hat Qt den `setCurrentIndex()`-Wechsel oben noch
        # nicht verarbeitet, das QOpenGLWidget ist also formal noch nicht
        # sichtbar. Qt verwirft Neuzeichnen-Anforderungen fuer unsichtbare
        # Widgets, die Anforderung verpufft. Das erste `paintGL()` kommt dann
        # erst durch das naechste Ereignis - und das ist in der Praxis die
        # erste Mausbewegung des Nutzers, genau wie berichtet.
        #
        # Hier, nach `update_display_mode()`, ist der Stapelwechsel
        # abgeschlossen.
        #
        # NACHTRAG 2026-08-16: `update()` allein reichte nicht - Nutzerbefund
        # weiterhin "Wege werden gezeichnet, das Mesh aber nicht, erst nach
        # Drehen/Bewegen". `update()` PLANT ein Neuzeichnen nur fuer die
        # naechste Gelegenheit, in der Qt seine Ereignisschlange abarbeitet -
        # kommt vorher (oder durch einen anderen Timer/Slot) noch etwas
        # dazwischen, wird die Anforderung mit dem naechsten `update()`
        # zusammengefasst, ohne dass zwischendurch tatsaechlich gezeichnet
        # wurde. `repaint()` erzwingt das Zeichnen STATTDESSEN sofort und
        # synchron, an Ort und Stelle - genau das fehlende Stueck.
        #
        # HEADLESS NICHT PRUEFBAR: ob es im laufenden Programm wirklich greift,
        # muss am Bildschirm bestaetigt werden (CLAUDE.md - GL-Rendering).
        if view_type == "3d" and self.map_display_3d is not None:
            for ziel in (self.map_display_3d,
                         getattr(self.map_display_3d, "display", None)):
                if ziel is not None and hasattr(ziel, "repaint"):
                    try:
                        ziel.repaint()
                    except Exception as e:
                        self.logger.debug(f"3D-Neuzeichnen nach Ansichtswechsel: {e}")

        self.view_switched.emit(view_type)
        self.logger.debug(f"Switched to {view_type} view")

    def _update_button_styles(self, active_view: str):
        """Aktualisiert Button-Styles für aktive View"""
        try:
            if active_view == "2d":
                self.view_2d_button.button_type = "primary"
                self.view_3d_button.button_type = "secondary"
            else:
                self.view_2d_button.button_type = "secondary"
                self.view_3d_button.button_type = "primary"

            # Styling neu anwenden
            for button in [self.view_2d_button, self.view_3d_button]:
                if hasattr(button, 'setup_styling'):
                    button.setup_styling()

        except Exception as e:
            self.logger.debug(f"Button style update failed: {e}")

    def get_current_display(self):
        """Gibt aktuell aktives Display zurück"""
        try:
            return self.map_display_2d if self.current_view == "2d" else self.map_display_3d
        except Exception as e:
            self.logger.debug(f"Get current display failed: {e}")
            return None

    # 2D-`layer_type`-Strings -> 3D-interne overlay_data-Layer-Namen (siehe
    # MapDisplay3DWidget.overlay_data in map_display_3d.py) - die beiden
    # Namensräume sind historisch unterschiedlich gewachsen (2D nutzt die
    # DataLODManager-Output-Keys wie "temp_map"/"soil_moist_map", 3D nutzt
    # kürzere UI-Namen wie "temperature"/"soil_moisture"). Layer ohne Eintrag
    # hier (z.B. "heightmap" selbst) werden nicht als Overlay behandelt.
    _LAYER_NAME_MAP_3D = {
        "temp_map": "temperature", "precip_map": "precipitation",
        "humid_map": "humidity", "wind_map": "wind",
        "water_map": "water_map", "soil_moist_map": "soil_moisture",
        "erosion_map": "erosion", "sedimentation_map": "sedimentation",
        "net_change_map": "net_change", "sediment_load_map": "sediment_load",
        "water_depth_map": "water_depth", "flow_velocity_map": "flow_velocity",
        "thermal_erosion_map": "thermal_erosion",
        "thermal_deposition_map": "thermal_deposition",
        "evaporation_map": "evaporation",
        "flow_map": "flow_map", "rock_map": "rock_map",
        "hardness_map": "hardness_map", "biome_map": "biome_map",
        "slopemap": "slope", "civ_map": "civ_map",
        # Geology-Diagnose-Modi (Terrain Hub/Tilt/Fold/Fault/Intrusion Only,
        # siehe gui/tabs/geology_tab.py _DELTA_DISPLAY_MODES) - identischer
        # Key auf beiden Seiten, hier trotzdem explizit gelistet (kein
        # Eintrag hier hieße: kein 3D-Overlay, siehe Docstring oben).
        "terrain_hub_delta": "terrain_hub_delta", "tilt_delta": "tilt_delta",
        "fold_delta": "fold_delta", "fault_delta": "fault_delta",
        "intrusion_delta": "intrusion_delta",
        # Regionen/Kuestentypen (2026-08-13, Nutzer-Vorgabe "die 3D
        # darstellung ALLER 2D maps, aber vor allem der Kuesten auf die 3D
        # Terrains bekommen") - der gepushte Wert ist hier bewusst das ROHE
        # Payload-Dict (wie fuer 2D), nicht ein fertiges Array. Siehe
        # MapDisplay3DWidget._render_dict_rgba_overlay().
        "region_map": "region_overlay", "kuesten_archetyp": "kuesten_overlay",
        # FLUSSREITER (2026-08-26). Er meldet sich mit
        # `generator_type = "terrain"` an, seine Layer standen aber in
        # KEINER der beiden Listen. Folge: `mapped_layer` blieb None,
        # es wurde nichts ans 3D gepusht - UND die Sichtbarkeits-
        # schleife weiter unten schaltete danach alle Terrain-Layer
        # unsichtbar. Im 3D blieb blankes Gelaende stehen, ohne jede
        # Meldung (Nutzerbefund: *"bei Flussnetzwerk gehen die 3D
        # karten nicht"*).
        #
        # DAS IST DIE STEHENDE REGEL AUS CLAUDE.md, nur an einer
        # anderen Stelle als bisher: nicht eine fehlende Methode,
        # sondern ein fehlender REGISTEREINTRAG. Beides ist von aussen
        # nicht zu unterscheiden - es passiert lautlos nichts.
        # tests/smoke_test_anzeige_register_3d.py wacht jetzt darueber.
        "river_water": "river_water", "river_order": "river_order",
        # Hoehenfaktor-Ansicht (2026-08-26) - ausdruecklich in 2D UND
        # 3D verlangt, deshalb von Anfang an hier eingetragen.
        "hinterland_height": "hinterland_height",
        "voronoi_map": "voronoi_map",
    }

    # Für welche Tabs die radio buttons ÜBER dem Canvas die EINZIGE Quelle
    # der Wahrheit für die 3D-Layer-Sichtbarkeit sind (siehe
    # _push_data_to_current_display() unten) - jeweils die Menge der "echten"
    # Daten-Layer-Keys in MapDisplay3DWidget.layer_visibility[tab_type]
    # (map_display_3d.py). Terrain behält "base"/"shadows" als eigene,
    # unabhängige Sichtbarkeits-Checkboxen (keine Layer-AUSWAHL, sondern ob
    # das Mesh/die Schatten überhaupt gezeichnet werden) - deshalb hier NICHT
    # gelistet. Settlement fehlt bewusst: dessen Checkboxen (Plots/
    # Settlements/Landmarks/Roads/Civ Map) sind ein Mehrfachauswahl-Overlay
    # (mehrere gleichzeitig sichtbar), kein Radio-Button-artiges
    # "genau ein Layer"-Muster wie bei den übrigen Tabs.
    _LAYER_SELECTION_KEYS_3D = {
        "terrain": {"slope", "region_overlay", "kuesten_overlay",
                    "river_water", "river_order",
                    "hinterland_height", "voronoi_map"},
        "geology": {"rock_map", "hardness_map", "terrain_hub_delta", "tilt_delta",
                    "fold_delta", "fault_delta", "intrusion_delta"},
        "weather": {"precipitation", "temperature", "wind", "humidity"},
        # Erosion-Umbau 2026-07-28: die vier gelaendeformenden Layer sind aus
        # "water" in den eigenen Erosion-Tab gewandert.
        "erosion": {"erosion", "sedimentation", "net_change", "sediment_load",
                    "water_depth", "flow_velocity", "thermal_erosion", "thermal_deposition"},
        "water": {"water_map", "soil_moisture", "flow_map", "evaporation"},
        "biome": {"biome_map", "super_biome_mask"},
    }

    def _push_data_to_current_display(self, data, layer_type: str):
        """
        Einheitlicher Versand von Anzeige-Daten ans aktuell aktive Display.
        Sub-Classes rufen das statt direkt current_display.update_display(),
        damit 2D und 3D korrekt geroutet werden:

        - 2D: layer_type entscheidet direkt, was gezeichnet wird (unverändert).
        - 3D: MapDisplay3DWidget.update_heightmap(heightmap, tab_type) erwartet
          als zweiten Parameter den Tab-Typ ("terrain"/"geology"/...), NICHT den
          Layer-Namen ("heightmap"/"rock_map"/...). Ohne diese Unterscheidung
          landete z.B. "rock_map" im tab_type-Feld, paintGL() erkannte keinen
          bekannten Rendermodus und zeichnete nichts.
          Overlay-Layer (alles außer "heightmap" selbst) werden zusätzlich
          IMMER an update_overlay_data() gepusht - unabhängig von
          current_view, da self.map_display_3d pro Tab immer existiert
          (siehe setup_ui()), auch wenn gerade 2D sichtbar ist. So zeigt die
          3D-Ansicht sofort die richtigen Daten, sobald der Nutzer
          umschaltet, ohne erneut generieren zu müssen.
        """
        import time as _time
        _t_push_start = _time.time()

        current_display = self.get_current_display()
        if not current_display:
            return

        # get_terrain_data_combined() statt get_terrain_data(): für jede visuelle
        # Referenz auf "die Heightmap" AUSSER Terrains eigenem "Terrain Rohform"-
        # Modus wollen wir das tatsächliche Endergebnis nach Geology-Tektonik/
        # Water-Erosion/-Sedimentation sehen (dieselbe Datenbasis, die auch alle
        # nachgelagerten Generatoren als heightmap_combined bekommen). Terrains
        # "Terrain Rohform"/"Heightmap"-Radios (bis 2026-08-13 "Terrain
        # Heightmap"/"Heightmap Combined" - auf Nutzerwunsch umbenannt, klangen
        # wie zwei Varianten desselben Dings statt zweier verschiedener Quellen)
        # sollen sich in 3D genauso unterscheidbar zeigen wie in 2D (Nutzer-
        # Vorgabe) - für beide wird deshalb `data` direkt verwendet, nur alle
        # anderen Layer-Typen (Slope etc., wo eine rohe Heightmap als 3D-Mesh
        # keinen Sinn ergäbe) fallen weiterhin auf die kombinierte Heightmap
        # zurück.
        if self.current_view == "3d" and hasattr(current_display.display, 'update_heightmap'):
            _t_fetch = _time.time()
            heightmap = data if layer_type in ("heightmap", "heightmap_combined") else (
                self.data_lod_manager.get_terrain_data_combined("heightmap") if self.data_lod_manager else None
            )
            print(f"DEBUG: _push_data_to_current_display({self.generator_type}.{layer_type}): "
                  f"Heightmap-Fetch {_time.time()-_t_fetch:.3f}s")
            if heightmap is not None:
                _t_uh = _time.time()
                current_display.display.update_heightmap(heightmap, self.generator_type)
                print(f"DEBUG: _push_data_to_current_display({self.generator_type}.{layer_type}): "
                      f"update_heightmap-Aufruf gesamt {_time.time()-_t_uh:.3f}s")
            # Live "Map Distance"-Wert mitschicken (siehe [[project-terrain-review]]
            # 4f) - das 3D-Widget hat keinen eigenen data_lod_manager-Zugriff.
            if self.data_lod_manager and hasattr(current_display.display, 'set_world_size_km'):
                current_display.display.set_world_size_km(self.data_lod_manager.get_map_distance_km())
        elif hasattr(current_display, 'update_display'):
            # Referenz-Heightmap fürs Contour-Overlay mitschicken, damit Höhenlinien
            # auch auf Nicht-Heightmap-Layern erscheinen (z.B. Water > Flowmap),
            # nicht nur wenn die Heightmap selbst der angezeigte Layer ist.
            if (layer_type not in ("heightmap", "heightmap_combined") and self.data_lod_manager
                    and hasattr(current_display.display, 'set_contour_reference_heightmap')):
                reference_heightmap = self.data_lod_manager.get_terrain_data_combined("heightmap")
                if reference_heightmap is not None:
                    current_display.display.set_contour_reference_heightmap(reference_heightmap)
            current_display.update_display(data, layer_type)

        mapped_layer = self._LAYER_NAME_MAP_3D.get(layer_type) if layer_type != "heightmap" else None

        if (mapped_layer and self.map_display_3d
                and hasattr(self.map_display_3d.display, "update_overlay_data")):
            self.map_display_3d.display.update_overlay_data(self.generator_type, mapped_layer, data)

        # Die radio buttons oben sind jetzt die EINZIGE Quelle der Wahrheit für
        # "welcher Layer wird gezeigt" - in 2D UND 3D (User-Report: separate
        # 3D-Checkboxen waren redundant zu den radio buttons, mehrere Layer
        # konnten dort gleichzeitig aktiv sein). Läuft IMMER, auch wenn gerade
        # "heightmap" (mapped_layer=None) gepusht wird - sonst blieb ein zuvor
        # gewähltes Overlay (z.B. Terrain "Slope") in 3D hängen, weil dieser
        # Block früher nur bei layer_type != "heightmap" lief und beim
        # Zurückwechseln zu Height/Basis-Layer nie mehr abgeschaltet wurde
        # (User-Report: "komme dort nicht mehr auf Height zurück"). Bei jedem
        # Push exakt den gepushten Layer sichtbar und alle anderen "echten"
        # Daten-Layer desselben Tabs unsichtbar setzen - dadurch ist der
        # 3D-Zustand immer synchron zur aktuellen radio-button-Auswahl, auch
        # beim Umschalten 2D->3D (switch_view() ruft bereits
        # update_display_mode() auf, das den aktuell gewählten Layer erneut
        # hier durchpusht).
        if self.map_display_3d and hasattr(self.map_display_3d.display, "set_layer_visibility"):
            selection_keys = self._LAYER_SELECTION_KEYS_3D.get(self.generator_type)
            if selection_keys:
                for key in selection_keys:
                    self.map_display_3d.display.set_layer_visibility(
                        self.generator_type, key, key == mapped_layer)

        # Shademap fürs 3D-Schatten-Overlay mitschicken (siehe MapDisplay3D.
        # update_shademap()/set_shadow_overlay(), [[project-3d-2d-ui-unification]]) -
        # unabhängig von layer_type/current_view, da Schatten tab-übergreifend über
        # die gemeinsame _render_terrain_base() gelten, nicht nur wenn gerade der
        # Terrain-Tab aktiv ist. shadowmap liegt als (H,W,7) vor (ein Kanal pro
        # Sonnenwinkel-Voreinstellung, siehe terrain_tab.py) - fester Standard-
        # Winkelindex 3 (vorherige Default-Position des jetzt entfernten Shadow-
        # Angle-Sliders, siehe gui/tabs/terrain_tab.py).
        DEFAULT_SHADOW_ANGLE_INDEX = 3
        print(f"DEBUG: _push_data_to_current_display({self.generator_type}.{layer_type}): "
              f"gesamt bis Schatten-Block {_time.time()-_t_push_start:.3f}s")
        if (self.map_display_3d and self.data_lod_manager
                and hasattr(self.map_display_3d.display, "update_shademap")):
            shadow_data = self.data_lod_manager.get_terrain_data("shadowmap")
            if (shadow_data is not None and hasattr(shadow_data, "ndim")
                    and shadow_data.ndim == 3 and shadow_data.shape[2] > DEFAULT_SHADOW_ANGLE_INDEX):
                self.map_display_3d.display.update_shademap(shadow_data[:, :, DEFAULT_SHADOW_ANGLE_INDEX])

        # Sonnenstand fürs 3D-Licht (siehe MapDisplay3D.set_sun_direction()) -
        # ersetzt den statischen, breitengrad-unabhängigen gui_default.py-
        # Default ("Sonne im Süden, 45°") durch den tatsächlich für die
        # eingestellte Karten-Breite berechneten Sonnenstand, damit z.B. eine
        # Südhalbkugel-Karte auch in 3D korrekt aus Norden beleuchtet wird.
        # Repräsentativer Referenztag/-stunde (Sommer, Sonnenmittag) statt
        # Monats-Animation - der 3D-View animiert nicht durchs Jahr, ein
        # fester Zeitpunkt genügt, um zumindest Hemisphäre/Breitengrad korrekt
        # abzubilden (analog zum festen Winkelindex 3 der Shademap oben).
        if (self.map_display_3d and self.parameter_manager
                and hasattr(self.map_display_3d.display, "set_sun_direction")):
            try:
                from core.terrain_generator import calculate_solar_position
                from gui.config.value_default import WEATHER
                weather_params = self.parameter_manager.get_tab_parameters("weather") or {}
                latitude = weather_params.get("map_latitude", WEATHER.MAP_LATITUDE["default"])
                longitude = weather_params.get("map_longitude", WEATHER.MAP_LONGITUDE["default"])
                elevation, azimuth = calculate_solar_position(
                    day_of_year=172, hour=13.0, latitude=latitude, longitude=longitude)
                self.map_display_3d.display.set_sun_direction(elevation, azimuth)
            except Exception as e:
                self.logger.debug(f"3D-Sonnenstand-Update übersprungen: {e}")

    def _push_overlays(self, overlays: List["Overlay"]) -> None:
        """
        Geschwistermethode zu _push_data_to_current_display(), aber fuer
        Vektor-Overlays (Siedlungen, Fluesse, ...) statt Skalarlayer
        (docs/spezifikation/15_ANZEIGE.md).

        Ein Reiter meldet nur noch WAS gezeigt werden soll (siehe Overlay-
        Werttyp oben); WIE - per overlay_*()-Aufruf in 2D, per RGBA-Skin in
        3D - entscheidet ausschliesslich das Register (_OVERLAY_REGISTER).

        Laeuft IMMER gegen BEIDE Anzeigen, unabhaengig von current_view -
        `current_view` kommt hier absichtlich NICHT vor (Spezifikation
        Punkt 4, durch tests/smoke_test_push_overlays.py nachgeprueft):
        self.map_display_3d existiert je Reiter immer (siehe setup_ui()),
        auch waehrend gerade 2D sichtbar ist - exakt dieselbe Begruendung
        wie oben in _push_data_to_current_display() fuer Skalarlayer. Genau
        das Fehlen dieser Regel war die Ursache aller vier Vorfaelle aus
        CLAUDE.md (zuletzt BiomeTab.apply_overlays(), Ticket #5).

        Ein unbekannter Overlay-Name ist ein Fehler beim Start, keine stille
        Auslassung (Spezifikation Punkt 5, erste Zeile) - wer ein neues
        Overlay anmeldet, ohne es in _OVERLAY_REGISTER einzutragen, bekommt
        das sofort gemeldet statt eines lautlos wirkungslosen Hakens.
        """
        for overlay in overlays:
            eintrag = _OVERLAY_REGISTER.get(overlay.name)
            if eintrag is None:
                raise ValueError(
                    f"Unbekanntes Overlay '{overlay.name}' - nicht in "
                    f"_OVERLAY_REGISTER (gui/tabs/base_tab.py) angemeldet. "
                    f"Siehe docs/spezifikation/15_ANZEIGE.md, Implementation Decisions 3+5."
                )
            if self.map_display_2d is not None:
                eintrag["2d"](self.map_display_2d.display, overlay)
            if self.map_display_3d is not None:
                eintrag["3d"](self.map_display_3d.display, overlay)

    @error_handler
    def update_display_mode(self):
        """Display-Update über DataLODManager"""
        try:
            if self.data_lod_manager:
                # Hole beste verfügbare Daten vom DataLODManager (kombiniert, siehe
                # _push_data_to_current_display())
                heightmap = self.data_lod_manager.get_terrain_data_combined("heightmap")
                if heightmap is not None:
                    display_id = f"{self.__class__.__name__}_{self.current_view}_heightmap"
                    if self.data_lod_manager.display_update_manager.needs_update(display_id, heightmap, "heightmap_combined"):
                        self._push_data_to_current_display(heightmap, "heightmap_combined")
                        self.data_lod_manager.display_update_manager.mark_updated(display_id, heightmap, "heightmap_combined")

        except Exception as e:
            self.logger.debug(f"Display mode update failed: {e}")

    # =============================================================================
    # GLOBALE OVERLAY-TOGGLES (Shell-Spalte 2: Contour Lines)
    # =============================================================================

    def set_contour_overlay(self, checked: bool):
        """
        Globaler Contour-Lines-Toggle vom Shell-Layout. KANN von Sub-Classes
        überschrieben werden, falls ein Generator eine abweichende Contour-
        Semantik braucht. Default delegiert an das aktive Display.
        """
        try:
            current_display = self.get_current_display()
            if current_display and hasattr(current_display.display, 'set_contour_overlay'):
                current_display.display.set_contour_overlay(checked)
        except Exception as e:
            self.logger.debug(f"Contour overlay toggle failed: {e}")

    # =============================================================================
    # GENERATION SYSTEM (nur UI-Proxy)
    # =============================================================================

    @error_handler
    def generate(self):
        """Generation-Request an GenerationOrchestrator (nur Signal)"""
        if self.generation_active:
            self.logger.warning("Generation already active")
            return

        self.generation_active = True

        try:
            # Status-Update
            if self.status_display:
                self.status_display.set_pending("Generation requested...")

            # Aktuelle Parameter aus der zentralen Quelle (ParameterManager) holen
            parameters = {}
            if self.parameter_manager:
                parameters = self.parameter_manager.get_tab_parameters(self.generator_type)

            # Signal an GenerationOrchestrator (keine eigene Logic)
            self.generate_requested.emit(self.generator_type, parameters)

        except Exception as e:
            self.generation_active = False
            self.logger.error(f"Generation request failed: {e}")
            if self.status_display:
                self.status_display.set_error(f"Generation failed: {e}")

    def start_generation_timing(self):
        """
        Startet die Zeitmessung für die laufende Generation. Gemeinsamer Helfer für
        Sub-Classes (Weather/Water/Biome/Settlement), die ihre Requests direkt am
        GenerationOrchestrator stellen statt über generate()/generate_requested.
        """
        self._generation_start_time = time.time()

    def end_generation_timing(self, success: bool = True):
        """
        Beendet die Zeitmessung und loggt die Laufzeit der Generation.
        Parameter: success - ob die Generation erfolgreich war (nur für den Log-Text)
        """
        start_time = getattr(self, '_generation_start_time', None)
        if start_time is not None:
            elapsed = time.time() - start_time
            status = "completed" if success else "failed"
            self.logger.info(f"Generation {status} in {elapsed:.2f}s")
            self._generation_start_time = None

    def handle_generation_error(self, error: Exception):
        """
        Zentrale Fehlerbehandlung für Generation-Requests, die nicht über generate()
        laufen. Setzt Status auf Fehler und beendet die Zeitmessung, damit ein
        fehlgeschlagener Request den Tab nicht dauerhaft in "aktiv" hängen lässt.
        """
        self.generation_active = False
        self.generation_in_progress = False
        self.logger.error(f"Generation error: {error}")
        if self.status_display:
            self.status_display.set_error(f"Generation failed: {error}")
        self.end_generation_timing(success=False)

    # =============================================================================
    # SIGNAL HANDLERS (vereinfacht)
    # =============================================================================

    @pyqtSlot(str, str, object, object)
    def on_parameter_changed(self, generator_type: str, param_name: str, old_value, new_value):
        """Handler für Parameter-Änderungen vom ParameterManager"""
        if generator_type == self.generator_type:
            self.update_parameter_ui(param_name, new_value)

    # Generatoren, die das GELÄNDE verändern - ihre Ergebnisse fließen über
    # DataLODManager.get_terrain_data_combined() in die kombinierte Heightmap
    # ein, aus der JEDER Tab sein 3D-Mesh baut (siehe
    # _push_data_to_current_display()).
    #
    # Sie brauchen deshalb eine Sonderbehandlung in on_data_updated(): ohne sie
    # aktualisiert ein Tab seine Anzeige nur, wenn SEIN EIGENER Generator
    # meldet. Der Erosion-Generator (seit 2026-07-28 ein eigener, der nach
    # Geology läuft) blieb damit für alle anderen Tabs unsichtbar - das
    # 3D-Mesh im Terrain-Tab zeigte weiter das unerodierte Gelände, obwohl
    # "Heightmap Combined" die Erosion längst enthielt. Nutzer-Report
    # 2026-07-28: "schau mal ob die 3D terrain meshes neu erzeugt werden
    # sobald erosion neu gerechnet wurde".
    _TERRAIN_FORMING_GENERATORS = ("terrain", "geology", "erosion")

    @pyqtSlot(str, str)
    def on_data_updated(self, generator_type: str, data_key: str):
        """Handler für Data-Updates vom DataLODManager"""
        try:
            # Dependency-Check
            if hasattr(self, 'required_dependencies') and data_key in self.required_dependencies:
                self.check_input_dependencies()

            # Display-Update bei relevanten Daten. Geländeformende Generatoren
            # lösen es in JEDEM Tab aus, nicht nur im eigenen - siehe
            # _TERRAIN_FORMING_GENERATORS.
            #
            # NUR WENN DIESER TAB GERADE SICHTBAR IST (2026-08-11, Nutzerbefund
            # via Pipeline-Log). Vorher redraw'te bei JEDEM Update eines
            # terrain-formenden Generators JEDER der zehn Tabs synchron auf dem
            # Hauptthread - gemessen 187s bei 1024px fuer Terrain allein, weil
            # `data_updated` innerhalb von set_terrain_data_complete_lod()
            # feuert (siehe generation_orchestrator._maybe_assemble_generator())
            # und dabei alle zehn on_data_updated()-Handler synchron durchlief,
            # bevor auch nur der naechste Calculator-Knoten dispatcht werden
            # konnte. self.viewport_widget ist das Shell-eingehaengte Widget
            # (siehe _create_viewport_container()) - nur die aktuell sichtbare
            # Tab-Seite im gemeinsamen QStackedWidget liefert isVisible()==True,
            # unabhaengig vom generator_type (der bei Terrain/Fluss bzw.
            # Siedlungen Global/Regional mehrfach vorkommt, also selbst keine
            # eindeutige Tab-Identitaet waere). update_display_mode() nutzt
            # bereits einen Dirty-Check (display_update_manager.needs_update())
            # statt hier selbst Zustand nachzuhalten: bleibt der Aufruf hier
            # aus, bleibt auch mark_updated() aus, und needs_update() liefert
            # beim naechsten Aufruf wieder True - der Tab holt sich seine
            # aktuellen Daten also von selbst nach. map_editor.py._on_tab_changed()
            # ruft update_display_mode() ohnehin bereits bei JEDEM Tab-Wechsel
            # auf (QTimer.singleShot(0, tab_instance.update_display_mode)) -
            # das deckt das Nachholen ab, ohne dass hier zusaetzlicher Code
            # noetig waere.
            if (self.viewport_widget is not None and self.viewport_widget.isVisible()
                    and (generator_type == self.generator_type
                         or generator_type in self._TERRAIN_FORMING_GENERATORS
                         or data_key in getattr(self, 'display_data_keys', []))):
                self.update_display_mode()

        except Exception as e:
            self.logger.debug(f"Data update handling failed: {e}")

    @pyqtSlot(str)
    def on_generation_started(self, generator_type: str):
        """Handler für Generation-Start"""
        if generator_type == self.generator_type:
            if self.status_display:
                self.status_display.set_pending("Generation in progress...")

    @pyqtSlot(str, dict)
    def on_generation_completed(self, result_id: str, result_data: dict):
        """Handler für Generation-Completion vom GenerationOrchestrator"""
        generator_type = result_data.get("generator_type", "")
        success = result_data.get("success", False)

        if generator_type == self.generator_type:
            self.generation_active = False

            if self.status_display:
                if success:
                    self.status_display.set_success("Generation completed")
                else:
                    self.status_display.set_error("Generation failed")

    @pyqtSlot(int, str)
    def on_generation_progress(self, progress: int, message: str):
        """Handler für Generation-Progress vom GenerationOrchestrator"""
        if self.generation_active and self.status_display:
            self.status_display.set_pending(f"{message} ({progress}%)")

    # =============================================================================
    # EXTENSIBILITY INTERFACE FÜR SUB-CLASSES
    # =============================================================================

    def _stillgelegte_regler_sperren(self):
        """
        Regler sperren, die im aktuellen Programmstand nichts bewirken.

        Welche das sind, steht in gui/config/value_default.stillgelegte_regler()
        - hergeleitet aus den Schaltern (WELTKARTE_AKTIV, EROSION_AKTIV) und
        nicht fest verdrahtet, damit ein Umschalten sie sofort wieder freigibt.

        Warum ueberhaupt: gemessen am 2026-08-06 bewirkten 49 von 109 Reglern
        nichts. Der Nutzer stellt etwas ein, die Karte bleibt gleich, und er
        kann nicht unterscheiden, ob das Absicht oder ein Fehler ist - das
        untergraebt jede Beurteilung der Karte.
        """
        try:
            from gui.config.value_default import stillgelegte_regler
            gesperrt = stillgelegte_regler()
        except Exception as fehler:                      # pragma: no cover
            self.logger.debug("Sperrliste nicht verfuegbar: %s", fehler)
            return

        regler = getattr(self, "parameter_sliders", None) or {}
        anzahl = 0
        for schluessel, widget in regler.items():
            grund = gesperrt.get(schluessel)
            if grund and hasattr(widget, "stilllegen"):
                widget.stilllegen(grund)
                anzahl += 1
        if anzahl:
            self.logger.info("%s: %d von %d Reglern gesperrt (ohne Wirkung)",
                             self.__class__.__name__, anzahl, len(regler))

    def create_parameter_controls(self):
        """MUSS von Sub-Classes implementiert werden - erstellt Parameter-UI-Controls"""
        self.logger.warning(f"{self.__class__.__name__} should implement create_parameter_controls()")

    def create_visualization_controls(self):
        """KANN von Sub-Classes überschrieben werden - erstellt Display-Mode-Controls"""
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Standard Heightmap Mode
        self.display_mode_group = QButtonGroup()

        height_radio = QRadioButton("Height")
        height_radio.setChecked(True)
        height_radio.toggled.connect(self.update_display_mode)
        self.display_mode_group.addButton(height_radio, 0)
        layout.addWidget(height_radio)

        widget.setLayout(layout)
        return widget

    def check_input_dependencies(self) -> bool:
        """
        KANN von Sub-Classes überschrieben werden - prüft Input-Dependencies.

        War zuvor doppelt kaputt: (1) die literale Zahl 1 wurde statt der
        Parameter-Liste required_dependencies übergeben, was in
        DataLODManager.check_dependencies() einen TypeError auslöste (int ist
        nicht iterierbar), vom umschließenden except abgefangen wurde und
        immer beim True-Fallback landete; (2) check_dependencies() gibt ein
        Tuple (bool, missing_list) zurück - ein Tuple ist in Python IMMER
        truthy, auch (False, [...]), wodurch `if not check_input_dependencies()`
        selbst nach Fix (1) nie ausgelöst hätte. Dependency-Checks liefen
        dadurch faktisch nie echt.
        """
        try:
            if self.data_lod_manager and getattr(self, 'required_dependencies', None):
                is_complete, _missing = self.data_lod_manager.check_dependencies(
                    self.generator_type, self.required_dependencies
                )
                return is_complete
        except Exception as e:
            self.logger.debug(f"Dependency check failed: {e}")

        return True  # Fallback: keine Dependencies definiert oder Check nicht möglich

    def update_parameter_ui(self, param_name: str, value):
        """KANN von Sub-Classes überschrieben werden - aktualisiert Parameter-UI"""
        self.logger.debug(f"Parameter UI update: {param_name} = {value}")

    # =============================================================================
    # RESOURCE MANAGEMENT (delegiert an DataLODManager)
    # =============================================================================

    @error_handler
    def cleanup_resources(self):
        """Systematische Resource-Cleanup über DataLODManager"""
        self.logger.debug("Cleaning up resources")

        try:
            # Signal-Disconnections
            self._disconnect_all_signals()

            # Delegiere Cleanup an DataLODManager
            if self.data_lod_manager and hasattr(self.data_lod_manager, 'resource_tracker'):
                self.data_lod_manager.resource_tracker.cleanup_resources()

            # Generation-State Reset
            self.generation_active = False

            self.logger.debug("Resource cleanup completed")

        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")

    def reset_data(self):
        """Setzt alle generierten Daten zurück über DataLODManager"""
        try:
            if self.data_lod_manager:
                # Delegiere an DataLODManager
                self.data_lod_manager.clear_generator_data(self.generator_type)

            # Status zurücksetzen
            if self.status_display:
                self.status_display.set_success("Ready")

            self.logger.info(f"Data reset completed for {self.generator_type}")

        except Exception as e:
            self.logger.error(f"Data reset failed: {e}")

    def _disconnect_all_signals(self):
        """Disconnected alle Signal-Verbindungen sicher"""
        managers_and_signals = [
            (self.parameter_manager, ['parameter_changed']),
            (self.data_lod_manager, ['data_updated']),
            (self.generation_orchestrator, ['generation_started', 'generation_completed', 'generation_progress'])
        ]

        for manager, signals in managers_and_signals:
            if manager:
                for signal_name in signals:
                    try:
                        signal = getattr(manager, signal_name, None)
                        if signal:
                            signal.disconnect()
                    except (TypeError, RuntimeError):
                        pass  # Signal bereits disconnected


# =============================================================================
# HELPER CLASSES
# =============================================================================

@dataclass
class LayoutConfiguration:
    """Konfiguration für Layout-Parameter"""
    MIN_CONTROL_WIDTH: int = 280
    MAX_CONTROL_WIDTH: int = 400
    CANVAS_RATIO: float = 0.7
    CONTROL_RATIO: float = 0.3
