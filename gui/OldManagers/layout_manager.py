"""
Path: gui/managers/layout_manager.py

Funktionsweise: Auslagert Layout-Logic aus BaseTab in spezialisierte Manager für bessere Wartbarkeit und Wiederverwendung
Aufgabe: Standard-Layout-Patterns, Signal-Koordination, State-Management für wiederverwendbare UI-Komponenten
Features: 70/30-Layout-Manager, Signal-Koordination, Layout-State-Persistence, Theme-Management
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass

from PyQt5.QtCore import QObject, pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy,
    QSplitter, QScrollArea, QStackedWidget, QLabel
)
from PyQt5.QtGui import QPalette, QFont


@dataclass
class LayoutConstants:
    """Standard-Layout-Konstanten für konsistente UI"""
    MIN_CONTROL_PANEL_WIDTH: int = 280
    MAX_CONTROL_PANEL_WIDTH: int = 450
    CANVAS_RATIO: float = 0.7
    CONTROL_RATIO: float = 0.3
    SCROLLBAR_MARGIN: int = 25
    CONTENT_MARGIN: int = 20
    SPLITTER_SPACING: int = 10
    CONTAINER_MARGINS: int = 5
    WIDGET_SPACING: int = 10


class BaseLayoutManager(QObject):
    """
    Funktionsweise: Manager für Standard-70/30-Layout mit wiederverwendbaren Komponenten
    Aufgabe: Erstellt und verwaltet Standard-Layout-Strukturen für Tabs
    Features: Dynamic-Resizing, State-Persistence, Component-Factory
    """

    # Signals für Layout-Management
    layout_created = pyqtSignal(str)  # (layout_type)
    splitter_moved = pyqtSignal(str, list)  # (widget_name, sizes)
    control_panel_resized = pyqtSignal(int)  # (new_width)

    def __init__(self, parent_widget: QWidget, layout_constants: Optional[LayoutConstants] = None):
        super().__init__()

        self.parent = parent_widget
        self.layout_constants = layout_constants or LayoutConstants()

        # Layout-Komponenten
        self.splitter = None
        self.canvas_container = None
        self.control_widget = None
        self.control_panel = None

        # State-Management
        self.layout_state = {}
        self.widget_registry = {}  # {widget_name: widget_instance}

        self.logger = logging.getLogger(__name__)

    def setup_70_30_layout(self, widget_name: str = "default") -> Dict[str, QWidget]:
        """
        Funktionsweise: Erstellt standardisiertes 70/30 Layout
        Parameter: widget_name für State-Management
        Return: dict mit Layout-Komponenten
        """
        self.logger.debug(f"Creating 70/30 layout for {widget_name}")

        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )
        main_layout.setSpacing(self.layout_constants.SPLITTER_SPACING)

        # Canvas Container (70%)
        self.canvas_container = self._create_canvas_container()

        # Control Widget (30%)
        self.control_widget = self._create_control_widget()

        # Splitter für resizable Layout
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.canvas_container)
        self.splitter.addWidget(self.control_widget)

        # 70/30 initial ratio
        canvas_width = int(1000 * self.layout_constants.CANVAS_RATIO)
        control_width = int(1000 * self.layout_constants.CONTROL_RATIO)
        self.splitter.setSizes([canvas_width, control_width])

        self.splitter.setCollapsible(0, False)  # Canvas nicht kollabierbar
        self.splitter.setCollapsible(1, False)  # Control Panel nicht kollabierbar

        # Signal-Verbindung
        self.splitter.splitterMoved.connect(
            lambda pos, index: self._on_splitter_moved(widget_name, pos, index)
        )

        main_layout.addWidget(self.splitter)
        self.parent.setLayout(main_layout)

        # Widget Registry
        components = {
            'splitter': self.splitter,
            'canvas_container': self.canvas_container,
            'control_widget': self.control_widget,
            'control_panel': self.control_panel
        }

        self.widget_registry[widget_name] = components

        self.layout_created.emit("70_30_layout")
        self.logger.debug(f"70/30 layout created successfully for {widget_name}")

        return components

    def _create_canvas_container(self) -> QFrame:
        """
        Funktionsweise: Erstellt Canvas-Container für 2D/3D Content
        Return: QFrame für Canvas-Content
        """
        canvas_container = QFrame()
        canvas_container.setFrameStyle(QFrame.StyledPanel)
        canvas_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Canvas Layout vorbereiten
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )
        canvas_layout.setSpacing(self.layout_constants.WIDGET_SPACING)

        canvas_container.setLayout(canvas_layout)

        return canvas_container

    def _create_control_widget(self) -> QFrame:
        """
        Funktionsweise: Erstellt Control-Widget mit Scroll-Area
        Return: QFrame für Control-Panel
        """
        optimal_width = self.calculate_control_panel_width()

        control_widget = QFrame()
        control_widget.setFrameStyle(QFrame.StyledPanel)
        control_widget.setMaximumWidth(optimal_width)
        control_widget.setMinimumWidth(min(self.layout_constants.MIN_CONTROL_PANEL_WIDTH, optimal_width))
        control_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        # Control Panel Layout
        control_main_layout = QVBoxLayout()
        control_main_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )

        # Scroll Area für Parameter
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Control Panel Content (scrollable)
        self.control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS,
            self.layout_constants.CONTAINER_MARGINS
        )
        control_layout.setSpacing(self.layout_constants.WIDGET_SPACING)
        self.control_panel.setLayout(control_layout)

        scroll_area.setWidget(self.control_panel)
        control_main_layout.addWidget(scroll_area)

        control_widget.setLayout(control_main_layout)

        return control_widget

    def calculate_control_panel_width(self) -> int:
        """
        Funktionsweise: Berechnet optimale Control-Panel-Breite
        Return: int - Optimale Control-Panel-Breite
        """
        total_width = self.parent.width() if self.parent.width() > 0 else 1200
        target_width = int(total_width * self.layout_constants.CONTROL_RATIO)

        optimal_width = max(
            self.layout_constants.MIN_CONTROL_PANEL_WIDTH,
            min(
                self.layout_constants.MAX_CONTROL_PANEL_WIDTH,
                target_width - self.layout_constants.SCROLLBAR_MARGIN - self.layout_constants.CONTENT_MARGIN
            )
        )

        return optimal_width

    def setup_display_stack(self, canvas_container: QFrame, display_widgets: List[QWidget],
                            toggle_buttons: Optional[List[QWidget]] = None) -> QStackedWidget:
        """
        Funktionsweise: Erstellt Display-Stack für 2D/3D-Wechsel
        Parameter: canvas_container, display_widgets, toggle_buttons
        Return: QStackedWidget
        """
        canvas_layout = canvas_container.layout()

        # Toggle-Buttons Layout (optional)
        if toggle_buttons:
            button_layout = QHBoxLayout()
            for button in toggle_buttons:
                button_layout.addWidget(button)
            button_layout.addStretch()
            canvas_layout.addLayout(button_layout)

        # Stacked Widget für Display-Wechsel
        display_stack = QStackedWidget()

        for widget in display_widgets:
            display_stack.addWidget(widget)

        canvas_layout.addWidget(display_stack)

        return display_stack

    def add_navigation_panel(self, control_widget: QFrame, navigation_widget: QWidget):
        """
        Funktionsweise: Fügt Navigation-Panel zum Control-Widget hinzu (nicht scrollbar)
        Parameter: control_widget, navigation_widget
        """
        control_layout = control_widget.layout()
        if control_layout:
            control_layout.addWidget(navigation_widget)

    def save_layout_state(self, widget_name: str):
        """
        Funktionsweise: Speichert Layout-State für Restore
        Parameter: widget_name
        """
        if widget_name in self.widget_registry:
            components = self.widget_registry[widget_name]
            splitter = components.get('splitter')

            if splitter:
                self.layout_state[widget_name] = {
                    'splitter_sizes': splitter.sizes(),
                    'control_panel_width': components.get('control_widget', {}).width() if components.get(
                        'control_widget') else 0
                }

    def restore_layout_state(self, widget_name: str):
        """
        Funktionsweise: Stellt Layout-State wieder her
        Parameter: widget_name
        """
        if widget_name in self.layout_state and widget_name in self.widget_registry:
            state = self.layout_state[widget_name]
            components = self.widget_registry[widget_name]
            splitter = components.get('splitter')

            if splitter and 'splitter_sizes' in state:
                splitter.setSizes(state['splitter_sizes'])

    def update_control_panel_width(self, widget_name: str):
        """
        Funktionsweise: Aktualisiert Control-Panel-Breite bei Resize
        Parameter: widget_name
        """
        if widget_name in self.widget_registry:
            components = self.widget_registry[widget_name]
            control_widget = components.get('control_widget')

            if control_widget:
                optimal_width = self.calculate_control_panel_width()
                control_widget.setMaximumWidth(optimal_width)
                control_widget.setMinimumWidth(min(self.layout_constants.MIN_CONTROL_PANEL_WIDTH, optimal_width))

                self.control_panel_resized.emit(optimal_width)

    def _on_splitter_moved(self, widget_name: str, pos: int, index: int):
        """
        Funktionsweise: Handler für Splitter-Movement
        Parameter: widget_name, pos, index
        """
        if widget_name in self.widget_registry:
            components = self.widget_registry[widget_name]
            splitter = components.get('splitter')

            if splitter:
                sizes = splitter.sizes()
                self.splitter_moved.emit(widget_name, sizes)
                self.save_layout_state(widget_name)


class BaseSignalCoordinator(QObject):
    """
    Funktionsweise: Koordiniert Signal-Verbindungen für wiederverwendbare Patterns
    Aufgabe: Standard-Signal-Setups, Safe-Disconnection, Signal-Management
    Features: Connection-Tracking, Safe-Disconnection, Standard-Patterns
    """

    def __init__(self, parent_widget: QWidget):
        super().__init__()

        self.parent_widget = parent_widget
        self.connected_signals = []  # [(signal, slot, connection)]

        self.logger = logging.getLogger(__name__)

    def connect_data_manager_signals(self, data_manager: Any):
        """
        Funktionsweise: Verbindet Standard Data-Manager Signals
        Parameter: data_manager
        """
        if not data_manager:
            return

        # Standard-Verbindungen
        connections = [
            (data_manager.data_updated, 'on_external_data_updated'),
            (data_manager.cache_invalidated, 'on_cache_invalidated')
        ]

        for signal, slot_name in connections:
            if hasattr(self.parent_widget, slot_name):
                slot = getattr(self.parent_widget, slot_name)
                connection = signal.connect(slot)
                self.connected_signals.append(('data_manager', signal, slot, connection))
                self.logger.debug(f"Connected data_manager.{signal} to {slot_name}")

    def connect_navigation_signals(self, navigation_manager: Any):
        """
        Funktionsweise: Verbindet Navigation-Manager Signals
        Parameter: navigation_manager
        """
        if not navigation_manager:
            return

        # Standard-Navigation-Signals
        connections = [
            ('tab_changed', 'on_tab_changed'),
            ('navigation_state_changed', 'on_navigation_state_changed')
        ]

        for signal_name, slot_name in connections:
            if (hasattr(navigation_manager, signal_name) and
                    hasattr(self.parent_widget, slot_name)):
                signal = getattr(navigation_manager, signal_name)
                slot = getattr(self.parent_widget, slot_name)
                connection = signal.connect(slot)
                self.connected_signals.append(('navigation_manager', signal, slot, connection))
                self.logger.debug(f"Connected navigation_manager.{signal_name} to {slot_name}")

    def connect_orchestrator_signals(self, orchestrator: Any, generator_type: str):
        """
        Funktionsweise: Verbindet Standard-Orchestrator Signals
        Parameter: orchestrator, generator_type
        """
        if not orchestrator:
            return

        # Standard-Orchestrator-Signals
        connections = [
            ('generation_started', 'on_orchestrator_generation_started'),
            ('generation_completed', 'on_orchestrator_generation_completed'),
            ('generation_progress', 'on_orchestrator_generation_progress')
        ]

        for signal_name, slot_name in connections:
            if (hasattr(orchestrator, signal_name) and
                    hasattr(self.parent_widget, slot_name)):
                signal = getattr(orchestrator, signal_name)
                slot = getattr(self.parent_widget, slot_name)
                connection = signal.connect(slot)
                self.connected_signals.append(('orchestrator', signal, slot, connection))