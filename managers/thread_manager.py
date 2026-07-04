import logging
from managers.parameter_manager import ParameterManager
from PyQt5.QtCore import pyqtSignal


class ThreadManager:

    cancel_calculation = pyqtSignal()  # Tells generator to cancel as soon as possible
    start_calculation = pyqtSignal()
    finished_calculation = pyqtSignal(str, bool) # state, success

    def __init__(self, parameter_manager=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        if parameter_manager is None:
            self.logger.error(f"Parameter Manager wurde nicht an Thread Manager übergeben.")
        self.param = parameter_manager

        self.is_calculating = False
        self.pending_calculation = False

    def on_button_pressed_calculate(self, tab_name):
        """
        Funktionsweise: Wenn in einem Tab "Berechnen" gedrückt wird, löst request_calculation aus.
        """
        if self.param.mapValidated:
            self.logger.info(f"{tab_name}: Keine Parameter wurden geändert.")
        if self.check_if_calculation_possible():
            self.logger.info(f"{tab_name}: Berechnungscheck bestanden. Starte neue Berechnung.")
            self._start_new_calculation()

    def check_if_calculation_possible(self):
        if self.is_calculating:
            self.pending_calculation = True
            self.logger.info(f"Laufende Berechnung wird beendet.")
            return False
        return True

    def cancel_all_calculations(self):
        pass

    def calculation_cancelled(self):
        if self.pending_calculation:
            self.check_if_calculation_possible(self)

    def _start_new_calculation(self):
        self.is_calculating = True

    def _end_calculation(self, state: str, success: bool):
        self.is_calculating = False
        (self.logger.info(f"Emitting signal finished_calculation to "
                          f"ParameterManager with state = {state} and success = {success}"))
        self.finished_calculation.emit(state, success)