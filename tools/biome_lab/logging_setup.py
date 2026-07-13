"""
Path: tools/biome_lab/logging_setup.py

Zentrales Logging fuer das Physics Lab. Alle anderen Module importieren
``logger`` von hier, damit Voronoi-Worker-Abstuerze, Topologie-Fehler usw.
in einer gemeinsamen Log-Datei (logs/plot_physics_lab.log) landen.
"""
import logging
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(REPO_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class _FlushingFileHandler(logging.FileHandler):
    """Erzwingt sofortiges Flush + fsync, damit bei einem nativen Crash
    (Qhull STATUS_STACK_BUFFER_OVERRUN im Voronoi-Worker-Prozess) keine
    Log-Zeilen im Puffer verloren gehen."""

    def emit(self, record):
        super().emit(record)
        self.flush()
        try:
            os.fsync(self.stream.fileno())
        except (OSError, ValueError):
            pass


logger = logging.getLogger("plot_physics_lab")
logger.setLevel(logging.DEBUG)
_file_handler = _FlushingFileHandler(
    os.path.join(LOG_DIR, "plot_physics_lab.log"), encoding="utf-8"
)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logger.addHandler(_file_handler)
