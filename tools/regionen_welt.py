"""
Path: tools/regionen_welt.py

WEITERLEITUNG. Der Rechenkern der Regionenwelt liegt seit dem 2026-08-05 in
core/terrain_weltkarte.py, weil ihn jetzt auch das Hauptprogramm benutzt
(docs/INTEGRATIONSPLAN.md, Stufe P1).

Diese Datei bleibt unter ihrem Namen bestehen, damit die Labore
(regionen_fluesse.py, die Renderskripte) unveraendert weiterlaufen. Sie
enthaelt KEINE eigene Fassung: zwei Kopien wuerden auseinanderlaufen, und
genau das hat dieses Projekt schon einmal Tage gekostet.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.terrain_weltkarte import *            # noqa: F401,F403
from core.terrain_weltkarte import (            # noqa: F401
    _STAPEL_CACHE, _wellenlaengen, REGIONEN, REGLER, SPREIZUNG)
