"""
Path: tools/regionen_fluesse.py

WEITERLEITUNG auf core/terrain_weltfluesse.py - siehe tools/regionen_welt.py
fuer die Begruendung: zwei Kopien wuerden auseinanderlaufen.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.terrain_weltfluesse import *          # noqa: F401,F403
from core.terrain_weltfluesse import (          # noqa: F401
    _kanten_und_kosten, STUFEN, STUFEN_FARBE, MUENDUNGSTIEFE_M, ERBE_KOSTEN)
