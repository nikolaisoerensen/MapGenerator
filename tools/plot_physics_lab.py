"""
Path: tools/plot_physics_lab.py

Duenner Kompatibilitaets-Einstiegspunkt: die eigentliche Implementierung
wurde nach tools/biome_lab/ aufgeteilt (models/scene/topology/field/
physics/traffic/draw/ui/app), siehe dortige Docstrings. Dieser Wrapper
bleibt bestehen, damit der bekannte Start-Befehl unveraendert funktioniert:

Start: .venv/Scripts/python.exe tools/plot_physics_lab.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.biome_lab.app import main

if __name__ == "__main__":
    main()
