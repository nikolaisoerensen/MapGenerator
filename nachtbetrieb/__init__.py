"""
Path: nachtbetrieb/__init__.py

Alles, was noetig ist, damit nachts ein Agent allein arbeiten kann, ohne dass
morgens jemand ueberrascht wird.

  * sperrliste.toml - wo nachts niemand hinfasst, mit Begruendung je Eintrag
  * sperre.py       - die Durchsetzung dieser Liste
  * branch.py       - das Branch- und Commit-Verfahren einer Nacht

Der Ablauf und die Entscheidungen dahinter stehen in docs/NACHTBETRIEB.md.
Bedient wird das Ganze ueber tools/nachtlauf.py.

Bewusst KEIN Teil des Programms: dieser Ordner steht nicht in der
include-Liste von pyproject.toml und wird nie von core/, gui/ oder managers/
importiert. Er ist Werkzeug, so wie tools/.
"""
