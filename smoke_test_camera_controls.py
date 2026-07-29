"""
Path: smoke_test_camera_controls.py

Prueft die KAMERA-RECHNUNG der 3D-Ansicht ohne sichtbares Fenster.

Warum das geht, obwohl Rendering headless nicht pruefbar ist: die Bewegung der
Kamera ist reine Vektorrechnung auf `camera_target`, `camera_azimuth` und
`camera_distance`. Ob das Bild danach richtig AUSSIEHT, muss der Nutzer
beurteilen - ob sich die Kamera in die richtige Richtung bewegt, kann diese
Datei belegen.

Der eigentliche Anlass ist ein Vorzeichen: _update_view_matrix() negiert die
X-Zeile der View-Matrix (alte Ost-West-Korrektur, dort begruendet).
Bildschirm-rechts ist deshalb NICHT cross(forward, up).

Wo dieses SCREEN_RIGHT_SIGN hingehoert, ist erfahrungsgemaess nicht zu
erraten - beide moeglichen Fehler sind in dieser Datei passiert und werden
hier festgehalten:

  VERSCHIEBUNG entlang der Rechts-Achse braucht es - Panning wie Strafen,
  beide holen sich `right` aus _screen_axes().
  DREHSINN des Azimuts braucht es NICHT. Beleg: die Maus dreht seit jeher
  ohne jedes Vorzeichen und fuehlt sich richtig an. (Als A/D noch drehten
  statt zu strafen, waren sie mit dem Faktor prompt vertauscht.)
  Es darf ausserdem nicht VOR der Berechnung der Hoch-Achse angewandt
  werden, sonst kippt es ueber das Kreuzprodukt durch und die Leertaste
  senkt die Kamera, statt sie zu heben.

Alle Bildschirm-Aussagen werden deshalb gegen die ECHTE View-Matrix geprueft
(siehe view_space()), nie gegen ein Vorzeichen im Quelltext - sonst wuerde
der Test nur belegen, dass dieselbe Annahme zweimal aufgeschrieben wurde.
"""

import sys

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

import math

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication  # noqa: F401  (Kontext fuer QWidget-Klassen)
from PyQt6.QtWidgets import QApplication

_APP = QApplication.instance() or QApplication([])

from gui.widgets.map_display_3d import MapDisplay3D, _create_lookat_matrix  # noqa: E402


def check(label, condition):
    print("[{}] {}".format("OK" if condition else "FAIL", label))
    return bool(condition)


def make_camera():
    """Eine Kamera ohne GL-Kontext - fuer die reine Rechnung genuegt das."""
    camera = MapDisplay3D()
    camera.camera_target = [0.0, 0.0, 0.0]
    camera.camera_azimuth = 180.0
    camera.camera_elevation = 55.0
    camera.camera_distance = 17.0
    return camera


def view_space(camera, point):
    """Ein Weltpunkt im Kameraraum - inklusive der X-Spiegelung.

    Bildschirm-Aussagen werden gegen die ECHTE View-Matrix geprueft, nicht
    gegen ein Vorzeichen im Quelltext. Sonst prueft der Test nur, dass ich
    zweimal dasselbe geschrieben habe.
    """
    camera._update_view_matrix()
    matrix = camera.view_matrix
    return [sum(matrix[row, col] * point[col] for col in range(3)) + matrix[row, 3]
            for row in range(3)]


def run_forward_follows_the_view():
    """
    W faehrt entlang der ECHTEN Blickrichtung, S entgegen.

    Zusammen mit dem Free Look (linke Maustaste, Neigung frei) heisst das:
    man fliegt dorthin, wo man hinschaut. Eine waagerechte Variante stand
    hier kurz, solange die Neigung fest bei 55 Grad war.
    """
    camera = make_camera()
    forward = camera._forward()

    ok = check("W ist normiert",
               abs(math.sqrt(sum(c * c for c in forward)) - 1.0) < 1e-9)

    offset = camera._eye_offset()
    expected = [-c / math.sqrt(sum(o * o for o in offset)) for c in offset]
    ok &= check("W zeigt genau vom Auge zum Blickpunkt",
                max(abs(forward[a] - expected[a]) for a in range(3)) < 1e-12)

    ok &= check("W hat bei geneigter Kamera eine Abwaertskomponente "
                "(y = {:.2f} < 0)".format(forward[1]), forward[1] < 0)

    # Ein Punkt VOR der Kamera muss beim Vorwaertsfliegen naeher kommen.
    ahead = [camera.camera_target[a] + forward[a] * 5.0 for a in range(3)]
    before = view_space(camera, ahead)[2]
    camera._pressed_keys = {Qt.Key.Key_W}
    for _ in range(10):
        camera._advance_flight()
    after = view_space(camera, ahead)[2]
    ok &= check("W bringt einen Punkt voraus naeher (Tiefe {:.3f} -> {:.3f})"
                .format(before, after), after > before + 1e-6)
    return ok


def run_pan_matches_the_screen():
    """
    Panning muss sich auf dem BILDSCHIRM richtig anfuehlen.

    Geprueft wird gegen die tatsaechliche View-Matrix inklusive der
    X-Spiegelung: zieht die Maus nach rechts, muss der Blickpunkt im
    Kamera-Raum nach LINKS wandern (die Welt zieht mit der Maus mit).
    """
    camera = make_camera()

    def to_view_space(point):
        camera._update_view_matrix()
        matrix = camera.view_matrix
        return [sum(matrix[row, col] * point[col] for col in range(3)) + matrix[row, 3]
                for row in range(3)]

    before = to_view_space([0.0, 0.0, 0.0])
    camera._pan(40, 0)          # Maus nach rechts
    after = to_view_space([0.0, 0.0, 0.0])

    ok = check("Maus nach rechts schiebt die Welt nach rechts "
               "(x im Kameraraum {:.3f} -> {:.3f})".format(before[0], after[0]),
               after[0] > before[0] + 1e-6)

    camera = make_camera()
    before = to_view_space([0.0, 0.0, 0.0])
    camera._pan(0, 40)          # Maus nach unten
    after = to_view_space([0.0, 0.0, 0.0])
    # Die Szene folgt dem Cursor: geht die Maus nach unten, wandert die Welt
    # nach unten, ihr y im Kameraraum also nach UNTEN. Dieselbe Konvention wie
    # waagerecht - Ziehen heisst "die Karte anfassen und mitnehmen".
    ok &= check("Maus nach unten schiebt die Welt nach unten "
                "(y im Kameraraum {:.3f} -> {:.3f})".format(before[1], after[1]),
                after[1] < before[1] - 1e-6)

    camera = make_camera()
    far, near = 200.0, 5.0
    camera.camera_distance = far
    camera._pan(40, 0)
    weit = abs(camera.camera_target[0]) + abs(camera.camera_target[2])
    camera = make_camera()
    camera.camera_distance = near
    camera._pan(40, 0)
    nah = abs(camera.camera_target[0]) + abs(camera.camera_target[2])
    ok &= check("Panning skaliert mit dem Zoom ({:.2f} weit gegen {:.2f} nah)"
                .format(weit, nah), weit > nah * 5)
    return ok


def run_strafe_moves_sideways():
    """
    A/D strafen seitwaerts - sie drehen nicht.

    Geprueft gegen die echte View-Matrix: driftet man nach rechts, wandert ein
    Punkt geradeaus im Bild nach LINKS. Eine Zusicherung "die Position aendert
    sich" wuerde die Links/Rechts-Vertauschung nicht finden, und genau die ist
    hier schon einmal passiert.
    """
    camera = make_camera()
    forward = camera._forward()
    ahead = [camera.camera_target[a] + forward[a] * 5.0 for a in range(3)]

    before_azimuth = camera.camera_azimuth
    before_x = view_space(camera, ahead)[0]

    camera._pressed_keys = {Qt.Key.Key_D}
    for _ in range(10):
        camera._advance_flight()

    after_x = view_space(camera, ahead)[0]

    ok = check("D driftet nach RECHTS (Punkt voraus wandert im Bild nach links, "
               "x {:.3f} -> {:.3f})".format(before_x, after_x),
               after_x < before_x - 1e-6)
    ok &= check("A/D drehen NICHT (Azimut bleibt {:.1f})".format(
        camera.camera_azimuth), camera.camera_azimuth == before_azimuth)
    ok &= check("A/D bleiben in der Bildebene, ohne Hoehenversatz "
                "(y = {:.2e})".format(camera.camera_target[1]),
                abs(camera.camera_target[1]) < 1e-9)

    camera_a = make_camera()
    camera_a._pressed_keys = {Qt.Key.Key_A}
    camera_a._advance_flight()
    camera_d = make_camera()
    camera_d._pressed_keys = {Qt.Key.Key_D}
    camera_d._advance_flight()
    ok &= check("A und D sind Gegenrichtungen",
                max(abs(camera_a.camera_target[axis] + camera_d.camera_target[axis])
                    for axis in range(3)) < 1e-12)
    return ok


def run_mouse_gestures():
    """
    Drei Gesten, drei verschiedene Wirkungen - jede gegen die andere abgegrenzt.

    LINKS   Free Look: das AUGE bleibt stehen, der Blickpunkt wandert
    MITTE   Panning
    RECHTS  die alte Geste: der BLICKPUNKT bleibt stehen, das Auge wandert
    """
    camera = make_camera()

    def eye(cam):
        offset = cam._eye_offset()
        return [cam.camera_target[axis] + offset[axis] for axis in range(3)]

    before_eye = eye(camera)
    camera._rotate_in_place(20.0, 10.0)
    drift = max(abs(eye(camera)[axis] - before_eye[axis]) for axis in range(3))

    ok = check("Free Look laesst das Auge stehen (Versatz {:.2e})".format(drift),
               drift < 1e-6)
    ok &= check("Free Look aendert auch die NEIGUNG ({:.1f} Grad)".format(
        camera.camera_elevation), abs(camera.camera_elevation - 55.0) > 1e-6)

    camera = make_camera()
    camera._rotate_in_place(0.0, 1000.0)
    ok &= check("die Neigung wird begrenzt ({:.1f} <= {:.1f} Grad) - bei genau "
                "90 Grad kippt die lookAt-Matrix weg".format(
                    camera.camera_elevation, MapDisplay3D.ELEVATION_LIMIT_DEG),
                abs(camera.camera_elevation) <= MapDisplay3D.ELEVATION_LIMIT_DEG + 1e-9)

    # Gegenprobe zur alten Geste: dort bleibt der Blickpunkt stehen.
    camera = make_camera()
    target_before = list(camera.camera_target)
    eye_before = eye(camera)
    camera.camera_azimuth = (camera.camera_azimuth + 20.0) % 360.0
    ok &= check("Gegenprobe Rechtsklick: der BLICKPUNKT bleibt stehen, das Auge "
                "wandert",
                camera.camera_target == target_before
                and max(abs(eye(camera)[a] - eye_before[a]) for a in range(3)) > 1e-6)
    return ok


def run_vertical_is_orthogonal_to_the_view():
    """Leertaste/Shift bewegen orthogonal zur Blickrichtung (Nutzer-Vorgabe)."""
    camera = make_camera()
    right, up = camera._screen_axes()

    offset = camera._eye_offset()
    length = math.sqrt(sum(c * c for c in offset))
    forward = [-c / length for c in offset]

    ok = check("Hoch-Achse steht senkrecht auf der Blickrichtung "
               "(Skalarprodukt {:.2e})".format(
                   sum(up[a] * forward[a] for a in range(3))),
               abs(sum(up[a] * forward[a] for a in range(3))) < 1e-9)
    ok &= check("Rechts-Achse steht senkrecht auf der Hoch-Achse",
                abs(sum(up[a] * right[a] for a in range(3))) < 1e-9)
    ok &= check("Hoch-Achse zeigt tatsaechlich nach oben (y = {:.2f} > 0)"
                .format(up[1]), up[1] > 0)

    camera = make_camera()
    camera._pressed_keys = {Qt.Key.Key_Space}
    camera._advance_flight()
    ok &= check("Leertaste hebt die Kamera (y = {:.4f} > 0)".format(
        camera.camera_target[1]), camera.camera_target[1] > 0)

    camera = make_camera()
    camera._pressed_keys = {Qt.Key.Key_Shift}
    camera._advance_flight()
    ok &= check("Shift senkt die Kamera (y = {:.4f} < 0)".format(
        camera.camera_target[1]), camera.camera_target[1] < 0)
    return ok


def run_generate_no_longer_listens_to_space():
    """
    Die Leertaste darf nicht mehr generieren - sie gehoert der Kamera.

    Es gab nie einen Space-Shortcut; die Leertaste loeste den FOKUSSIERTEN
    QPushButton aus. Geprueft wird deshalb die Fokus-Politik des Knopfes und
    die Existenz des Enter-Ersatzes, nicht ein Tastendruck.
    """
    import inspect
    from gui import map_editor

    source = inspect.getsource(map_editor.MapEditorWindow._create_footer_bar)

    ok = check("der [GENERIEREN]-Knopf nimmt keinen Tastaturfokus mehr",
               "setFocusPolicy(Qt.FocusPolicy.NoFocus)" in source)
    ok &= check("Enter/Return generiert stattdessen",
                "Key_Return" in source and "Key_Enter" in source)

    # Ueber den SYNTAXBAUM, nicht per Textsuche: der Quelltext erklaert im
    # Kommentar, was hier frueher stand ("QTimer.singleShot(0,
    # self._auto_start_generation)"), und eine Textsuche findet genau diesen
    # Kommentar wieder. Der Syntaxbaum kennt keine Kommentare.
    import ast
    import textwrap

    startup = ast.parse(textwrap.dedent(
        inspect.getsource(map_editor.MapEditorWindow.__init__)))
    called = {node.attr for node in ast.walk(startup)
              if isinstance(node, ast.Attribute)}
    ok &= check("beim Start wird nicht mehr automatisch generiert "
                "(kein _auto_start_generation im Code von __init__)",
                "_auto_start_generation" not in called)

    # Gegenprobe: die Methode gibt es weiterhin - [GENERIEREN] und
    # "Regenerate All" brauchen sie. Nur der automatische Aufruf ist weg.
    ok &= check("Gegenprobe: _auto_start_generation existiert weiterhin",
                hasattr(map_editor.MapEditorWindow, "_auto_start_generation"))
    return ok


def main():
    tests = [
        ("forward_follows_the_view", run_forward_follows_the_view),
        ("pan_matches_the_screen", run_pan_matches_the_screen),
        ("strafe_moves_sideways", run_strafe_moves_sideways),
        ("mouse_gestures", run_mouse_gestures),
        ("vertical_is_orthogonal", run_vertical_is_orthogonal_to_the_view),
        ("space_no_longer_generates", run_generate_no_longer_listens_to_space),
    ]
    results = {}
    for name, func in tests:
        print("\n=== {} ===".format(name))
        try:
            results[name] = func()
        except Exception as error:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n=== SUMMARY ===")
    for name, passed in results.items():
        print("{}: {}".format(name, "PASS" if passed else "FAIL"))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
