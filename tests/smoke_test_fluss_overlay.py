"""
Path: tests/smoke_test_fluss_overlay.py

Prueft das Flussnetz-Overlay fuer die 3D-Ansicht.

ANLASS (Nutzerbefund 2026-08-24): *"dass man im 3D modus bei dem
Flussnetzwerk keine fluesse sehn kann. ich will in den jeweiligen reitern
die fluesse auf dem boden sehen."*

DIE URSACHE WAR EIN STILLER AUSFALL. `gui/tabs/river_tab.py` ruft
`overlay_river_generations()` ueber ein `hasattr` auf. Die 2D-Anzeige hat
die Methode seit dem 2026-08-06, die 3D-Anzeige hatte sie nie - der Aufruf
fiel lautlos aus, ohne Fehler und ohne Warnung. Genau das Muster, vor dem
CLAUDE.md warnt.

WAS HIER GEPRUEFT WIRD und was nicht: die Rasterfunktion
`rasterize_fluesse_rgba()` und die Verdrahtung im 3D-Widget lassen sich
headless pruefen - das sind Arrays und Attribute. Ob das Netz auf dem
Bildschirm ERSCHEINT, kann dieser Test NICHT sagen; OpenGL braucht dafuer
ein sichtbares Fenster (CLAUDE.md). Die Sichtpruefung bleibt am Nutzer.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_fluss_overlay.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui.widgets.overlay_rasterizer import rasterize_fluesse_rgba


def _testfeld(n=64):
    """Ein Netz mit allen drei Generationen und etwas Meer."""
    gen = np.zeros((n, n), dtype=np.float64)
    # REIHENFOLGE BEACHTEN: Meso zuerst, Makro darueber - sonst
    # ueberschreibt die zweite Zuweisung die Kreuzung, und der Test
    # prueft an dieser Stelle das Falsche (erster Anlauf 2026-08-24).
    gen[:, n // 3] = 2.0                      # Meso laengs
    gen[n // 2, :] = 3.0                      # Makro quer darueber
    gen[n // 4, : n // 2] = 1.0               # Mikro, halbe Breite
    H = np.full((n, n), 120.0)
    H[:, : n // 8] = -20.0                    # Meer am linken Rand
    return gen, H


def farben_und_generationen():
    """1. Jede Generation bekommt ihre Farbe, Makro liegt oben."""
    fehler = []
    gen, H = _testfeld()

    rgba = rasterize_fluesse_rgba(gen, H, zeige_mikro=True, breite_px=0)
    erwartet = {3.0: (224, 48, 48), 2.0: (37, 160, 58), 1.0: (232, 192, 32)}
    for wert, farbe in erwartet.items():
        # Eine Stelle, an der NUR diese Generation liegt (rechte Haelfte,
        # damit sich Makro und Meso nicht kreuzen).
        treffer = (gen == wert)
        treffer[:, : gen.shape[1] // 3 + 1] = False
        if not treffer.any():
            continue
        yy, xx = np.nonzero(treffer)
        ist = tuple(int(v) for v in rgba[yy[0], xx[0], :3])
        ok = ist == farbe
        print(f"[{'OK' if ok else 'FEHLER'}] Generation {wert:.0f}: "
              f"RGB {ist} (erwartet {farbe})")
        if not ok:
            fehler.append(f"Generation {wert:.0f}: {ist} statt {farbe}")

    # An der Kreuzung von Makro und Meso muss MAKRO gewinnen - sonst laege
    # ein Nebenfluss ueber seinem Strom.
    ky, kx = gen.shape[0] // 2, gen.shape[1] // 3
    ist = tuple(int(v) for v in rgba[ky, kx, :3])
    ok = ist == erwartet[3.0]
    print(f"[{'OK' if ok else 'FEHLER'}] an der Kreuzung gewinnt Makro: "
          f"RGB {ist}")
    if not ok:
        fehler.append(f"Kreuzung: {ist} statt {erwartet[3.0]}")
    return fehler


def mikro_bleibt_normalerweise_weg():
    """
    2. Ohne `zeige_mikro` ist die feinste Stufe unsichtbar.

    Nutzer 2026-08-06: *"die kleineren fluesse sind nicht zu sehen, zu
    insignifikant."* Auf 21 km sind das Rinnsale von wenigen hundert
    Metern - sie fuellen das Bild, ohne etwas auszusagen.
    """
    fehler = []
    gen, H = _testfeld()
    for zeige, soll_sichtbar in ((False, False), (True, True)):
        rgba = rasterize_fluesse_rgba(gen, H, zeige_mikro=zeige, breite_px=0)
        mikro = (gen == 1.0) & (gen != 2.0) & (gen != 3.0)
        mikro[:, : gen.shape[1] // 3 + 1] = False
        sichtbar = bool((rgba[mikro, 3] > 0).any()) if mikro.any() else False
        ok = sichtbar == soll_sichtbar
        print(f"[{'OK' if ok else 'FEHLER'}] zeige_mikro={zeige}: "
              f"Mikro sichtbar {sichtbar} (erwartet {soll_sichtbar})")
        if not ok:
            fehler.append(f"zeige_mikro={zeige}: sichtbar {sichtbar}")
    return fehler


def nichts_im_meer():
    """
    3. Kein Flusspixel liegt ueber Wasser.

    Ein Lauf reicht konstruktionsbedingt bis MUENDUNGSTIEFE_M (-50 m) ins
    Meer, damit die Muendungsrichtung stimmt (Modulkopf von
    core/terrain_weltfluesse.py). Gezeichnet wird nur der Teil ueber Null -
    sonst laege ein blauer Strich auf blauem Wasser.
    """
    fehler = []
    gen, H = _testfeld()
    rgba = rasterize_fluesse_rgba(gen, H, zeige_mikro=True, breite_px=1)
    im_meer = int((rgba[H <= 0.0, 3] > 0).sum())
    ok = im_meer == 0
    print(f"[{'OK' if ok else 'FEHLER'}] sichtbare Pixel ueber Wasser: "
          f"{im_meer} (soll 0)")
    if not ok:
        fehler.append(f"{im_meer} Flusspixel liegen im Meer")
    return fehler


def verdrahtung_im_3d():
    """
    4. DIE EIGENTLICHE ZUSICHERUNG: die 3D-Anzeige kennt das Overlay.

    Genau hier lag der Fehler - die Methode fehlte, und `river_tab` rief
    sie ueber `hasattr` auf. Geprueft wird deshalb die ANWESENHEIT, nicht
    das Bild.
    """
    fehler = []
    import gui.widgets.map_display_3d as M

    pruefungen = [
        # BEIDE Klassen. `base_tab` legt das Widget in einen
        # `DisplayWrapper`; `river_tab` greift ueber `.display` zu und
        # landet damit bei `MapDisplay3DWidget`. Die Methode nur in der
        # GL-Klasse zu haben genuegt nicht - `hasattr` schlaegt dann fehl
        # und der Aufruf faellt lautlos aus.
        ("Methode in MapDisplay3D (GL)",
         hasattr(M.MapDisplay3D, "overlay_river_generations")),
        ("Methode in MapDisplay3DWidget (das ist `.display`)",
         hasattr(M.MapDisplay3DWidget, "overlay_river_generations")),
    ]
    quelle = ""
    try:
        import inspect
        quelle = inspect.getsource(M)
    except Exception:                                        # noqa: BLE001
        pass
    pruefungen += [
        ("Slot river_overlay unter terrain",
         '"river_overlay": None' in quelle),
        # AUS als Vorgabe, eingeschaltet vom Setter. Waere sie fest auf
        # True, laege das Netz ueber jeder anderen Ansicht des Reiters.
        ("Sichtbarkeit river_overlay ist AUS als Vorgabe",
         '"river_overlay": False' in quelle),
        ("Setter schaltet sie ein",
         'set_layer_visibility("terrain", "river_overlay", True)' in quelle),
        ("clear_river_overlay in beiden Klassen",
         hasattr(M.MapDisplay3D, "clear_river_overlay")
         and hasattr(M.MapDisplay3DWidget, "clear_river_overlay")),
        # Der Cache darf NICHT auf id(payload) schluesseln - das Dict wird
        # bei jedem Aufruf neu gebaut, der Cache griffe nie.
        ("Cache schluesselt auf den Inhalt",
         'id(payload.get("river_generation"))' in quelle),
        ("wird im Terrain-Zweig gezeichnet",
         '_render_dict_rgba_overlay("terrain", "river_overlay")' in quelle),
        ("Rasterfunktion eingebunden",
         "rasterize_fluesse_rgba" in quelle),
    ]
    for name, bedingung in pruefungen:
        print(f"[{'OK' if bedingung else 'FEHLER'}] {name}")
        if not bedingung:
            fehler.append(name)

    # Die frühere Zusicherung hier prüfte zusätzlich, dass river_tab.py sich
    # als "terrain" anmeldet UND das Overlay über `_push_overlays()`
    # abschaltet - das galt fürs eigene "rivers"-Ansichtsmodul des
    # Fluss-Reiters (Ticket #11), das seit 2026-09-23 entfernt ist
    # (Begründung im Modul-Docstring von gui/tabs/river_tab.py). River_tab
    # setzt das "fluesse"-Overlay seither nirgends mehr - geprüft wird die
    # Verdrahtung des Overlays selbst jetzt ausschließlich über BiomeTab,
    # siehe tests/smoke_test_biome_overlays_3d.py.
    return fehler


def randfaelle():
    """5. Leeres Netz, falsche Formen."""
    fehler = []
    faelle = [
        ("kein Fluss", np.zeros((16, 16)), np.ones((16, 16)) * 50.0),
        ("kein Array", None, np.ones((16, 16))),
        ("eindimensional", np.zeros(16), np.ones((16, 16))),
    ]
    for name, gen, H in faelle:
        try:
            rgba = rasterize_fluesse_rgba(gen, H)
            ok = isinstance(rgba, np.ndarray) and rgba.ndim == 3
            sichtbar = int((rgba[..., 3] > 0).sum()) if ok else -1
            print(f"[{'OK' if ok and sichtbar == 0 else 'FEHLER'}] {name}: "
                  f"Form {rgba.shape}, sichtbar {sichtbar}")
            if not ok or sichtbar != 0:
                fehler.append(f"{name}: {rgba.shape}, sichtbar {sichtbar}")
        except Exception as e:                               # noqa: BLE001
            print(f"[FEHLER] {name}: Absturz - {e}")
            fehler.append(f"{name}: Absturz - {e}")
    return fehler


def lauf():
    gruppen = [
        ("farben_und_generationen", farben_und_generationen),
        ("mikro_bleibt_normalerweise_weg", mikro_bleibt_normalerweise_weg),
        ("nichts_im_meer", nichts_im_meer),
        ("verdrahtung_im_3d", verdrahtung_im_3d),
        ("randfaelle", randfaelle),
    ]
    ergebnis, alle = {}, []
    for name, fn in gruppen:
        print(f"\n--- {name} ---")
        f = fn()
        ergebnis[name] = not f
        alle.extend(f)

    print("\n=== SUMMARY ===")
    for name, ok in ergebnis.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    print("\nNICHT geprueft: ob das Netz auf dem Bildschirm erscheint.")
    print("OpenGL braucht dafuer ein sichtbares Fenster (CLAUDE.md).")
    if alle:
        print(f"\nNICHT IN ORDNUNG - {len(alle)} Befunde:")
        for f in alle:
            print(f"   {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(lauf())
