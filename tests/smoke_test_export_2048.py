"""
Path: tests/smoke_test_export_2048.py

Prueft den Export auf feste Weltgroesse (docs/OFFENE_PUNKTE.md 13.1) und die
Daempfungsmaske fuers Engine-Rauschen (13.2).

DIE ZUSICHERUNG, AUF DIE ES ANKOMMT

Kategorische Layer duerfen beim Hochrechnen NICHT interpoliert werden.
Zwischen Biom 3 und Biom 7 liegt kein Biom 5 - ein gemitteltes Kuestenpixel
waere ein Biom, das es in der Welt gar nicht gibt, und im Spiel stuende dort
die falsche Vegetation. Der Test prueft deshalb nicht nur "Groesse stimmt",
sondern dass die Wertemenge unveraendert bleibt.

Bei der Daempfungsmaske ist die entscheidende Zusicherung, dass auf
Bauflaechen und Wegen wirklich gedaempft wird - eine Maske, die ueberall 1
ist, waere von "keine Maske" nicht zu unterscheiden.

Aufruf:
    .venv/Scripts/python.exe tests/smoke_test_export_2048.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

from gui.utils.map_export import (EXPORT_KANTENLAENGE_PX, VEKTOR_DATEI,
                                  _auf_exportgroesse, daempfungsmaske,
                                  export_all_layers, vektordaten)


def check(label, bedingung, zusatz=""):
    status = "OK" if bedingung else "FAIL"
    print(f"[{status}] {label}{(' - ' + zusatz) if zusatz else ''}")
    return bool(bedingung)


class FakeDLM:
    """Nur so viel Manager, wie der Export tatsaechlich anfasst."""

    def __init__(self, size=256):
        yy, xx = np.mgrid[0:size, 0:size]
        self.H = (200 * np.sin(xx / (size / 6.4)) * np.cos(yy / (size / 7.3))
                  + 100).astype(np.float32)
        self.size = size
        self.city = np.zeros((size, size), dtype=bool)
        self.city[size // 3:size // 3 + 20, size // 3:size // 3 + 20] = True
        self.roads = [[(i, size // 2) for i in range(10, size - 10, 3)]]

    def get_terrain_data_combined(self, key):
        return self.H if key == "heightmap" else None

    def get_terrain_data(self, key):
        return self.H if key == "heightmap" else None

    def get_settlement_data(self, key):
        return {"city_mask": self.city, "roads": self.roads}.get(key)

    def get_water_data(self, key):
        return None

    def get_geology_data(self, key):
        return None

    def get_weather_data(self, key):
        return None

    def get_biome_data(self, key):
        return None


def run_groesse_und_verfahren():
    ok = check(f"Exportkante ist {EXPORT_KANTENLAENGE_PX}",
               EXPORT_KANTENLAENGE_PX == 2048)

    klassen = np.repeat(np.arange(9, dtype=np.int32)[None, :], 9, axis=0)
    klassen = np.kron(klassen, np.ones((16, 16), dtype=np.int32))
    gross, vermerk = _auf_exportgroesse(klassen, "categorical")
    ok &= check("kategorischer Layer hat Zielgroesse",
                gross.shape == (EXPORT_KANTENLAENGE_PX, EXPORT_KANTENLAENGE_PX),
                f"{gross.shape}, {vermerk}")
    ok &= check("kategorischer Layer erfindet KEINE neuen Klassen",
                set(np.unique(gross).tolist()) <= set(np.unique(klassen).tolist()),
                f"vorher {sorted(np.unique(klassen).tolist())}, "
                f"nachher {sorted(np.unique(gross).tolist())}")
    ok &= check("und er wurde per naechstem Nachbarn vergroessert",
                "Nachbar" in vermerk, vermerk)

    glatt = np.linspace(0, 100, 64 * 64).reshape(64, 64).astype(np.float32)
    gross_g, vermerk_g = _auf_exportgroesse(glatt, "scalar")
    ok &= check("skalarer Layer wird bilinear vergroessert",
                "bilinear" in vermerk_g, vermerk_g)
    ok &= check("Wertebereich bleibt erhalten",
                abs(float(gross_g.min()) - float(glatt.min())) < 1e-3
                and abs(float(gross_g.max()) - float(glatt.max())) < 1e-3,
                f"{gross_g.min():.2f}..{gross_g.max():.2f}")

    schon_richtig = np.zeros((EXPORT_KANTENLAENGE_PX, EXPORT_KANTENLAENGE_PX),
                             dtype=np.float32)
    _u, v = _auf_exportgroesse(schon_richtig, "scalar")
    ok &= check("bereits passende Groesse wird nicht angefasst", v == "nativ", v)
    return ok


def run_daempfungsmaske():
    dlm = FakeDLM(256)
    mpp = 21300.0 / 256
    maske = daempfungsmaske(dlm, mpp)
    ok = check("Maske entsteht", maske is not None)
    if maske is None:
        return False
    ok &= check("Maske liegt in [0,1]",
                float(maske.min()) >= 0.0 and float(maske.max()) <= 1.0,
                f"{maske.min():.2f}..{maske.max():.2f}")
    ok &= check("es gibt geschuetzte Flaechen (Wert 0)",
                bool((maske <= 0.001).any()),
                f"{float((maske <= 0.001).mean()) * 100:.1f} % der Karte")
    ok &= check("es gibt volle Wildnis (Wert 1)", bool((maske >= 0.999).any()),
                f"{float((maske >= 0.999).mean()) * 100:.1f} % der Karte")

    # Auf der Stadtflaeche MUSS gedaempft sein - sonst steht der Grundriss
    # schief, sobald die Engine Hoehenrauschen addiert.
    stadt = dlm.city
    ok &= check("Stadtflaeche ist gedaempft",
                float(maske[stadt].max()) < 0.5,
                f"hoechster Wert auf der Stadt {float(maske[stadt].max()):.3f}")
    wy, wx = 256 // 2, 100
    ok &= check("Weg ist gedaempft", float(maske[wy, wx]) < 0.5,
                f"Wert am Weg {float(maske[wy, wx]):.3f}")
    return ok


def run_export_laeuft_durch():
    from PIL import Image

    dlm = FakeDLM(256)
    ordner = tempfile.mkdtemp(prefix="mapexport_test_")
    try:
        erfolg, meldung, ziel = export_all_layers(dlm, None, ordner, "probe")
        ok = check("Export meldet Erfolg", erfolg, meldung)
        if not ok:
            return False
        with open(os.path.join(ziel, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        ok &= check("Manifest nennt die Exportkante",
                    manifest.get("export_kantenlaenge_px") == EXPORT_KANTENLAENGE_PX)
        ok &= check("Daempfungsmaske ist dabei",
                    "noise_damping_mask" in manifest["layers"],
                    ", ".join(sorted(manifest["layers"])[:6]))
        falsch = []
        for name, eintrag in manifest["layers"].items():
            bild = Image.open(os.path.join(ziel, eintrag["file"]))
            if bild.size != (EXPORT_KANTENLAENGE_PX, EXPORT_KANTENLAENGE_PX):
                falsch.append(f"{name} {bild.size}")
        ok &= check("alle Bilder haben die Exportgroesse", not falsch,
                    "; ".join(falsch) if falsch else
                    f"{len(manifest['layers'])} Bilder geprueft")
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_vektordaten():
    """13.5 - Wege und Orte muessen in METERN herauskommen, nicht in Pixeln."""
    dlm = FakeDLM(256)
    mpp = 21300.0 / 256
    v = vektordaten(dlm, mpp)
    ok = check("Einheit ist Meter", v.get("einheit") == "meter")
    ok &= check("Wege sind dabei", len(v["wege"]) == 1, f"{len(v['wege'])}")
    if v["wege"]:
        erster = v["wege"][0][0]
        # Der erste Wegpunkt liegt bei Pixel (10, 128)
        ok &= check("Wegpunkt ist in Meter umgerechnet",
                    abs(erster[0] - 10 * mpp) < 0.5
                    and abs(erster[1] - 128 * mpp) < 0.5,
                    f"{erster} erwartet ~[{10*mpp:.1f}, {128*mpp:.1f}]")
    # Was fehlt, MUSS begruendet sein - sonst ist "leer" von "kaputt" nicht
    # zu unterscheiden.
    ok &= check("Fehlendes ist begruendet", len(v["fehlt"]) > 0,
                f"{len(v['fehlt'])} Eintraege")
    ok &= check("Fluesse sind ausdruecklich benannt",
                any("fluesse" in eintrag for eintrag in v["fehlt"]),
                "; ".join(v["fehlt"])[:90])
    return ok


def main():
    print("=" * 70)
    print("Export auf 2048 (13.1) und Daempfungsmaske (13.2)")
    print("=" * 70)
    ergebnisse = []
    for name, funktion in [("Groesse und Verfahren", run_groesse_und_verfahren),
                           ("Daempfungsmaske", run_daempfungsmaske),
                           ("Vektordaten", run_vektordaten),
                           ("Export laeuft durch", run_export_laeuft_durch)]:
        print(f"\n--- {name} ---")
        ergebnisse.append(funktion())
    print("\n" + "=" * 70)
    fehlend = ergebnisse.count(False)
    print(f"{len(ergebnisse) - fehlend}/{len(ergebnisse)} Gruppen gruen")
    return 0 if fehlend == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
