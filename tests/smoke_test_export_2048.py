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
                                  SEESPIEGEL_STANDARD_M, _auf_exportgroesse,
                                  daempfungsmaske, export_all_layers,
                                  vektordaten, wassertiefe)


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
        # Ticket #37: ein Flusslinienzug wie ihn core.fluss_export.fluss_linien()
        # liefert - Punkte in [x_px, y_px], wie "roads" oben.
        self.river_lines = [{"punkte": [[5, 20], [5, 80], [5, 140]],
                             "ordnung": 2, "breite_m": 12.5}]

        # --- Eignungsfeld und Wasser, wie sie die echte Pipeline liefert ---
        #
        # Die Hoehenkarte oben geht von -100 bis +300 m; alles unter dem
        # Meeresspiegel wird hier als Meer markiert, damit die Wassertiefe
        # etwas zu rechnen hat.
        self.super_mask = np.zeros((size, size), dtype=np.uint8)
        self.super_mask[self.H < SEESPIEGEL_STANDARD_M] = 15      # ocean
        self.water_map = np.zeros((size, size), dtype=np.float32)
        self.water_map[size // 4:size // 4 + 6, :] = 3.5          # ein Fluss

        # Drei Biom-Kennungen je Ort mit Anteilen, die auf 1 summieren.
        rng = np.random.default_rng(4711)
        self.top3_ids = rng.integers(0, 15, (size, size, 3), dtype=np.uint8)
        roh = rng.random((size, size, 3)).astype(np.float32) + 0.05
        roh = np.sort(roh, axis=2)[:, :, ::-1]
        self.top3_anteil = (roh / roh.sum(axis=2, keepdims=True)).astype(np.float32)
        self.eindeutigkeit = rng.random((size, size)).astype(np.float32) * 3.0

    def get_terrain_data_combined(self, key):
        return self.H if key == "heightmap" else None

    def get_terrain_data(self, key):
        if key == "heightmap":
            return self.H
        if key == "river_lines":
            return self.river_lines
        return None

    def get_settlement_data(self, key):
        return {"city_mask": self.city, "roads": self.roads}.get(key)

    def get_water_data(self, key):
        return self.water_map if key == "water_map" else None

    def get_geology_data(self, key):
        return None

    def get_weather_data(self, key):
        return None

    def get_biome_data(self, key):
        return {"super_biome_mask": self.super_mask,
                "biom_top3_ids": self.top3_ids,
                "biom_top3_anteil": self.top3_anteil,
                "biom_eindeutigkeit": self.eindeutigkeit}.get(key)


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
        bilder = 0
        for name, eintrag in manifest["layers"].items():
            # Die Hoehenkarte ist kein Bild mehr, sondern eine rohe
            # 16-Bit-Datei - sie wird in run_godot_formate() geprueft.
            if eintrag["kind"] == "hoehe_r16":
                continue
            bilder += 1
            bild = Image.open(os.path.join(ziel, eintrag["file"]))
            if bild.size != (EXPORT_KANTENLAENGE_PX, EXPORT_KANTENLAENGE_PX):
                falsch.append(f"{name} {bild.size}")
        ok &= check("alle Bilder haben die Exportgroesse", not falsch,
                    "; ".join(falsch) if falsch else
                    f"{bilder} Bilder geprueft")
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

    # TICKET #37: Fluesse sind jetzt kein "fehlt"-Eintrag mehr, sondern
    # echte Linienzuege mit Ordnung und Breite - vorher stand hier die
    # Gegenprobe (Fluesse ausdruecklich als fehlend benannt).
    ok &= check("Fluesse sind dabei", len(v["fluesse"]) == 1,
                f"{len(v['fluesse'])}")
    ok &= check("Fluesse sind NICHT mehr unter 'fehlt' gelistet",
                not any("fluesse" in eintrag for eintrag in v["fehlt"]),
                "; ".join(v["fehlt"])[:90])
    if v["fluesse"]:
        fluss = v["fluesse"][0]
        erster = fluss["punkte"][0]
        # Der erste Flusspunkt liegt bei Pixel (5, 20) - [x_px, y_px].
        ok &= check("Flusspunkt ist in Meter umgerechnet",
                    abs(erster[0] - 5 * mpp) < 0.5
                    and abs(erster[1] - 20 * mpp) < 0.5,
                    f"{erster} erwartet ~[{5*mpp:.1f}, {20*mpp:.1f}]")
        ok &= check("Flussordnung und -breite sind uebernommen",
                    fluss.get("ordnung") == 2
                    and abs(fluss.get("breite_m", 0.0) - 12.5) < 0.5,
                    f"{fluss}")
    return ok


def run_godot_formate():
    """
    Die Zusicherungen, an denen der Godot-Import haengt.

    ZWEI ENDEN GEGENEINANDER: die Hoehenkarte wird geschrieben UND wieder
    eingelesen und in Meter zurueckgerechnet. Eine Pruefung nur auf "Datei
    ist da und hat die richtige Groesse" wuerde eine vertauschte
    Bytereihenfolge oder eine falsche Normalisierung nicht bemerken - in
    Godot saehe man dann verrauschtes Gebirge und wuesste nicht, warum.
    """
    from PIL import Image

    dlm = FakeDLM(256)
    ordner = tempfile.mkdtemp(prefix="mapexport_godot_")
    try:
        erfolg, meldung, ziel = export_all_layers(dlm, None, ordner, "probe")
        if not check("Export meldet Erfolg", erfolg, meldung):
            return False
        with open(os.path.join(ziel, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)
        schichten = manifest["layers"]

        # --- Hoehenkarte als rohe 16-Bit-Datei ---
        h = schichten.get("heightmap")
        ok = check("Hoehenkarte ist eine .r16",
                   h is not None and h["file"].endswith(".r16"),
                   h["file"] if h else "fehlt")
        if not ok:
            return False
        pfad = os.path.join(ziel, h["file"])
        bytes_soll = EXPORT_KANTENLAENGE_PX * EXPORT_KANTENLAENGE_PX * 2
        ok &= check("Dateigroesse passt genau",
                    os.path.getsize(pfad) == bytes_soll,
                    f"{os.path.getsize(pfad)} statt {bytes_soll}")
        ok &= check("Manifest nennt die Wertespanne",
                    "value_min" in h and "value_max" in h,
                    f"{h.get('value_min')} .. {h.get('value_max')}")

        roh = np.fromfile(pfad, dtype="<u2").reshape(
            EXPORT_KANTENLAENGE_PX, EXPORT_KANTENLAENGE_PX)
        zurueck = h["value_min"] + roh / 65535.0 * (h["value_max"] - h["value_min"])
        original, _v = _auf_exportgroesse(dlm.H, "hoehe_r16")
        abweichung = float(np.abs(zurueck - original).max())
        stufe = (h["value_max"] - h["value_min"]) / 65535.0
        ok &= check("Hoehen kommen in Metern zurueck", abweichung <= stufe,
                    f"groesste Abweichung {abweichung:.5f} m, "
                    f"eine Stufe ist {stufe:.5f} m")

        # --- Eignungsfeld ---
        for name in ("biom_top3_ids", "biom_top3_anteil", "biom_eindeutigkeit"):
            ok &= check(f"{name} ist im Export", name in schichten,
                        ", ".join(sorted(schichten)))
        if "biom_top3_anteil" in schichten:
            bild = np.asarray(Image.open(
                os.path.join(ziel, schichten["biom_top3_anteil"]["file"])))
            ok &= check("Anteile sind ein RGB-Bild",
                        bild.ndim == 3 and bild.shape[2] == 3, str(bild.shape))
            summe = bild.astype(np.int32).sum(axis=2)
            ok &= check("die drei Anteile summieren auf 100 %",
                        int(np.abs(summe - 255).max()) <= 2,
                        f"groesste Abweichung {int(np.abs(summe - 255).max())}/255")
            ok &= check("Platz 1 ist nirgends kleiner als Platz 2",
                        bool((bild[:, :, 0] >= bild[:, :, 1]).all()))
        if "biom_top3_ids" in schichten:
            ids = np.asarray(Image.open(
                os.path.join(ziel, schichten["biom_top3_ids"]["file"])))
            # Kennungen duerfen beim Hochrechnen NICHT gemittelt werden -
            # zwischen Biom 3 und Biom 7 liegt kein Biom 5.
            ok &= check("Kennungen erfinden keine neuen Biome",
                        set(np.unique(ids).tolist())
                        <= set(np.unique(dlm.top3_ids).tolist()))
        if "biom_eindeutigkeit" in schichten:
            e = Image.open(os.path.join(
                ziel, schichten["biom_eindeutigkeit"]["file"]))
            ok &= check("Eindeutigkeit ist echtes 8-Bit-Grau", e.mode == "L",
                        e.mode)

        # --- Daempfungsmaske jetzt ebenfalls 8 Bit ---
        if "noise_damping_mask" in schichten:
            m = Image.open(os.path.join(
                ziel, schichten["noise_damping_mask"]["file"]))
            ok &= check("Daempfungsmaske ist echtes 8-Bit-Grau", m.mode == "L",
                        m.mode)

        # --- Wassertiefe ---
        w = schichten.get("wassertiefe")
        ok &= check("Wassertiefe ist im Export", w is not None)
        if w is not None:
            ok &= check("Wassertiefe hat eine benannte Spanne in Metern",
                        w.get("value_max", 0) > 0,
                        f"0 .. {w.get('value_max')} m")
        return ok
    finally:
        shutil.rmtree(ordner, ignore_errors=True)


def run_wassertiefe_inhaltlich():
    """
    Meer und Binnengewaesser haben VERSCHIEDENE Wasserspiegel.

    Wuerde man auch fuer einen Bergsee "Meeresspiegel minus Hoehe" rechnen,
    kaeme im Bergland ueberall 0 heraus und jeder See saehe aus wie eine
    Pfuetze. Darum zwei Quellen, und darum werden hier beide einzeln
    geprueft.
    """
    dlm = FakeDLM(256)
    tiefe = wassertiefe(dlm, None, None)
    ok = check("Wassertiefe entsteht", tiefe is not None)
    if tiefe is None:
        return False
    hoch = dlm.H > 200
    if hoch.any():
        ok &= check("auf hohem Land ist die Tiefe 0",
                    float(tiefe[hoch].max()) == 0.0)
    meer = dlm.super_mask == 15
    ok &= check("im Meer ist die Tiefe positiv", bool((tiefe[meer] > 0).any()),
                f"tiefste Stelle {float(tiefe[meer].max()):.1f} m")
    ok &= check("Meerestiefe ist Meeresspiegel minus Hoehe",
                abs(float(tiefe[meer].max())
                    - float(SEESPIEGEL_STANDARD_M - dlm.H[meer].min())) < 1e-3)
    fluss = dlm.water_map > 0
    ok &= check("der Fluss bekommt seine eigene Tiefe",
                float(tiefe[fluss].max()) >= 3.5,
                f"{float(tiefe[fluss].max()):.1f} m")
    return ok


def main():
    print("=" * 70)
    print("Export auf 2048 (13.1) und Daempfungsmaske (13.2)")
    print("=" * 70)
    ergebnisse = []
    for name, funktion in [("Groesse und Verfahren", run_groesse_und_verfahren),
                           ("Daempfungsmaske", run_daempfungsmaske),
                           ("Vektordaten", run_vektordaten),
                           ("Export laeuft durch", run_export_laeuft_durch),
                           ("Godot-Formate", run_godot_formate),
                           ("Wassertiefe inhaltlich", run_wassertiefe_inhaltlich)]:
        print(f"\n--- {name} ---")
        ergebnisse.append(funktion())
    print("\n" + "=" * 70)
    fehlend = ergebnisse.count(False)
    print(f"{len(ergebnisse) - fehlend}/{len(ergebnisse)} Gruppen gruen")
    return 0 if fehlend == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
