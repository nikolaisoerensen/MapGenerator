"""
Path: core/wegarten.py

Wegarten-Staffelung und Messung fuer das Wegenetz, Ticket #41
("Wegekostenstaffelung nach Wegart kalibrieren").

WAS DAS IST. `core/settlement_generator.py::calculate_road_network()` baut
das Wegenetz zwischen Siedlungen ueber A*-Pathfinding auf einem Kostenfeld
(bau_kostenfeld()). Bis Ticket #41 kannte dieses Kostenfeld nur EINEN
Rabatt-Schritt (WEGERABATT=0.4, sobald ein Pixel ueberhaupt schon einmal
Teil eines gebauten Weges war). Dieses Modul ersetzt den einen Schritt durch
eine STAFFEL: ein selten benutztes Pixel bleibt ein Pfad, ein oefter
benutztes wird zum Karrenweg, ein vielbenutztes zur Strasse - mit
entsprechend sinkenden Kosten. Die Tabelle selbst steht in
`core/daten/wegarten.toml`, versioniert und ausserhalb des Codes (Abnahme-
kriterium 1 von Ticket #41).

DIE NAHT: `rabattfeld_und_stufen(nutzung, wegarten)` ist der einzige Ort, an
dem ein Nutzungs-Zaehler in ein Kostenfaktor-Feld uebersetzt wird - sowohl
`calculate_road_network()` beim Routen als auch `wegekennzahlen()` bei der
Messung rufen dieselbe Funktion, damit "wie billig ist dieses Pixel" und
"welcher Wegart zaehlt dieses Pixel" nie auseinanderlaufen koennen.

DER REGELKREIS UND SEIN RISIKO. Benutzung macht einen Weg billiger, ein
billigerer Weg zieht mehr Benutzung an (siehe bereitschaft_von()/route() in
calculate_road_network() - der Rabatt wirkt schon auf die Bau-ENTSCHEIDUNG,
nicht erst auf die spaeter gezeichnete Geometrie). Das ist gewuenscht
("Wege buendeln sich zu Hauptstrecken"), kann aber weglaufen: zu grosszuegige
Rabatte lassen einen einzelnen Weg alles an sich ziehen, zu geizige
verhindern jede Buendelung. `wegekennzahlen()` misst das Ergebnis
(Gesamtlaenge, Anteil je Wegart, mittlerer Umwegfaktor, Anzahl Knoten);
`tests/smoke_test_wegarten_kalibrierung.py` haelt es gegen das Band in
`tests/toleranzen.toml` Abschnitt [wegenetz], ueber drei Seeds und drei
Kartengroessen.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEGARTEN_DATEI = Path(__file__).resolve().parent / "daten" / "wegarten.toml"


@dataclass(frozen=True)
class Wegart:
    name: str
    mindestnutzung: int
    kostenfaktor: float


def lade_wegarten(pfad=None):
    """Liest core/daten/wegarten.toml, aufsteigend nach mindestnutzung sortiert.

    Kein stiller Rueckfall: eine fehlende oder kaputte Datei wirft, statt mit
    einer erfundenen Standardtabelle weiterzurechnen - genau der Fehler, vor
    dem CLAUDE.md ("Jeder stille Rueckfall auf einen Ersatzpfad braucht eine
    laute Logzeile") warnt.
    """
    pfad = Path(pfad) if pfad else WEGARTEN_DATEI
    if not pfad.exists():
        raise FileNotFoundError("Wegarten-Tabelle fehlt: %s" % pfad)
    roh = tomllib.loads(pfad.read_text(encoding="utf-8"))
    eintraege = roh.get("wegart", [])
    if not eintraege:
        raise ValueError("Wegarten-Tabelle %s ist leer" % pfad)
    arten = []
    for e in eintraege:
        fehlend = [f for f in ("name", "mindestnutzung", "kostenfaktor") if f not in e]
        if fehlend:
            raise ValueError("Wegart-Eintrag unvollstaendig, es fehlt: %s" % fehlend)
        arten.append(Wegart(name=str(e["name"]),
                            mindestnutzung=int(e["mindestnutzung"]),
                            kostenfaktor=float(e["kostenfaktor"])))
    arten.sort(key=lambda w: w.mindestnutzung)
    if arten[0].mindestnutzung < 1:
        raise ValueError("Die unterste Wegart muss mindestnutzung >= 1 haben "
                         "(0 ist unbenutzter Grund und traegt keine Wegart)")
    for a, b in zip(arten, arten[1:]):
        if not (b.kostenfaktor < a.kostenfaktor):
            raise ValueError(
                "Wegarten muessen mit steigender Nutzung billiger werden: "
                "%s (%.3f) vor %s (%.3f) verletzt das" % (
                    a.name, a.kostenfaktor, b.name, b.kostenfaktor))
    return arten


def rabattfeld_und_stufen(nutzung, wegarten):
    """Aus dem Nutzungs-Zaehler (wie oft ein Pixel schon Teil einer GEBAUTEN
    Route war) das Kostenfaktor-Feld und den Stufenindex je Pixel bauen.

    nutzung: (H,W) int-Array. wegarten: aufsteigend sortierte Liste von
    Wegart (siehe lade_wegarten()).

    Rueckgabe: (rabatt (H,W) float64 - 1.0 wo unbenutzt, stufe (H,W) int32 -
    -1 wo unbenutzt, sonst Index in `wegarten`).

    Aufsteigend iteriert, damit ein Pixel, das mehrere Schwellen zugleich
    erreicht, am Ende auf der HOECHSTEN qualifizierenden Stufe landet - die
    letzte Zuweisung gewinnt.
    """
    nutzung = np.asarray(nutzung)
    rabatt = np.ones(nutzung.shape, dtype=np.float64)
    stufe = np.full(nutzung.shape, -1, dtype=np.int32)
    for idx, art in enumerate(wegarten):
        erreicht = nutzung >= art.mindestnutzung
        rabatt = np.where(erreicht, art.kostenfaktor, rabatt)
        stufe = np.where(erreicht, idx, stufe)
    return rabatt, stufe


def wegekennzahlen(roads, wegnutzung, wegarten, settlements=None):
    """Kennzahlen aus einem fertigen Wegenetz (Ticket #41, Abnahmekriterium 3).

    Parameter:
        roads: List[List[Tuple[float, float]]] - die gebauten (geglaetteten)
            Pfade, wie von calculate_road_network() zurueckgegeben.
        wegnutzung: (H,W) int-Array, wie von calculate_road_network() unter
            self.letzte_wegnutzung abgelegt - derselbe Zaehler, der beim
            Routen die Rabatte steuerte.
        wegarten: Liste von Wegart, siehe lade_wegarten().
        settlements: optional, fuer anzahl_knoten.

    Rueckgabe (dict):
        gesamtlaenge: Summe der Pfadlaengen in Pixeln.
        anzahl_wege: len(roads).
        anteil_je_wegart: {name: Anteil an der benutzten Flaeche}, aus dem
            tatsaechlichen Nutzungs-Zaehler - nicht aus roads neu gerastert,
            damit Messung und Rueckkopplung denselben Zaehler lesen.
        mittlerer_umwegfaktor: Mittelwert aus (Pfadlaenge / Luftlinie) ueber
            alle roads - wie stark das Netz von der direkten Verbindung
            abweicht.
        anzahl_knoten: len(settlements), falls uebergeben, sonst None.
    """
    gesamtlaenge = 0.0
    umwege = []
    for pfad in roads:
        arr = np.asarray(pfad, dtype=np.float64)
        if len(arr) < 2:
            continue
        segmente = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
        laenge = float(segmente.sum())
        gesamtlaenge += laenge
        luftlinie = float(np.hypot(arr[-1, 0] - arr[0, 0], arr[-1, 1] - arr[0, 1]))
        if luftlinie > 1e-6:
            umwege.append(laenge / luftlinie)

    _rabatt, stufe = rabattfeld_und_stufen(wegnutzung, wegarten)
    benutzt = np.asarray(wegnutzung) > 0
    benutzte_px = int(benutzt.sum())
    anteil_je_wegart = {}
    for idx, art in enumerate(wegarten):
        anteil_je_wegart[art.name] = (
            float(np.sum(stufe == idx)) / benutzte_px if benutzte_px > 0 else 0.0)

    return {
        "gesamtlaenge": gesamtlaenge,
        "anzahl_wege": len(roads),
        "anteil_je_wegart": anteil_je_wegart,
        "mittlerer_umwegfaktor": float(np.mean(umwege)) if umwege else float("nan"),
        "anzahl_knoten": len(settlements) if settlements is not None else None,
    }
