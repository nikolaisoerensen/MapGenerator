# Die Welt-Naht: `welt_backen()` und `welt_laden()`

Ticket #38. Erste Fassung 2026-09-22 (Nachtbetrieb), am selben Tag mit der
zweiten, unabhängig entstandenen Fassung zusammengeführt (siehe
"Zusammenführung zweier Fassungen" unten). Status: headless verifiziert,
visuell NICHT bestätigt, und noch an keinen Knopf in der Oberfläche
angeschlossen (siehe "Was fehlt").

## Warum

Der Karteneditor konnte bislang eine Welt erzeugen, aber nicht wieder
finden - jede generierte Welt ging beim Schließen des Programms verloren.
Es gab kein Format, das für Godot/Terrain3D gedacht war, und die
bestehenden Export-Funktionen (PNG-, OBJ-, Layer-, Parameter-Export) waren
einzeln aufrufbar, aber keine davon deckt "alles, was zu einer Welt gehört,
an einem Ort" ab, und keine hat einen Ladeweg zurück.

## Die Naht

**Naht**: die eine Stelle, an der jeder Aufrufer - ein Programmneustart,
später ein Godot-Import - dieselbe, vollständige Auskunft über eine
generierte Welt abholt, statt dass jede Stelle im Code sich ihre eigene
Teilmenge zusammensucht.

Die Naht ist **`core/welt_io.py`**:

```python
from core.welt_io import welt_backen, welt_laden

manifest = welt_backen(pfad, data_lod_manager, parameter_manager)
manifest = welt_laden(pfad, data_lod_manager, parameter_manager)
```

Sie liegt in `core/`, nicht in einem Reiter, und ist damit **Qt-frei**: ein
Skript, ein Test oder ein späterer Kommandozeilen-Export kann eine Welt
schreiben und lesen, ohne dass eine Oberfläche läuft.

`gui/tabs/overview_tab.py` behält zwei gleichnamige Methoden, die aber nur
noch **weiterleiten**:

```python
manifest = reiter.welt_backen(pfad)   # -> core.welt_io.welt_backen(...)
daten    = reiter.welt_laden(pfad)    # -> core.welt_io.welt_laden(...)
```

Der einzige Unterschied: `OverviewTab.welt_laden()` gibt zusätzlich die
Antwortform `{kategorie: {feld: wert}}` zurück, die seine bisherigen
Aufrufer kannten - und liest sie dafür **aus dem Manager zurück**
(`get_all_data()`), nicht neben ihm zusammengebaut. Was dort ankommt, ist
also genau das, was das laufende Programm danach auch sieht.

`WeltLadenFehler` und `WeltBackenFehler` sind weiterhin aus
`gui/tabs/overview_tab.py` importierbar - als Re-Export derselben Klasse,
nicht als Nachbau. Ein bestehendes `except WeltLadenFehler` fängt also auch
das, was `core/welt_io.py` wirft.

### Woher die Daten kommen

| Baustein | Herkunft | Rolle |
|---|---|---|
| `DataLODManager.get_all_data(kategorie)` | in `managers/data_lod_manager.py` für dieses Ticket ergänzt | liefert den vollständigen Schnappschuss einer Kategorie beim aktuellen LOD |
| `DataLODManager.set_all_data(kategorie, daten, lod_level)` | ebenda | schreibt ihn zurück, Feld für Feld über den bestehenden `_set_data_lod()`-Pfad |
| `ParameterManager.get_all_parameters()` / `set_tab_parameters()` | bereits vorhanden | sichert und stellt die eingestellten Regler-Werte wieder her |
| `export_all_layers()` | `gui/utils/map_export.py`, bereits vorhanden | der Godot/Terrain3D-Export, unverändert in einen `godot/`-Unterordner aufgerufen |
| `export_single_map_png()` / `export_world_statistics_txt()` | `gui/utils/map_export.py` | die Vorschaubilder und die Textstatistik |

Keine dieser Funktionen wird verändert oder dupliziert
(Akzeptanzkriterium 6).

**Warum `get_all_data()` und keine Feldliste:** die frühere Fassung führte
eine von Hand gepflegte Liste aller Felder je Generator mit. Ergänzt ein
Generator einen neuen Data-Key, fällt er aus einer solchen Liste heraus -
lautlos, denn die Welt wird ja weiterhin erfolgreich geschrieben, nur ohne
dieses Feld. Genau der Fehlertyp, vor dem CLAUDE.md warnt.
`get_all_data()` fragt statt dessen den Speicher selbst, und was darin
liegt, kommt mit.

## Was auf der Platte liegt

```
<pfad>/
  welt_manifest.json          - was in dieser Welt steckt, je Kategorie mit LOD
  zustand/
    terrain.pkl               - ein Schnappschuss je Kategorie (pickle, verlustfrei)
    geology.pkl
    settlement.pkl
    weather.pkl
    erosion.pkl
    water.pkl
    biome.pkl
    globals.json              - map_seed, map_distance_km, map_latitude
    parameter.json            - die Regler-Werte aller Tabs (optional)
  godot/                      - der bestehende Terrain3D-Export
  vorschau/                   - Einbahnstraße: PNGs, statistik.txt,
                                zustand_lesbar.json
```

Drei Ordner für drei Zwecke:

- **`zustand/`**: verlustfrei, für den Rundgang backen→laden. `pickle`
  bringt auch die `Location`-Dataclasses als **Objekte** zurück, nicht als
  Wörterbücher - der Manager bekommt beim Laden also genau das, was er beim
  Speichern hatte. Je Kategorie eine Datei, damit eine beschädigte oder
  fehlende Kategorie beim Laden **namentlich** benannt werden kann.
- **`godot/`**: die Godot/Terrain3D-Ansicht derselben Welt, absichtlich in
  einem anderen, verlustbehafteten Format (siehe "Godot-Bedarf" unten).
- **`vorschau/`**: für Menschen, nicht fürs Programm. `welt_laden()` liest
  hier **nichts** zurück. Höhen-, Hang- und Biomkarte als PNG, eine
  Textstatistik, und `zustand_lesbar.json` - der Nicht-Array-Zustand
  (Siedlungs-/Landmark-/Roadsite-Listen, Skalare) im Klartext, mit Arrays
  nur als `{"array": true, "shape": ..., "dtype": ...}` vermerkt. Diese
  lesbare Fassung ist aus der zweiten Fassung übernommen, die sie als
  `welt_zustand.json` führte; ohne sie wäre beim Umstieg auf pickle die
  einzige außerhalb von Python lesbare Form der Siedlungsdaten verloren
  gegangen (`docs/spezifikation/14_SIEDLUNGEN.md` Abschnitt 7).

## Das LOD gehört ins Manifest

`get_all_data()` liefert immer den Schnappschuss des **aktuellen**, also
höchsten gespeicherten LOD einer Kategorie. Deshalb hält das Manifest je
Kategorie fest, welches LOD das war:

```json
"kategorien": {
  "terrain": {"vorhanden": true, "keys": ["heightmap", "slopemap"], "lod": 3}
}
```

`welt_laden()` stellt genau dieses LOD wieder her und **verlangt** den
Eintrag: eine Welt aus einer älteren Fassung ohne `lod` wird laut
abgelehnt, statt auf LOD 1 zurückzufallen. Der Grund ist kein Formalismus -
läge die geladene Welt unterhalb eines schon vorhandenen höheren LOD, dann
lieferte `get_all_data()` weiterhin die alten Daten unter denselben
Schlüsselnamen: das Programm zeigte die alte Welt und meldete "geladen".

## Zurückgeschrieben: alle sieben

`welt_laden()` schreibt **alle sieben** Kategorien in den laufenden
`DataLODManager` zurück (`terrain`, `geology`, `settlement`, `weather`,
`erosion`, `water`, `biome`). Die zweite Fassung dieser Naht schrieb nur
`terrain` und `geology` zurück und gab die übrigen fünf bloß aus der
Methode heraus - eine geladene Welt war im Programm damit halb unsichtbar.

Dazu gehört ein Sonderfall: das zusammengesetzte **`terrain_data_object`**.
Von den sieben `lod_{n}_<kategorie>_data_object`-Schlüsseln des Managers
wird genau einer je wieder gelesen - der von `terrain`, über
`get_terrain_data("complete")`, woran der `GenerationOrchestrator` hängt.
Die anderen sechs werden geschrieben und nie abgefragt. `set_all_data()`
setzt das Terrain-Objekt deshalb aus den Einzelfeldern neu zusammen, wenn
es im Schnappschuss nicht mitkam (`_terrain_objekt_nachziehen()`), und
**loggt jeden Ausgang** - auch den Fall "konnte nicht gebaut werden, weil
keine heightmap da ist".

## Kein stiller Ersatzwert

`welt_laden()` bricht mit `WeltLadenFehler` ab, sobald eine dieser fünf
Prüfungen anschlägt:

1. es gibt kein `welt_manifest.json` unter dem Pfad;
2. das Manifest nennt eine Kategorie als vorhanden, deren `zustand/*.pkl`
   fehlt oder nicht lesbar ist;
3. eine Datei enthält nicht alle Schlüssel, die das Manifest für sie nennt;
4. der Manager hat beim Zurückschreiben Schlüssel **abgelehnt** -
   `set_all_data()` überspringt ungültige Einzelfelder nur mit einer
   Logzeile (bestehender Vertrag von `_set_data_lod()`, an dem über 40
   Aufrufer hängen), deshalb liest `welt_laden()` danach zurück und
   vergleicht die Schlüsselmenge;
5. der Manager liefert nach dem Schreiben zwar dieselben Schlüssel, aber
   **andere Werte** (Identitätsprüfung, nicht nur Namensgleichheit) - das
   ist der Verschattungsfall aus dem LOD-Abschnitt oben. Die Meldung nennt
   die betroffenen Felder und sagt, was zu tun ist
   (`invalidate_cache_lod()` oder ein frischer Manager).

Es gibt keinen Pfad, der ein fehlendes Feld durch `None`, `0` oder eine
leere Struktur ersetzt. Prüfung 5 ist dabei genau die Lehre aus CLAUDE.md
("zwei Enden der Kette gegeneinander messen"): die Prüfungen 1-4 waren
einzeln grün, während die geladene Welt unsichtbar blieb.

Nicht hart geprüft wird, was **Einbahnstraße** ist: scheitert der Godot-
oder Vorschau-Export, ist das kein Abbruch - der Misserfolg steht aber mit
Meldung im Manifest und im Log, nicht nur "irgendwie nicht da".

## Godot-Bedarf: was Terrain3D konkret erwartet

Ticket-Vorgabe war, das nicht zu behaupten, sondern zu recherchieren. Laut
der offiziellen Terrain3D-Dokumentation (TokisanGames/Terrain3D,
[Heightmaps](https://terrain3d.readthedocs.io/en/stable/docs/heightmaps.html),
[Control Map Format](https://terrain3d.readthedocs.io/en/stable/docs/controlmap_format.html)):

- **Höhenkarte**: `.r16` (16-Bit unsigned int, roh, ohne Header - Min/Max
  und Bildgröße müssen extern mitgeführt werden) oder `.exr` (32-Bit-Float,
  bevorzugt, da ohne Wertebereichs-Remapping). Der bestehende
  `export_all_layers()`-Export schreibt `heightmap.r16` - genau dieses
  Format, mit dem Wertebereich (Min/Max) im mitgelieferten `manifest.json`
  im selben Ordner festgehalten (das externe Mitführen, das Terrain3D
  verlangt, ist damit erledigt).
- **Kontrollkarte** (Textur-Zuordnung/Blending): ein 32-Bit-Wert pro
  Bildpunkt (`FORMAT_RF`, intern als `R32_UINT` gelesen) - darin codiert:
  eine Basis-Textur-ID (0-31), eine Overlay-Textur-ID und ein 8-Bit-
  Blend-Wert zwischen beiden. Das ist NICHT dasselbe wie eine gewöhnliche
  RGB-Textur-Zuordnung.
- **Texturzuordnung**: die Textur-IDs 0-31 in der Kontrollkarte müssen mit
  der Reihenfolge der Textur-Assets im Godot-Projekt übereinstimmen - das
  ist eine Projekt-Konvention auf Godot-Seite, keine Datei, die dieser
  Export selbst erzeugen kann.

**Offener Punkt** (siehe "Was fehlt" unten): `export_all_layers()`
schreibt heute `rock_map.png` (RGB, sedimentär/magmatisch/metamorph als
Kanäle) statt einer Terrain3D-Kontrollkarte im beschriebenen 32-Bit-Format.
Das vorhandene Rohmaterial (Gesteinsart, Höhe, Hangneigung) reicht aus, um
eine solche Kontrollkarte zu bauen - das eigentliche Umrechnen in das
32-Bit-Packformat ist aber nicht Teil von Ticket #38 (das nur die Naht
`welt_backen`/`welt_laden` verlangt, nicht einen neuen Godot-Export) und
wird hier nur benannt, nicht umgesetzt.

## Siedlungsnaht (Verweis, nicht Neuerfindung)

`docs/spezifikation/14_SIEDLUNGEN.md` Abschnitt 7 dokumentiert bereits, welche 8 Felder das
Spiel aus Siedlungsdaten liest (`city_id`, `city_center`,
`city_boundary_polygons`, `road_entry_points`, `city_size`/`house_count`,
`city_size`/`radius`, `city_type`, `rank`, `culture`). 6 davon liefert die
`Location`-Dataclass (`core/settlement_generator.py`) bereits, 2 sind noch
offen (`city_boundary_polygons` → Ticket #72, `road_entry_points` →
Ticket #73). `welt_backen`/`welt_laden` transportieren einfach die
komplette `Location`-Liste über den Kategorie-Schnappschuss - sie erfinden
keine eigene Siedlungs-Feldliste, sondern reichen §6 unverändert durch.

## Die `erosion`-Lücke ist geschlossen

Die frühere Fassung sammelte ihre Felder über
`OverviewTab.collect_all_available_data()`. Diese Methode legt den
Schlüssel `"erosion": {}` an, befüllt ihn aber nie - die Erosionsdaten
fehlten in jeder gebackenen Welt, ohne dass etwas warnte. Über
`get_all_data("erosion")` kommen sie jetzt mit; der Bug in
`collect_all_available_data()` selbst besteht unverändert weiter und
betrifft weiterhin die Anzeige der Weltstatistik, aber nicht mehr das
Speichern.

## Zusammenführung zweier Fassungen (2026-09-22)

Ticket #38 wurde versehentlich zweimal gelöst: einmal als `core/welt_io.py`
(Commit `651c927`, Nachtbranch) und einmal direkt in
`gui/tabs/overview_tab.py` (Commit `c571db4`, auf `main`). Behalten wurde
`core/welt_io.py`, weil die andere Fassung nur zwei von sieben Kategorien
zurückschrieb und ihre Begründung dafür - die übrigen fünf bräuchten
typisierte Dataclasses - nicht trägt: `set_weather_data_complete_lod()`
zerlegt das Dataclass-Objekt seinerseits nur in Einzelfelder, und
`get_weather_data_lod()` liest genau diese Einzelfelder wieder.

Aus der verworfenen Fassung übernommen wurden:

- die lesbare Klartextfassung des Nicht-Array-Zustands (jetzt
  `vorschau/zustand_lesbar.json`, dort `welt_zustand.json`);
- die Antwortform `{kategorie: {feld: wert}}` von `welt_laden()`;
- ihr Test, der auf die neue Naht umgebaut wurde
  (`tests/smoke_test_welt_backen_laden.py`).

Der Godot-Export lag dort unter `godot/welt/`, hier liegt er flach unter
`godot/`.

## Prüfung

Zwei Tests, absichtlich getrennt:

| Test | prüft | braucht |
|---|---|---|
| `tests/smoke_test_welt_io_roundtrip.py` | `core/welt_io.py` für sich: bitgenauer Rundlauf, LOD-Erhalt, die lauten Fehlerfälle | kein Qt, Manager als Test-Double |
| `tests/smoke_test_welt_backen_laden.py` | die GUI-Weiterleitung **gegen den echten `DataLODManager`**: alle sieben Kategorien kommen an, `get_terrain_data("complete")` ist gefüllt, Godot-Ordner entsteht, Verschattung schlägt laut fehl | Qt (offscreen) |

Beide laufen mit echten Kartengrößen (128), nicht mit ausgedachten -
CLAUDE.md, "Gruene Tests koennen eine tote Funktion verdecken".

## Was fehlt (für eine spätere Sitzung)

1. **Kein Knopf in der Oberfläche.** Nichts in `gui/` oder `main.py` ruft
   `welt_backen()`/`welt_laden()` auf - die Naht ist gebaut und geprüft,
   aber nur aus Skripten und Tests erreichbar. Ein Menüeintrag
   "Welt speichern"/"Welt laden" fehlt.
2. **Visuell nicht bestätigt.** Ob eine geladene Welt in der laufenden App
   auch tatsächlich sichtbar wird (Reiter neu zeichnen, 3D-Netz neu
   aufbauen), wurde nicht getestet - das braucht die laufende Qt-App und
   eine echte generierte Welt.
3. **Terrain3D-Kontrollkarte** existiert noch nicht - nur die Höhenkarte
   ist im terrain3d-gerechten Format vorhanden (siehe "Godot-Bedarf").
4. **`pickle` ist kein Austauschformat.** `zustand/*.pkl` ist an Python und
   an die aktuellen Klassendefinitionen gebunden; eine umbenannte
   `Location`-Dataclass macht alte Welten unlesbar. Für den Rundlauf
   innerhalb dieses Programms ist das richtig (verlustfrei, Objekte bleiben
   Objekte), für einen Austausch mit Godot ist es das nicht - dafür ist
   `godot/` da.
