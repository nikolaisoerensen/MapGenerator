# Die Welt-Naht: `welt_backen()` und `welt_laden()`

Ticket #38, 2026-09-22 (Nachtbetrieb). Status: erste Fassung, headless
verifiziert per Rundgang-Test, visuell NICHT bestätigt (siehe "Was fehlt"
unten).

## Warum

Der Karteneditor konnte bislang eine Welt erzeugen, aber nicht wieder
finden - jede generierte Welt ging beim Schließen des Programms verloren.
Es gab kein Format, das für Godot/Terrain3D gedacht war, und die sechs
bestehenden Export-Funktionen in `gui/tabs/overview_tab.py` (PNG-Export,
JSON-Export, OBJ-Export, Layer-Export, Parameter-Export) waren einzeln
aufrufbar, aber keine davon deckt "alles, was zu einer Welt gehört, an
einem Ort" ab, und keine hat einen Ladeweg zurück.

## Die Naht

**Naht**: die eine Stelle, an der jeder Aufrufer - ein Programmneustart,
später ein Godot-Import - dieselbe, vollständige Auskunft über eine
generierte Welt abholt, statt dass jede Stelle im Code sich ihre eigene
Teilmenge zusammensucht.

Zwei Methoden auf `OverviewTab` (`gui/tabs/overview_tab.py`):

```python
manifest = reiter.welt_backen(pfad)   # schreibt die Welt nach `pfad`
daten     = reiter.welt_laden(pfad)   # liest sie zurück
```

Beide rufen ausschließlich schon bestehende Bausteine auf:

| Baustein | Herkunft | Rolle in `welt_backen` |
|---|---|---|
| `collect_all_available_data()` | bereits vorhanden, Zeile 188 | sammelt alle Generator-Daten - dieselbe Sammlung, die auch `check_world_completeness()` und `export_world_data()` benutzen |
| `ParameterSummaryWidget.get_all_parameters()` | bereits vorhanden, Zeile ~1105 | sammelt die tatsächlich eingestellten Regler-Werte aller sieben Generator-Tabs |
| `export_all_layers()` | `gui/utils/map_export.py`, bereits vorhanden | der Godot/Terrain3D-Export (siehe unten) - unverändert in einen `godot/`-Unterordner aufgerufen |

Keine dieser drei wird verändert oder dupliziert (Akzeptanzkriterium 6).
Was `welt_backen` NEU hinzufügt, ist nur die Verteilung der gesammelten
Daten auf drei Dateien plus das Manifest, das beschreibt, wo was liegt.

## Was auf der Platte liegt

```
<pfad>/
  welt_manifest.json   - die Feldliste selbst (siehe unten)
  welt_arrays.npz       - jedes numpy-Array, bit-genau (numpy.savez_compressed)
  welt_zustand.json     - Listen/Skalare + alle Generator-Parameter
  godot/welt/           - der bestehende Terrain3D-Export (siehe unten)
```

Zwei getrennte Formate für zwei getrennte Zwecke:

- **`welt_arrays.npz` + `welt_zustand.json`**: verlustfrei, für den
  Rundgang bake→load. `numpy.savez_compressed`/`numpy.load` ist bit-genau -
  im Unterschied zur 16-Bit-normierten PNG, die für den Godot-Export
  benutzt wird und die (bewusst, siehe dort) nicht bit-genau ist.
- **`godot/`**: die Godot/Terrain3D-Ansicht derselben Welt, absichtlich in
  einem anderen, verlustbehafteten Format (siehe "Godot-Bedarf" unten).

## Die Feldliste

Pro Generator, wie von `collect_all_available_data()` gesammelt:

| Generator | Felder | Anmerkung |
|---|---|---|
| `terrain` | `heightmap`, `slopemap`, `shadowmap` | Arrays |
| `geology` | `rock_map`, `hardness_map` | Arrays |
| `settlement` | `settlement_list`, `landmark_list`, `roadsite_list`, `plot_map`, `civ_map` | Listen enthalten `Location`-Objekte (siehe Siedlungsnaht unten), `plot_map`/`civ_map` sind Arrays |
| `weather` | `wind_map`, `temp_map`, `precip_map`, `humid_map` | Arrays |
| `water` | `water_map`, `flow_map`, `flow_speed`, `cross_section`, `soil_moist_map`, `erosion_map`, `sedimentation_map`, `rock_map_updated`, `evaporation_map`, `ocean_outflow`, `water_biomes_map` | überwiegend Arrays, `ocean_outflow` ein Skalar |
| `biome` | `biome_map`, `biome_map_super`, `super_biome_mask` | Arrays |
| `erosion` | **keine** | vorbestehende Lücke, siehe unten - nicht durch dieses Ticket verursacht |

Jedes Feld, das tatsächlich vorhanden war, landet im Manifest unter
`felder.<generator>__<feld>` mit `ablage` (`"arrays"` oder `"zustand"`) und
bei Arrays zusätzlich `shape`/`dtype`. `welt_laden()` geht diese Liste durch
und prüft jeden Eintrag gegen die Datei - fehlt einer, kommt
`WeltLadenFehler` mit dem genauen Feldnamen (siehe "Kein stiller
Ersatzwert" unten).

### Vorbestehende Lücke: `erosion` ist immer leer

`collect_all_available_data()` legt den Schlüssel `"erosion": {}` an, aber
keine Schleife befüllt ihn - dieser Bug existierte schon vor Ticket #38 und
wird hier nur sichtbar gemacht (im Manifest als `hinweis_erosion`
vermerkt), nicht behoben. Wer ihn beheben will, muss in
`collect_all_available_data()` eine Schleife über
`self.data_lod_manager.get_erosion_data(key)` ergänzen - das ist bewusst
NICHT Teil dieses Tickets, weil es außerhalb des Naht-Auftrags liegt.

### Siedlungsnaht (Verweis, nicht Neuerfindung)

`docs/SIEDLUNGEN_ENTWURF.md` §6 dokumentiert bereits, welche 8 Felder das
Spiel aus Siedlungsdaten liest (`city_id`, `city_center`,
`city_boundary_polygons`, `road_entry_points`, `city_size`/`house_count`,
`city_size`/`radius`, `city_type`, `rank`, `culture`). 6 davon liefert die
`Location`-Dataclass (`core/settlement_generator.py`) bereits, 2 sind noch
offen (`city_boundary_polygons` → Ticket #72, `road_entry_points` →
Ticket #73). `welt_backen`/`welt_laden` transportieren einfach die
komplette `Location`-Liste (`settlement_list`/`landmark_list`/
`roadsite_list`) über `dataclasses.asdict()` verlustfrei - sie erfinden
keine eigene Siedlungs-Feldliste, sondern reichen §6 unverändert durch.

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

## Kein stiller Ersatzwert

`welt_laden()` prüft jedes im Manifest gelistete Feld gegen die
tatsächlichen Dateien. Fehlt eines - Datei fehlt ganz, oder ein einzelnes
Feld darin - bricht `WeltLadenFehler` (definiert in `overview_tab.py`,
direkt neben `OverviewTab`) mit dem genauen Feldnamen ab. Es gibt keinen
Pfad, der ein fehlendes Feld durch `None`, `0` oder eine leere Struktur
ersetzt. Das folgt derselben Regel, die in CLAUDE.md mehrfach mit
konkreten, teuren Fehlern begründet wird (der Shader-Pfad-Bug, der
adaptive-Mesh-Bug) - ein stiller Rückfall sieht im Log wie Erfolg aus.

## Nicht zurückgeschrieben

`welt_laden()` gibt ALLE geladenen Daten zurück (`{generator: {feld:
wert}}`), schreibt aber nur `terrain` und `geology` in den laufenden
`data_lod_manager` zurück (über `set_terrain_data_lod`/
`set_geology_data_lod`). Grund: nur diese beiden Generatoren haben einen
einfachen Pro-Feld-Setter. Die übrigen fünf (`weather`, `erosion`, `water`,
`biome`, `settlement`) haben nur
`set_<generator>_data_complete_lod(<typisiertes Dataclass-Objekt>, ...)` -
das für alle fünf Generatoren korrekt aus rohen Arrays/Listen
zusammenzubauen (`WeatherData`, `ErosionData`, `WaterData`, `BiomeData`,
`SettlementData` - die genauen Klassennamen stehen in
`managers/data_lod_manager.py`) wäre ein eigenständiger, riskanter Umbau
für fünf verschiedene Verträge gewesen und ist bewusst außerhalb des
Zeitrahmens dieses Nachtbetrieb-Tickets geblieben.

Das ist eine **bewusste Scope-Grenze**, kein übersehener Fall: `welt_laden`
loggt eine WARNING mit den Namen der betroffenen Generatoren, sobald einer
von ihnen geladene, aber nicht zurückgeschriebene Daten hat - nichts wird
verschwiegen.

## Was fehlt (für eine spätere Sitzung)

1. **Visuell nicht bestätigt.** Der Rundgang-Test läuft headless mit einer
   erfundenen `FakeDataLODManager`. Ob `welt_laden()` in der echten,
   laufenden App tatsächlich sichtbar ein Terrain wiederherstellt (Reiter
   neu zeichnet o.ä.), wurde nicht getestet - das bräuchte die laufende
   Qt-App und eine echte generierte Welt.
2. **Terrain3D-Kontrollkarte** existiert noch nicht - nur die Höhenkarte
   ist im terrain3d-gerechten Format vorhanden (siehe "Godot-Bedarf").
3. **Rückschreiben für weather/erosion/water/biome/settlement** fehlt
   (siehe "Nicht zurückgeschrieben") - wer eine geladene Welt vollständig
   in die laufende Pipeline zurückspielen will (nicht nur Terrain/Geologie),
   muss die fünf `*Data`-Dataclasses aus den geladenen rohen Feldern bauen.
4. **`erosion`-Lücke** in `collect_all_available_data()` (siehe oben) -
   unabhängig von #38, aber jeder künftige Aufruf von `welt_backen()`
   erbt sie.
