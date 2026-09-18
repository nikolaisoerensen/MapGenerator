# Welt-Format (Ticket #38, "welt_backen und welt_laden als Naht")

Naht: die eine Stelle, an der eine komplette generierte Welt weggeschrieben
und wieder eingelesen wird. Vorher gab es dafür nur "Export" (sechs
Methoden, verstreut in `gui/tabs/overview_tab.py`) und gar kein "Import" -
der Editor konnte eine Welt erzeugen, aber nicht wieder laden.

* **Naht:** `gui/utils/welt_format.py`, Funktionen `welt_backen(pfad,
  data_lod_manager, parameter_manager)` und `welt_laden(pfad,
  data_lod_manager)`.
* **Bausteine:** `welt_backen()` ruft `export_complete_json()` aus
  `gui/tabs/overview_tab.py` auf - das JSON-Format wird dort geschrieben,
  hier nur wieder eingelesen. Die anderen fünf früheren Export-Methoden
  (PNG-Collection, 3D-OBJ, Material-File, Statistik-Text, Einzel-PNG)
  bleiben eigenständige Export**formate** für andere Zwecke (Bildschirmfoto,
  3D-Druck, o.ä.) und sind nicht Teil der Welt-Datei.
* **Rundreise-Test:** `tests/smoke_test_welt_backen_laden.py`.

## Die Feldliste: was eine Welt ausmacht

Quelle der Wahrheit ist die Konstante `WELT_DATEN_SCHLUESSEL` in
`gui/tabs/overview_tab.py` (nicht diese Tabelle - bei Abweichung gilt der
Code). Sieben Kategorien, für jede die zugehörigen Generator-Daten:

| Kategorie | Felder | Woher |
|---|---|---|
| `terrain` | `heightmap`, `slopemap`, `shadowmap` | `data_lod_manager.get_terrain_data(...)` |
| `geology` | `rock_map`, `hardness_map` | `data_lod_manager.get_geology_data(...)` |
| `erosion` | `erosion_map`, `sedimentation_map`, `thermal_erosion_map`, `thermal_deposition_map`, `sediment_load_map`, `water_depth_map`, `flow_velocity_map` | `data_lod_manager.get_erosion_data(...)` |
| `weather` | `wind_map`, `temp_map`, `precip_map`, `humid_map` | `data_lod_manager.get_weather_data(...)` |
| `water` | `water_map`, `flow_map`, `flow_speed`, `cross_section`, `soil_moist_map`, `rock_map_updated`, `evaporation_map`, `ocean_outflow`, `water_biomes_map` | `data_lod_manager.get_water_data(...)` |
| `biome` | `biome_map`, `biome_map_super`, `super_biome_mask` | `data_lod_manager.get_biome_data(...)` |
| `settlement` | `settlement_list`, `landmark_list`, `roadsite_list`, `plot_map`, `civ_map`, `roads`, `sea_roads`, `plot_edges` | `data_lod_manager.get_settlement_data(...)` |

**Wichtiger Fund bei diesem Ticket:** `erosion_map`/`sedimentation_map`
standen in der alten Handschrift (vor Modul-Ebene-Umbau) unter `water` -
dort liefert `get_water_data()` für diese beiden Schlüssel aber seit dem
Erosion-Generator-Umzug (2026-07-28) immer `None`, weil sie inzwischen unter
`get_erosion_data()` liegen (`managers/data_lod_manager.py`, Zeile ~307,
`DATA_KEY_TO_TAB_MAPPING`). Die Kategorie `erosion` fehlte deshalb bislang
komplett in `collect_all_available_data()` - eine Welt ohne Erosionsdaten
wäre für eine bildpunktgenaue Rundreise unvollständig, weil
`erosion_map`/`sedimentation_map`/`thermal_*` die Geländeform mitbestimmen
(siehe `get_terrain_data_combined()`). Mit diesem Ticket ergänzt.

### Pflichtfelder vs. vollständige Feldliste

`WELT_DATEN_SCHLUESSEL` (oben) ist **alles, was geschrieben wird, wenn
vorhanden**. `REQUIRED_WORLD_DATA` (ebenfalls in `overview_tab.py`) ist eine
kleinere Teilmenge:

| Kategorie | Pflichtfelder |
|---|---|
| `terrain` | `heightmap`, `slopemap`, `shadowmap` |
| `geology` | `rock_map`, `hardness_map` |
| `erosion` | `erosion_map`, `sedimentation_map`, `sediment_load_map` |
| `weather` | `temp_map`, `precip_map` |
| `water` | `water_map`, `soil_moist_map`, `water_biomes_map` |
| `biome` | `biome_map` |
| `settlement` | `settlement_list`, `civ_map` |

**Wichtig: `welt_laden()` prüft NICHT diese Liste direkt gegen die Datei.**
So stand es in einem ersten Entwurf, aber der Kommentar über
`REQUIRED_WORLD_DATA` in `overview_tab.py` hält ausdrücklich das Gegenteil
fest: `REQUIRED_WORLD_DATA` ist die Grundlage der Vollständigkeits-Ampel in
der GUI (`WorldCompletenessWidget`, "ist die Welt fertig generiert?") - eine
feste Mindestliste, gegen die *jede* Datei geprüft würde, hätte genau die
Nebenwirkung, die vermieden werden soll: eine echte, aber nur teilweise
generierte Welt (z.B. eine, für die noch keine Erosion gerechnet wurde) hätte
`erosion.erosion_map` nie besessen und würde beim Laden fälschlich als
"kaputt" gemeldet, obwohl an der Datei selbst nichts beschädigt ist.

Stattdessen deklariert `welt_backen()` beim Schreiben selbst, welche
Kategorien zu diesem Zeitpunkt vollständig waren - über
`metadata.data_completeness.generator_status` (von
`analyze_data_completeness()` berechnet, derselben Funktion, die auch die
GUI-Ampel speist). `welt_laden()` prüft `REQUIRED_WORLD_DATA` nur für
Kategorien, die dort mit `true` deklariert sind: fehlt dort trotzdem eines
der Pflichtfelder, ist das kein normaler Zwischenstand mehr, sondern ein
Widerspruch zur eigenen Deklaration der Datei - ein Zeichen, dass die Datei
zwischen Backen und Laden beschädigt oder verstümmelt wurde. Für Kategorien,
die schon beim Backen als unvollständig deklariert waren, verlangt
`welt_laden()` nichts - das ist die reale Welt zum Zeitpunkt des Backens,
kein Fehler.

## Wie die Naht arbeitet

```python
from gui.utils.welt_format import welt_backen, welt_laden, WeltFormatFehler

# Schreiben
welt_backen("meine_welt.json", data_lod_manager, parameter_manager)

# Lesen (in einem NEUEN DataLODManager, z.B. nach Programmstart)
alle_parameter = welt_laden("meine_welt.json", data_lod_manager)
```

`welt_backen()` sammelt über `collect_all_available_data()` und
`get_all_parameters()` (beide Bausteine aus `overview_tab.py`) alles
Verfügbare ein und übergibt es an `export_complete_json()`. Diese Funktion
schreibt für jedes `np.ndarray` `{"data": ..., "shape": [...], "dtype":
"..."}`, für alles andere (Skalare, Listen, die `_json_default()`-Sonderfälle
wie Location-Dataclasses) die JSON-taugliche Form direkt.

`welt_laden()` liest dieselbe Datei, prüft zuerst **alle** Pflichtfelder der
beim Backen als vollständig deklarierten Kategorien (bevor irgendetwas
geschrieben wird - ein halb wiederhergestellter Zustand wäre schlimmer als
ein Abbruch), und schreibt dann jeden vorhandenen
Schlüssel aus `WELT_DATEN_SCHLUESSEL` über den internen, generischen
Ablage-Pfad `DataLODManager._set_data_lod()` unter einem neuen,
kategorieeigenen LOD-Level (aktuelles LOD + 1) zurück in den Manager. Damit
liefert `data_lod_manager.get_<kategorie>_data(schluessel)` (ohne
LOD-Angabe, "höchstes verfügbares") anschließend exakt den geladenen Wert.
np.ndarray-Felder werden aus `data`/`shape`/`dtype` bit-exakt
rekonstruiert (float32 übersteht json's float64-Zwischenform verlustfrei,
siehe `tests/smoke_test_welt_backen_laden.py`).

### Bekannte Grenze der Rundreise

Für Nicht-Array-Felder wie `settlement_list` (eine Liste von
`Location`-Dataclass-Instanzen aus `core/settlement_generator.py`) macht
`_json_default()` beim Schreiben ein `dict` daraus (`dataclasses.asdict`).
Nach dem Laden ist das wieder ein `dict`, keine `Location`-Instanz mehr -
inhaltlich identisch (jedes Feld stimmt), aber ein anderer Python-Typ. Das
Abnahmekriterium "bildpunktgenau identisch" bezieht sich auf die
Bildpunkt-Ebenen (Heightmap, Biome-Map, Rock-Map, ...), nicht auf den
Python-Typ von Objektlisten - dort ist die Rundreise bit-exakt und wird so
getestet.

## Godot-Bedarf (Terrain3D)

Terrain3D (das Godot-Plugin, mit dem generierte Welten später gespielt
werden sollen) erwartet je Region drei Rasterebenen - das steht bereits in
`docs/OFFENE_PUNKTE.md`, Punkt 13.3, und ist dort als offene
Nutzerentscheidung markiert, nicht als erledigt:

| Terrain3D-Ebene | Woher in unserer Feldliste | Stand |
|---|---|---|
| **Height** (32-Bit-Float) | `terrain.heightmap` | in der Feldliste enthalten; heutiger *Bild*-Export (`gui/utils/map_export.py`) ist 16-Bit-PNG und müsste für Terrain3D auf R32F/EXR umgestellt werden - das ist der bestehende offene Punkt, nicht Teil dieses Tickets |
| **Control** (Texturzuordnung als Bitfeld) | müsste aus `biome.biome_map` (+ ggf. `geology.rock_map`) abgeleitet werden | Quelldaten sind in der Feldliste enthalten; die Ableitungsfunktion (Biom → Textur-Bitfeld) existiert noch nicht |
| **Color** (Albedo + Rauheit in Alpha) | müsste ebenfalls aus `biome.biome_map` abgeleitet werden | dito - Quelldaten vorhanden, Ableitung offen |

**Ausdrücklich benannt, damit es nicht übersehen wird:** `welt_backen()`/
`welt_laden()` decken die *Naht* ab (eine Welt vollständig speichern und
wieder laden), nicht den *Terrain3D-Export* selbst. Die Quelldaten, die ein
künftiger Terrain3D-Export bräuchte (Höhe, Biom-Klassifikation,
Gesteinsart), liegen alle in der Feldliste und überstehen die Rundreise -
die Umrechnung in Terrain3Ds Control-/Color-Format ist weiterhin der offene
Punkt 13.3 in `docs/OFFENE_PUNKTE.md`.

## Fehlerverhalten

`welt_laden()` wirft `WeltFormatFehler` (keine stille Rückgabe von `None`
oder eines leeren dicts) in vier Fällen:

1. Datei existiert nicht.
2. Datei ist kein gültiges JSON, oder es fehlen `world_data`/`metadata` auf
   oberster Ebene (kein Welt-Format).
3. `metadata.data_completeness.generator_status` fehlt - die Datei stammt
   nicht aus `welt_backen()` oder ist beschädigt.
4. Eine Kategorie, die beim Backen selbst als vollständig deklariert wurde,
   hat jetzt trotzdem eines ihrer `REQUIRED_WORLD_DATA`-Pflichtfelder
   verloren - Widerspruch zur eigenen Deklaration der Datei.

Das ist bewusst so und keine Ergänzung "zur Sicherheit": siehe CLAUDE.md,
Abschnitt "Gruene Tests koennen eine tote Funktion verdecken" - ein
`.get(schluessel, ersatzwert)` an dieser Stelle wäre von einer echten,
vollständigen Welt nicht mehr zu unterscheiden gewesen.
