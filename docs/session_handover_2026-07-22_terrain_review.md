# MapGenerator — Übergabe: Terrain-Review-Runde + Heightmap-Semantik (Stand 2026-07-22)

## Zweck dieses Dokuments
Diese Session hat den ersten Durchgang der geplanten Generator-Review-Reihe
(Terrain → Geology → Weather → Water → Biome → Settlement, je 5 Standardfragen)
für **Terrain** abgeschlossen, danach auf Nutzer-Feedback hin die
Heightmap-Anzeige-Semantik generatorübergreifend korrigiert. Der Nutzer geht
mit **Geology** als nächstem Generator in einem neuen Gesprächskontext weiter
— dieses Dokument ist die Grundlage dafür, damit nichts aus dieser Session
verloren geht.

## Wichtig: Wo liegt der Code?
**Alle Änderungen dieser Session liegen direkt im Haupt-Checkout**
(`C:\Lokale Dateien\Projects\Python\MapGenerator`, Branch `main`, HEAD
`c62dd8c`), **nicht** im Worktree, in dem diese Konversation technisch
gestartet wurde. Das ist eine bewusste Abweichung vom üblichen
Worktree-Vorgehen (siehe `CLAUDE.md`) — vermutlich, weil frühere Sessions
direkt gegen den normalen `python main.py`-Checkout des Nutzers gearbeitet
haben, um Live-Tests ohne Merge-Schritt zu ermöglichen. **Nichts ist
committed** — alles steht als unstaged Änderung im Haupt-Checkout:

```
core/biome_generator.py             |  24 +++--
core/terrain_generator.py           | 177 ++++++++++++++++++------------------
core/water_generator.py             |   6 +-
core/weather_generator.py           |  12 ++-
managers/data_lod_manager.py |  83 +++++++++++++++++
managers/shader_manager.py   |  26 +++++-
gui/config/value_default.py         |  32 ++++++-
gui/tabs/base_tab.py                |  19 ++--
gui/tabs/geology_tab.py             |   2 +-
gui/tabs/overview_tab.py            |  12 +++
gui/tabs/terrain_tab.py             |  77 ++++++++++++----
gui/tabs/water_tab.py               |   2 +-
gui/tabs/weather_tab.py             |   2 +-
gui/widgets/map_display_2d.py       |   8 +-
gui/widgets/map_display_3d.py       |  39 +++++++-
15 files changed, 379 insertions(+), 142 deletions(-)
```

Für die nächste Session (neuer Kontext für Geology): entweder im selben
Haupt-Checkout weiterarbeiten (konsistent mit dieser Historie), oder vorher
mit dem Nutzer klären, ob zwischendurch committed werden soll. Bitte vor
jeder größeren neuen Änderung fragen, nicht automatisch committen.

## Teil 1: Terrain-Review-Runde (abgeschlossen)

Plan-Datei: `C:\Users\soere\.claude\plans\46-colormap-passt-jetzt-replicated-kay.md`
(vollständige Herleitung aller Punkte, inkl. Codezeilen-Referenzen zum
damaligen Stand).

Umgesetzt und per Headless-Smoke-Test verifiziert:
- **4a** Oktaven-Nyquist-Aliasing behoben (`_max_safe_octaves()` in
  `core/terrain_generator.py`, wirkt einheitlich auf GPU/CPU/Simple-Fallback).
- **4b** Shadow-Raycast-Konstanten zwischen GPU und CPU vereinheitlicht
  (`step_size=0.5`, `max_distance=128`, `height_scale=1.0` jetzt an beiden
  Stellen gleich) — **noch nicht live/visuell verglichen**, nur headless
  auf Signatur-Ebene verifiziert.
- **4c** Parameter-Grenzen-Validierung liest jetzt direkt aus
  `value_default.py TERRAIN.<PARAM>` statt dupliziert zu sein; tote,
  widersprüchliche `validate_terrain_parameters()`-Funktion entfernt.
- **4d** Tote `spacing`/`smoothing`-Slope-Parameter entfernt (bewusst NICHT
  als neue Slider ergänzt — Nutzer-Vorgabe: Glättung läuft über
  Erosion/Geology, nicht über einen eigenen Smoothing-Regler).
- **4e/Phase B** Terrain-Tab zeigt jetzt "Terrain Heightmap" (Rohform) UND
  "Heightmap Combined" (Endergebnis) als zwei separate Radio-Optionen —
  Details zur finalen Farbskalen-Korrektur siehe Teil 2 unten.
- **4f** `WORLD_SIZE_KM` ist jetzt ein echter Slider "Map Distance (km)"
  unter "Map Size" im Terrain-Tab, live propagiert an `DataLODManager`
  (`get/set_map_distance_km()`) und von dort an alle 5 bisherigen
  Konsumenten (`terrain_generator.py`, `biome_generator.py`,
  `water_generator.py`, `weather_generator.py`, `map_display_3d.py`) —
  ersetzt den vorherigen fest verdrahteten Import der Konstante.
- **5.1** Oktaven-Slider-Tooltip erklärt jetzt das Aliasing-Risiko.
- **5.3** Terrain-Slider in zwei Gruppen "Shape" und "Noise Detail" sortiert.

**Noch nicht live bestätigt** (laut Projekt-Konvention nicht headless
verifizierbar, braucht laufende App): 4b (Schatten-Vergleich GPU/CPU
visuell), 4f (sichtbarer Effekt bei Map-Distance ≠ 10 km auf
Slope/Shadow/Biome/Water/Weather), 5.3 (Layout-Check der neuen
Slider-Gruppen).

## Teil 2: Heightmap-Anzeige-Semantik korrigiert (diese Session, nach Live-Test)

### Ausgangsproblem
Nach dem Live-Test von 4e meldete der Nutzer: "Terrain Heightmap" und
"Heightmap Combined" sehen im Terrain-Tab fast identisch aus, und fragte,
ob evtl. bei einem LOD-Übergang (z.B. LOD4→LOD5) versehentlich die
kombinierte Heightmap als neue Terrain-Basis zurückgeschrieben wird.

### Untersuchungsergebnis: **kein Datenfehler**
Direkt im Code verifiziert (`core/terrain_generator.py`,
`_execute_generation()` → `calculate_heightmap()` → `_save_to_data_manager()`
Zeile ~1497-1512): jede LOD-Stufe berechnet Noise+Redistribution komplett
neu aus den aktuellen Parametern; `set_terrain_data_lod("heightmap", ...)`
bekommt ausschließlich `result.heightmap` aus dieser frischen Berechnung,
niemals einen Wert, der aus `get_terrain_data_combined()` stammt. Es gibt
kein Upsample-and-Refine irgendwo in der echten Pipeline. Geologys
`height_delta` und Waters `erosion_map`/`sedimentation_map` sind und bleiben
eigene, separate Arrays, die Terrains `heightmap`-Key nie überschreiben.

Die eigentliche Ursache war ein **Rendering-Bug**, den die 4e-Änderung
selbst eingeführt hatte: der neue Radio-Modus "Terrain Heightmap" nutzte
einen anderen `data_type`-String als "Heightmap Combined", und nur einer
der beiden traf im 2D-Dispatch (`map_display_2d.py`) auf den Zweig mit
fester Farbskala (`elevation_vmin=0`/`elevation_vmax=4000`) — der andere
fiel auf Auto-Skalierung zurück, die zufällig einen ähnlichen Wertebereich
zeigte, wodurch beide Ansichten visuell fast gleich aussahen, obwohl die
zugrunde liegenden Daten (leicht) unterschiedlich waren. Zusätzlich sind
die Geology-/Water-Deltas selbst bei Default-Parametern moderat (Tektonik
gedeckelt auf 15%/3%/3% der Höhenspanne je Durchgang, Sedimentation absichtlich
nahe am Minimum kalibriert) — das trägt zur Ähnlichkeit bei, war aber nicht
die Hauptursache.

### Fix: einheitliche Farbskala
`gui/widgets/map_display_2d.py`: `update_display()` routet jetzt sowohl
`"heightmap"` als auch `"heightmap_combined"` durch denselben
`_render_heightmap()`-Pfad (feste Skala) — beide Ansichten sind jetzt direkt
farblich vergleichbar, nur die Werte selbst unterscheiden sich noch.

### Grundsätzliche Umbenennung der `data_type`/`layer_type`-Strings
Um das eindeutig und konsistent zu halten, wurden die internen Strings
umbenannt (betrifft NUR die Anzeige-Ebene, keine Domain-Daten-Keys):
- **`"heightmap"`** = Terrains eigene, unveränderte Rohform
  (`get_terrain_data("heightmap")`).
- **`"heightmap_combined"`** = Endergebnis nach Geology+Water
  (`get_terrain_data_combined("heightmap")`), vorher unter dem
  irreführenden Namen `"heightmap"` geführt.

Angepasst in: `terrain_tab.py`, `geology_tab.py`, `water_tab.py`,
`weather_tab.py`, `base_tab.py` (Default-`update_display_mode()` UND der
3D-Mesh-Fast-Path in `_push_data_to_current_display()` — nutzt jetzt
`layer_type == "heightmap_combined"` als Signal "Daten sind schon
kombiniert, kein Re-Fetch nötig", vorher fälschlich an `"heightmap"`
geknüpft). **Nebeneffekt, der einen zweiten, unabhängigen Bug behoben hat:**
`settlement_tab.py`s 3D-Ansicht pushte Rohterrain (`get_terrain_data`) unter
dem alten `"heightmap"`-Namen, was im 3D-Fast-Path fälschlich direkt als
Mesh-Grundlage verwendet wurde (Rohform statt Endergebnis) — durch die
Umbenennung nimmt der Fast-Path jetzt korrekt den `else`-Zweig und holt sich
immer die echte kombinierte Heightmap fürs Mesh, unabhängig vom
`layer_type` des jeweiligen 2D-Pushes. `settlement_tab.py` selbst wurde
nicht angefasst — dieser Fix ergab sich rein aus der Umbenennung.

### Semantik-Korrektur: Geology_Heightmap / Water_Heightmap sind jetzt Delta-only
Der ursprüngliche Plan (Querschnittsprinzip B) sah vor, dass "Geology
Heightmap" = Terrain + Geology-Delta ist (eine absolute Heightmap). Der
Nutzer hat das nach Rücksprache **explizit korrigiert**: die einzelnen
Stufen sollen strikt getrennt bleiben, nicht kumulativ. Zitat: *"Heightmap
sollte immer nur Terrain-Input haben, Geology_Heightmap ist was durch
Geology auf Heightmap draufkommt oder abgezogen wird an jedem XY punkt.
Das gleiche für Water_Heightmap [...] Damit haben wir alle informationen
immernoch getrennt und erhalten."*

**Finale Definition (gilt für die Geology-/Water-Runden):**
- `Heightmap` (Terrain) = reine Terrain-Rohform, wird nie von etwas anderem
  verändert. Unverändert, bereits korrekt (`get_terrain_data("heightmap")`).
- `Geology_Heightmap` = **NUR** das Tektonik-Delta selbst an jedem XY-Punkt
  (kein Terrain-Anteil). Neue Methode:
  `DataLODManager.get_geology_height_delta(lod_level=None)` — dünner
  Wrapper um den bereits vorhandenen `geology.height_delta`-Key.
- `Water_Heightmap` = **NUR** das Netto-Water-Delta (Sedimentation minus
  Erosion) an jedem XY-Punkt. Neue Methode:
  `DataLODManager.get_water_height_delta(lod_level=None)` — kombiniert
  `sedimentation_map - erosion_map`, shape-guarded, tolerant wenn nur eine
  der beiden Karten schon vorliegt.
- `Heightmap_Combined` = **unverändert**, weiterhin
  `get_terrain_data_combined("heightmap")` = höchste-LOD-Terrain-Heightmap
  plus/minus allen bisher berechneten Geology-/Water-Anteilen.

Die frühere, jetzt falsche `get_geology_only_heightmap()`-Methode (gab
Terrain+Delta zurück) wurde ersetzt, da sie noch von keiner UI konsumiert
wurde — kein Breaking Change für bestehende Tabs.

### Was für Geology/Water noch NICHT gemacht wurde (bewusst, für nächste Runde)
- Kein neuer Radio-Button/Diff-Ansicht in `geology_tab.py` oder
  `water_tab.py` — nur die `DataLODManager`-Infrastruktur
  (`get_geology_height_delta()`/`get_water_height_delta()`) steht bereit.
- Keine Diverging-Colormap-Registrierung (`RdBu_r`-Muster wie bei
  `temp_map`, siehe `gui/config/gui_default.py CanvasSettings.CANVAS_2D
  ["layer_ranges"]`) für die neuen Delta-Layer — bewusst nicht vorab
  geraten, da konkrete vmin/vmax gegen echte Delta-Größenordnungen live
  kalibriert werden müssen (gleiches Vorgehen wie bei jeder bisherigen
  Farbskalen-Kalibrierung in diesem Projekt).
- Die eigentliche Geology-Review-Runde (5 Standardfragen für Geology,
  analog zum Terrain-Durchgang) hat noch nicht begonnen.

## Verifikation dieser Session
Zwei Headless-Smoke-Test-Skripte über die geteilte `.venv`
(`tests/smoke_test_terrain_review_full.py`, erweitert um Teil-2-Checks, und neu
`smoke_test_phaseB_heightmap_semantics.py`) — beide vollständig grün:
- Delta-only-Semantik von `get_geology_height_delta()`/
  `get_water_height_delta()` (inkl. Teilverfügbarkeit: nur Erosion
  vorhanden, Sedimentation fehlt noch).
- `get_terrain_data_combined()`-Formel unverändert.
- `"heightmap"` UND `"heightmap_combined"` routen beide durch
  `_render_heightmap()` (gleiche feste Farbskala).
- 3D-Fast-Path in `base_tab.py` nutzt `"heightmap_combined"` als
  Skip-Refetch-Signal.
- Voller Import-Sweep aller betroffenen Module.

**Nicht headless verifizierbar, braucht Live-App-Check:**
- Visueller Vergleich "Terrain Heightmap" vs. "Heightmap Combined" im
  Terrain-Tab — sollten jetzt farblich auf derselben Skala, aber mit
  sichtbar unterschiedlichen Werten erscheinen (Unterschied hängt von
  Geology-/Water-Parametern ab, ist bei Default-Werten evtl. weiterhin
  subtil, siehe oben).

## Empfehlung für den Einstieg in die Geology-Runde
Gleiches Vorgehen wie bei Terrain: 5 Standardfragen (Rechenschritte,
Slider vs. fixiert, Map-Size-Skalierung, Bewertung, Slider-Verbesserungen)
für Geology beantworten, dabei die oben festgelegte Delta-only-Semantik für
"Geology Heightmap" direkt mit einplanen (Radio-Button + Diff-Rendering +
Farbskalen-Kalibrierung als Teil der Umsetzung, nicht mehr als offene
Frage). Querschnittsprinzip A (CPU/GPU/Fallback-Vergleichbarkeit) weiterhin
bei jedem mehrstufigen Fix anwenden.
