# MapGenerator — Übergabe an Claude Code (Stand 2026-07-08, nach LOD-Lockstep-Umbau)

## Rolle & Arbeitsweise (unverändert)
Arbeitssprache Deutsch. Erst Änderungen in Worten beschreiben, erst nach ausdrücklicher
Aufforderung skripten. Keine Abkürzungen bei einem so großen Programm — alles konsistent
mit dem Gesamtkonzept. Bei kleinen Änderungen nur die betroffene Methode/Klasse zeigen.
Keine Hotfixes — große Änderungen sauber einarbeiten und erklären. Dateien mit gleichem
Aufbau erhalten denselben Stil. Verifikation von GUI-Änderungen: throwaway `.venv`
Smoke-Test-Skripte über Bash (nicht PowerShell-Redirect), da es für diese PyQt5-Desktop-App
keine Browser-Preview gibt. 3D-Rendering-Änderungen (`map_display_3d.py`, Shader) lassen
sich NICHT headless verifizieren (`grabFramebuffer()` unzuverlässig in dieser Umgebung) —
der Nutzer muss visuell in der laufenden App bestätigen.

**Wichtig:** Dieses Dokument ersetzt die vorherige Fassung (Terrain-Look/Geology-Rework-
Session) — jener Abschnitt ist jetzt unter "Frühere Session (Terrain-Look/Geology-Rework)"
zusammengefasst, Details bei Bedarf per `git log`/Diff nachvollziehbar. Die generelle
Pipeline-Architektur (Calculator-Graph, Data-Keys, LOD-System) steht in
`docs/generation_pipeline_dependencies.md` — weiterhin die verlässlichste Quelle für die
Grundarchitektur. Offene Tickets/Backlog-Punkte stehen jetzt in **`docs/tickets.xlsx`**
(Sheet "Tickets", Spalten Nr./Ticket/Bereich/Priorität/Status/Notizen) — das ist ab jetzt
die Quelle der Wahrheit statt `docs/backlog.md` (Kanban-Plugin selbst nicht auslesbar,
`backlog.md` war nur ein Zwischenschritt und ist jetzt 1:1 in Zeile 1-7 der xlsx enthalten).
Bitte beim Abschließen eines Tickets dort den Status auf "Erledigt" setzen statt es zu
löschen.

## Git-Stand
Branch `fix/generation-pipeline-dispatch-and-mapsize`. Letzter Commit `65c690e`. **Nichts
aus dieser Session committed** — der Nutzer hat das nicht angefragt, vor dem nächsten
großen Schritt fragen. Aktuell unstaged:

```
core/biome_generator.py                    | 243 ++++++--
core/geology_generator.py                  | 222 ++++---
core/settlement_generator.py               | 300 +++++++---
core/terrain_generator.py                  | 175 +++---
core/water_generator.py                    | 378 ++++++++----
core/weather_generator.py                  | 194 ++++---
gui/OldManagers/data_lod_manager.py        | 118 ++++
gui/OldManagers/generation_orchestrator.py | 903 ++++++++++++-----------------
gui/map_editor.py                          |  48 ++
9 files changed, 1563 insertions(+), 1018 deletions(-)
```

Plus neue, noch nicht getrackte Dateien: `gui/OldManagers/calculator_graph.py` (zentrale
neue Graph-Definition), `docs/tickets.xlsx`, `docs/backlog.md`, dieses Handover-Dokument.
Außerdem liegen ein paar Debug-Logs aus Smoke-Test-Läufen im Repo-Root
(`smoke_test_*.log`, `test_geology_height_feedback.log`) — reine Nebenprodukte, können
gelöscht werden, sind nicht Teil der eigentlichen Arbeit.

## Pipeline-Gesamtstatus
Alle 6 Generatoren (Terrain → Geology → Weather → Water → Biome → Settlement) laufen
end-to-end durch — jetzt zusätzlich mit **echtem calculator-feingranularem LOD-Lockstep**
statt nur grober Generator-Reihenfolge, und **automatischem Start beim Öffnen des Map
Editors** mit Default-Parametern. Verifiziert per `.venv`-Smoke-Tests über den echten
`GenerationOrchestrator`/Qt-Event-Loop (nicht gemockt).

## Was diese Session behoben/gebaut wurde: LOD-Lockstep-Umbau (Tracker #16) — ABGESCHLOSSEN

Ausgangsproblem (Nutzer-Beobachtung): der alte Orchestrator ließ jeden der 6 Generatoren
unabhängig durch ALLE seine eigenen LOD-Stufen laufen, bevor der nächste Generator dran
war ("kein echtes Gleichschritt-Verhalten, nach einer Umrundung sollten eigentlich alle
LODs einmal erhöht werden"). Das war grob (6-Knoten-Ebene), nicht fein genug — z.B.
Settlement-Phasen, die gar keine Biome-Daten brauchen, mussten trotzdem unnötig auf Biome
warten.

**Architektur (siehe Memory `project-lod-lockstep-calculator-dispatch` für Details):**
- `gui/OldManagers/calculator_graph.py` (neu): `CALCULATOR_GRAPH` — 34 einzeln
  dispatchbare Rechenknoten über alle 6 Generatoren (aus
  `docs/generation_pipeline_dependencies.md` plus `geology.faceted_boundaries`, das dort
  nicht dokumentiert war). `CalculatorDispatcher` — globaler Runden-Scheduler mit echter
  Rundenbarriere: kein Knoten erreicht LOD N+1, bevor nicht jeder für LOD N angefragte
  Knoten fertig ist, aber Knoten aus verschiedenen Generatoren laufen innerhalb derselben
  Runde, sobald ihre Abhängigkeiten es zulassen (kein künstliches Warten auf den ganzen
  Generator). Bewiesen: Settlement-Phasen #28-#33 laufen vollständig OHNE dass Biome
  überhaupt angefragt wurde — nur `settlement.plot_nodes` (#34) braucht `biome_map`.
- `gui/OldManagers/data_lod_manager.py`: neuer feingranularer Calculator-Storage
  (`set_calculator_output`/`get_calculator_output`/`get_calculator_completed_lod`),
  getrennt vom bisherigen Domain-Level-Storage (`_terrain_data`/`_geology_data`/etc.).
  Zusätzlich neue `get_calculator_combined_heightmap(lod_level)` (siehe Bugfix unten).
- Alle 6 `core/*_generator.py`: jeweils in einzeln aufrufbare `_calc_*(calculator_id,
  lod_level)`-Methoden zerlegt (eine pro CALCULATOR_GRAPH-Knoten) + `assemble_*_data
  (lod_level, parameters)` (baut das finale Domain-Objekt, sobald ALLE Knoten eines
  Generators eine Runde fertig haben). Die alten öffentlichen Einstiegspunkte
  (`calculate_heightmap`/`calculate_geology`/etc.) sind jetzt dünne Standalone-Convenience-
  Wrapper für Legacy-Aufrufer/Tests.
- `gui/OldManagers/generation_orchestrator.py`: `CalculatorThread` (ersetzt
  `GenerationThread`) läuft EINEN Rechenknoten pro Thread; `advance_calculator_dispatch()`
  treibt den Dispatcher an; `_maybe_assemble_generator()` ruft `assemble_*_data` auf,
  sobald ein Generator seine Runde komplett hat. Signal-Vertrag für die Tabs unverändert
  (`generation_started`/`generation_completed`/`generation_failed`/
  `lod_progression_completed`/`queue_status_changed`).
- `gui/map_editor.py`: neue `_auto_start_generation()`, per `QTimer.singleShot(0, ...)`
  am Ende von `MapEditorWindow.__init__` ausgelöst — fragt alle 6 Generatoren mit ihren
  echten Default-Parametern (`tab.get_current_parameters()`) direkt beim Orchestrator an
  (NICHT über `tab.generate()`, das würde bei Geology/Weather/Water/Biome/Settlement sofort
  an `check_input_dependencies()` scheitern, da Terrain zum Zeitpunkt t=0 noch nichts
  geliefert hat — die tab-seitige Prüfung ist nur eine UI-Hilfe für manuelles Klicken,
  die eigentliche Korrektheit stellt der Calculator-Dispatcher sicher).

**Wichtigster Bugfix in dieser Session (nach der Verdrahtung gefunden):** mehrere
`_calc_*`-Methoden lasen `heightmap_combined`/gemeinsame Cross-Generator-Inputs über die
Domain-Ebene (`get_terrain_data_combined`, gemeinsame "hole alle Dependencies"-Helfer in
Water/Biome), die aber erst befüllt wird, NACHDEM ein ganzer Generator seine Runde
abgeschlossen hat — der Dispatcher gibt einen Folgeknoten aber schon frei, sobald nur
seine ECHTE Einzelabhängigkeit (z.B. `terrain.redistribution`) fertig ist, was oft viel
früher ist. Außerdem holten die gemeinsamen Helfer in Water/Biome (`_get_prepared_water_
inputs`/`_get_prepared_biome_inputs`) IMMER alle möglichen Inputs, auch wenn der jeweilige
Rechenknoten laut Graph nur einen Bruchteil davon braucht (z.B. `water.lake_detection`
braucht laut Graph nur die Heightmap, wartete aber trotzdem auf Geology/Weather-Outputs).
Beides gefixt: neue `get_calculator_combined_heightmap()` liest direkt aus dem Calculator-
Storage; die gemeinsamen Helfer bekommen jetzt ein explizites `needed`-Argument pro
Aufrufer. Alle Standalone-Convenience-Wrapper (für Legacy-Tests) wurden ebenfalls auf den
korrekten Calculator-Storage umgestellt (seedeten vorher denselben jetzt falschen Domain-
Level-Key).

**Verifiziert** über echte `.venv`-Smoke-Tests (nicht gemockt): alle 6 Generatoren
gleichzeitig über `GenerationOrchestrator.request_generation()` angefragt, echter Qt-Event-
Loop, 2 LOD-Runden, korrekte `generation_started`-Reihenfolge, alle Domain-Daten am Ende
vorhanden und plausibel; separat je Generator die Cross-LOD-Akkumulation (Geology
`height_delta`, Water `erosion_map`/`sedimentation_map`) über echte Dispatcher-Runden
geprüft; zusätzlich ein Test, der eine echte `MapEditorWindow`-Instanz konstruiert und
bestätigt, dass Auto-Start beim Öffnen tatsächlich alle 6 Generatoren ohne Zutun
durchlaufen lässt.

Alle 19 Einzelschritte dieses Umbaus stehen jetzt auch als abgeschlossene Tickets (Nr.
8-26) in `docs/tickets.xlsx`.

## Offene Punkte (unverändert aus der letzten Session, nicht bearbeitet)

1. **"Guided Carving"-Architekturvorschlag vom Nutzer**: einmalige globale Flow-
   Accumulation+Watershed-Maske auf LOD0, die bei jedem LOD-Sprung als "Masterplan" neues
   Terrain-Rauschen dort blockiert, wo ein Fluss verläuft. Architektonisch umkehrend
   (Terrain müsste vor Water von einer Fluss-Maske wissen). **Keine Antwort vom Nutzer
   erhalten, offen** — nicht ohne explizites Go-Ahead anfangen.
2. **`sedimentation_map` bleibt bei 0.0** trotz nicht-trivialer `erosion_map`-Werte.
   Ursache identifiziert (`transport_capacity` wächst mit `velocity**2.5` schneller als
   die Sedimentlast). **Keine Antwort vom Nutzer erhalten, offen.**
3. **3D-View "mehrere Elemente ineinander"** bei map_size=256 — kein Doppel-Draw-Call im
   Code gefunden, Vermutung: extreme Höhen-Überzeichnung. Nicht headless verifizierbar,
   Nutzer muss live bestätigen.
4. **Wind-Speed-Warnung** ("Unrealistic wind speeds detected", bis ±215 m/s) und
   **Biome-Validierungs-Warnungen** ("Temperature-elevation relationship appears
   inverted") — noch nicht untersucht, tauchen auch in den neuen Lockstep-Smoke-Tests
   wieder auf (harmlos für die Tests, aber inhaltlich nicht geklärt).
5. Drei widersprüchliche Dependency-Listen im Code (`value_default.py:VALIDATION_RULES.
   DEPENDENCIES`, `descriptor.py`, `data_lod_manager.py:DATA_DEPENDENCY_MATRIX`) — nicht
   angefasst, `CALCULATOR_GRAPH` in `calculator_graph.py` ist jetzt die einzige verlässliche
   Quelle für Abhängigkeiten, die drei alten Listen sind weiterhin nur für Cache-
   Bookkeeping relevant, nicht für Ausführung.
6. Aus `docs/tickets.xlsx` (Nr. 1-7, vom Nutzer selbst eingetragen, alle noch "Offen"):
   GPU-Support-Prüfung, Flüsse versiegen statt aus der Map zu fließen, zu viele
   Fluss-Biom-Felder, Settlements/Voronoi-Plots überarbeiten, Ladebalken kalibrieren,
   Settlement-Tab-UI aufräumen, Slider-Breite ändert sich beim Klicken auf "Berechnen".

## Frühere Session (Terrain-Look/Geology-Rework, vor dem LOD-Lockstep-Umbau)
Kurzfassung — Details per `git log`/Diff, falls relevant: Terrain-Noise-Oktaven/
Persistence-Defaults korrigiert (Sub-Pixel-Rauschen bei alten Werten), 3D-Mesh-Generierung
vektorisiert, GPU-Shader-Aufrufe in `terrain_generator.py` auf die tatsächlich existierenden
`ShaderManager`-Methodennamen korrigiert, Geology-Tektonik-Effekte wirken jetzt kumulativ
auf die Heightmap (vorher nur auf `rock_map`, plus ein Parameter-Namens-Bug der 2 Slider
komplett wirkungslos machte), facettierte Rock-Map-Grenzen, Höhen→Härte-Tendenz, 9-Stufen-
Härtesystem, Glossy-3D-Shader auf matt reduziert (visuell noch nicht vom Nutzer bestätigt).

## Verifikationsmethode
Alle Backend-Änderungen wurden über throwaway `.venv`-Python-Skripte verifiziert (Python-
Interpreter mit PyQt5 liegt in `.venv/Scripts/python.exe` im Repo-Root, System-Python hat
kein PyQt5) — Generatoren/Manager/Orchestrator bzw. ganze `MapEditorWindow`-Instanzen
direkt konstruiert, `generate()`/`request_generation()` aufgerufen, auf die echten
Orchestrator-Signale gewartet (`QEventLoop`+`QTimer`), Ergebnisse aus `DataLODManager`
gelesen und auf Shape/Range/Konsistenz/Mass-Conservation geprüft. **Diese Skripte lagen im
Session-Scratchpad (`%LOCALAPPDATA%\Temp\claude\...`), nicht im Repo — sie sind für eine
neue Session nicht mehr verfügbar und müssten bei Bedarf neu geschrieben werden**, nach
demselben Muster (siehe Memory `feedback-verify-with-smoke-tests`). 3D-/Shader-Änderungen
lassen sich NICHT so verifizieren (kein Headless-GL-Kontext zuverlässig) — die müssen live
in der App geprüft werden.
