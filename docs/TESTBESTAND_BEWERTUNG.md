# Testbestand-Bewertung (Ticket #45)

Stand: 2026-09-18, Nachtbetrieb-Branch `nacht/2026-09-18`.

Auftrag aus GitHub-Issue #45 ("Testbestand sichten und bewerten"): jede Datei
unter `tests/` lesen, tatsächlich ausführen, und pro Datei beantworten: Was
wird wirklich geprüft? Läuft der geprüfte Codepfad in der echten Pipeline,
oder ist er tot/veraltet? Ist die verwendete Kartengröße echt oder
ausgedacht? Gibt es Duplikate? Urteil: behalten oder überarbeiten (verwerfen
kam in keinem der drei Durchläufe vor).

**Korrektur der Ausgangszahl:** Issue #45 nennt 70 Testdateien. Tatsächlich
sind es **78** (`ls tests/smoke_test_*.py` gezählt, Stand dieser Nacht) — die
Bewertung unten deckt alle 78 ab, aufgeteilt in drei parallel gelaufene
Batches (A: 27, B: 27, C: 24) mit alphabetisch fortlaufender Dateiliste.

## Ergebnis in Kürze

- **78 von 78 Dateien gelesen UND tatsächlich ausgeführt** (nicht nur
  überflogen) über die geteilte venv, in drei read-only Worktrees.
- **68 laufen grün**, **10 zeigen einen Fehlschlag** — davon **4 bereits
  bekannte, befristete oder erwartete Befunde** (siehe unten) und **6 NEUE,
  bisher nicht dokumentierte Befunde**, die als eigene Tickets nachverfolgt
  werden sollten.
- **1 Fehlschlag ist ein reines Umgebungsartefakt** der Worktree-Isolation
  (hardcodierter `sys.path` auf den Hauptcheckout), kein Testfehler.
- **Kein einziger Verwerfen-Kandidat** in 78 Dateien: jede geprüft ein
  reales, in der Pipeline tatsächlich benutztes Verhalten von außen, keine
  wurde als reine Attrappe oder komplett überflüssiges Duplikat eingestuft.
- **9 Dateien tragen einen hardcodierten `sys.path` auf den Hauptcheckout**
  statt auf das eigene Worktree-Verzeichnis — bei künftigen Änderungen an
  den betroffenen Modulen innerhalb eines Worktrees relevant (siehe
  Abschnitt "Umgebungsbefund" unten).

## NEUE Befunde (nicht Teil der bisher bekannten Altlasten — zur Ticketerstellung)

1. **`tests/smoke_test_erosion_quality.py` — 3 von 3 Prüfgruppen rot.**
   Betrifft `core/erosion_generator.py::HydraulicFieldSimulator`, den
   produktiv aktiven Erosionspfad (`EROSION_AKTIV=True`). Kanalnetz-
   Zusammenhang zu gering (45px statt >60px-Schwelle), Sedimentations-
   Flächenanteil außerhalb des Zielbands (7.2% statt 15–55%), Schwellen-
   wirkung im oberen Geländedrittel zeigt die FALSCHE Richtung (24.5%→33.0%
   statt sinkend), Schrittzahl-Skalierung zwischen 96px/144px überschreitet
   den erlaubten Faktor (4.25 statt <3.0). Klären, ob sich das Modell seit
   dem Einfrieren der Schwellen verändert hat, oder ob ein echter
   Qualitätsverlust vorliegt.

2. **`tests/smoke_test_erosion_realismus.py` — Slope-Area-Gesetz rot.**
   Gemessen beta=-0.262 gegen Zielkorridor -0.4..-0.7 aus
   `docs/spezifikation/10_REGIONEN.md` B.2, deutlich abweichend von der dort selbst
   dokumentierten Referenzmessung -0.638. Monotonie- und Konkavitäts-
   Prüfung (dieselbe Datei) bleiben grün. Gleicher produktiver Erosionspfad
   wie oben — beide Befunde könnten dieselbe Ursache haben.

3. **`tests/smoke_test_pipeline_outputs.py` — GPU/CPU-Paritätsverletzung.**
   Zusätzlich zu den 7 bekannten "NUR NULL"-Outputs (siehe unten, bekannt):
   `settlement.landmarks/landmark_list` und
   `settlement.landmark_roads/landmark_roads` liefern auf dem CPU-Pfad
   Daten, auf dem GPU-Pfad NUR NULL — ein Verstoß gegen die in
   `docs/spezifikation/02_INVARIANTEN.md` Abschnitt 1 geforderte CPU/GPU-Parität.

4. **`tests/smoke_test_regionen_welt.py` — 4 Befunde, nur die Laufzeit war
   bekannt, der Inhalt nicht:**
   - Macchia verfehlt Hang- und Wasserziel (Hang 18.6 statt 14.5, Wasser
     37.3 statt 40).
   - Thalassia verfehlt Hang- und Wasserziel (Hang 14.5 statt 11.5, Wasser
     48.9 statt 65).
   - Regionsgrenzen sind deutlich steiler als die Regionen selbst (1.624
     gegen 0.891 im Regioninneren; erlaubt wären maximal +25%) — ein
     Naht-Problem im Sinne von CLAUDE.md.
   - 512px und 1024px derselben Region korrelieren nur mit r=0.94 statt
     annähernd perfekt — verletzt die in
     `docs/spezifikation/90_MESSPROTOKOLLE.md` §10 belegte
     Auflösungsunabhängigkeit.

5. **`tests/smoke_test_weather_temperature_direktnormierung.py` — 3 von 36
   Region/Seed-Kombinationen außerhalb der 0.7-K-Toleranz:** Seed 20260804:
   Skerrheim -0.74K, Thalassia -0.84K; Seed 4242: Samarcia -0.86K. Der Test
   selbst ist sauber gebaut (echte Kartengröße, vier Seeds, konkrete
   Zieltabelle aus `OFFENE_PUNKTE` 1.10/1.11) — das Kalibrierungsergebnis
   ist das Problem, nicht die Prüfung.

6. **`tests/smoke_test_water_lake_detection_gpu.py` — zu schwache
   Zusicherung verdeckt eine echte GPU/CPU-Abweichung:** bei gleichem Seed
   finden CPU und GPU unterschiedlich viele Seen (Seed 7: 6 vs. 3, Seed 42:
   4 vs. 2). Der Test prüft nur "nicht komplett leer", nicht "ungefähr
   gleiche Fläche/Anzahl" — genau die Art zu schwacher Zusicherung, vor der
   CLAUDE.md warnt. Sollte auf einen Flächen- oder Komponentenvergleich mit
   Toleranz verschärft werden.

## Bereits bekannte Befunde (bestätigt, keine neue Handlung nötig)

- `tests/smoke_test_kuesten_naht_kruemmung.py` — rot wie erwartet, offener
  Nahtkrümmungs-Fehler in `core/vektor_kueste.py`, bereits dokumentiert.
- `tests/smoke_test_wasserbilanz_toleranz.py` — +34.62% Abweichung, Ticket
  #70, befristete Dispensation, deckt sich mit der zuvor gemessenen Zahl.
- `tests/smoke_test_pipeline_outputs.py` — die 7 "NUR NULL"-Outputs
  (geology.intrusions/height_delta, erosion.hydraulic+thermal_erosion_map,
  erosion.hydraulic+thermal_deposition_map, settlement.pathfinding/
  sea_roads, settlement.plot_nodes/plots) stehen bereits in
  `docs/TESTBERICHT.md` Zeile 24.
- `tests/smoke_test_siedlungsnaht_felder.py` — Feld "Anschlusspunkte der
  Wege" fehlt absichtlich, Folgeticket #73, dokumentierter
  Platzhalter-Fehlschlag.
- `tests/smoke_test_vektor_kueste.py` — vollständig grün zum Zeitpunkt
  dieser Prüfung, betrifft aber Code einer parallel laufenden Sitzung
  (`core/vektor_kueste.py`) — als Momentaufnahme zu verstehen, nicht als
  dauerhafte Bestätigung.

## Umgebungsbefund: hardcodierter `sys.path` auf den Hauptcheckout

**9 Dateien** setzen `sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")`
statt sich am eigenen Verzeichnis zu orientieren:
`smoke_test_biome_preseed.py`, `smoke_test_camera_controls.py`,
`smoke_test_erosion_field.py`, `smoke_test_erosion_gpu_contract.py`,
`smoke_test_erosion_gpu_parity.py`, `smoke_test_erosion_hauptschalter.py`,
`smoke_test_erosion_quality.py`, `smoke_test_erosion_realismus.py`,
`smoke_test_kuesten_naht_kruemmung.py`, `smoke_test_shader_paths.py`,
`smoke_test_terrain_erosion_filter.py`, `smoke_test_terrain_river_network.py`.

Das ist an sich beabsichtigt (der Test soll die echte Installation prüfen,
nicht einen zufälligen Arbeitsordner) — hat aber in dieser Nacht konkret
zwei Tests verfälscht: `smoke_test_terrain_erosion_filter.py` und
`smoke_test_terrain_river_network.py` schlagen NUR fehl, weil der
Hauptcheckout eine ältere Fassung von
`gui/config/value_default.py::EROSION_FILTER` hat (ohne `DETAIL`,
`GULLY_WEIGHT`, `CREASE_ROUNDING`) als dieser Worktree — eine
Zeitpunkt-Divergenz zwischen Hauptcheckout und Worktree, kein
Testlogik-Fehler. `smoke_test_shader_paths.py` scheitert aus demselben
Grund nur bei Prüfung 1 von 4. Alle drei sollten nach Merge in den
Hauptcheckout erneut laufen, bevor ein endgültiges Urteil gefällt wird.

## Duplikat-Cluster (kein Verwerfen, aber Konsolidierungs-Kandidaten)

- **`smoke_test_water_drainage_erosion.py` / `_water_edge_sediment.py` /
  `_water_erosion_quality.py`**: alle drei testen `DropletErosionSystem`
  mit Überschneidung bei "abgelagert ≤ abgetragen"/"Erosion bergab".
  Empfehlung: die beiden anderen behalten, `water_drainage_erosion.py` als
  dünnsten der drei zuerst prüfen, falls konsolidiert wird.
- **`smoke_test_region_tab.py` / `_regionsfeld.py` / `_regionsregler_wirken.py`**
  und **`smoke_test_settlement_clustering.py` / `_placement.py` /
  `_region_grid.py`**: jeweils derselbe Funktionsbereich aus drei sich
  ergänzenden Blickwinkeln (GUI-Verhalten / Formel-Konsistenz /
  End-zu-End-Durchgriff bzw. Klumpung / Faktorwirkung / Regionsverteilung)
  — kein Duplikat, aber Kandidat für eine spätere Zusammenlegung.

## Kleinere, risikofreie Aufräumpunkte (kein Urteil "überarbeiten" wert)

- `smoke_test_settlement_region_grid.py`: einen selbst als veraltet
  bezeichneten Vergleichswert nur informativ im Ausgabetext — aufräumen.
- `smoke_test_settlement_valley_routing.py`: Testgröße 100px durch ein
  32er-Vielfaches (z.B. 96 oder 128) ersetzen.
- `smoke_test_water_edge_sediment.py` / `smoke_test_water_thermal_erosion.py`:
  je 2–3 ausgedachte Größen (40/48/50/24px, keine 32er-Vielfachen) ohne
  Aussagekraftverlust durch echte Größen ersetzbar.
- `smoke_test_weather_climatology.py`: 10 von 17 Teilprüfungen nutzen
  wiederholt dieselbe ausgedachte Größe 48px — durch eine im selben File
  bereits genutzte echte Größe (z.B. 64px) ersetzen.
- `smoke_test_slope_compass_color.py`: schnell, deterministisch,
  headless-fähig, prüft reale Funktionslogik, ist aber ausdrücklich nicht
  Teil der automatischen Testsuite — sollte aufgenommen werden.
- `smoke_test_water_pipeline_full.py`: Docstring/Kommentar ist seit dem
  2026-07-28-Umbau veraltet (erosion_map/sedimentation_map liegen nicht
  mehr in `WaterData`) — reine Doku-Korrektur.

## Strukturell schwache Prüfung (grün, aber verdeckt eine mögliche Regression)

- `smoke_test_fluss_overlay.py::verdrahtung_im_3d` prüft nur per
  `hasattr()`/Quelltext-Stringsuche, ohne die Funktion wirklich
  aufzurufen und das Ergebnis zu prüfen — strukturell dieselbe Schwäche
  wie die in CLAUDE.md dokumentierte hasattr-Falle, nur eine Ebene tiefer
  versteckt.

## Dead-Code-Befund

- `gui/widgets/kuesten_mesh.py::baue_kuesten_mesh`/`klippenband` werden im
  ganzen Projekt nur von `tests/smoke_test_kuesten_mesh.py` und dem
  eigenständigen Werkzeug `tools/kuestenlaengsschnitt.py` aufgerufen — die
  produktive 3D-Küstenvernetzung läuft seit `KUESTEN_SCHNITT_AKTIV` über
  `gui/widgets/kuesten_schnitt.py`. `smoke_test_kuesten_mesh.py` testet
  damit ein geparktes Prototyp-Modul in voller Breite. Empfehlung: im
  Testkopf klarstellen, dass dies ein geparkter Prototyp ist, oder den
  Test auf die weiterhin genutzten Hilfsfunktionen (`_entlang_abtasten`,
  `kuestenlinien`) verschlanken.

---

## Vollständige Tabelle — Batch A (27 Dateien: adaptive_mesh_vectorized … kuesten_schnitt)

| Dateiname | Was wird tatsächlich geprüft | Kartengröße | Echter Codepfad? | Ergebnis | Urteil | Begründung |
|---|---|---|---|---|---|---|
| smoke_test_adaptive_mesh_vectorized.py | Vektorisierte vs. schleifenbasierte Fassung von `adaptive_terrain_mesh.py`, Geometrie muss übereinstimmen | echt (128/256/512/1024) | Ja | PASS | behalten | Äquivalenztest für Performance-Umbau auf echten Größen |
| smoke_test_adaptive_terrain_mesh.py | Wasserdichtigkeit, Dreiecksreduktion, Determinismus, Randfälle | echt (128/1024) | Ja | PASS | behalten | Prüft sichtbare Eigenschaften statt Interna |
| smoke_test_anzeige_register_3d.py | Registereintrag jedes 3D-Anzeigemodus vollständig | n/a | Ja | PASS | behalten | Fing beim Bau einen realen Fehler (fehlender Flussreiter) |
| smoke_test_archetyp_verteilung.py | Alle 27 Küsten-Archetypen kommen vor (Saatstationen vs. Küstenlänge) | echt (512, +16 Karten) | Ja | PASS | behalten, vorbildlich | Zwei-Enden-Vergleich nach 2026-08-24-Lehre |
| smoke_test_biome_overlays_3d.py | `BiomeTab.apply_overlays()` erreicht 2D UND 3D | n/a | Ja | PASS | behalten | Fing 2026-09-16 einen realen Fehler |
| smoke_test_biome_preseed.py | CALCULATOR_GRAPH: kein Zyklus Feuchte/Biom | n/a | Ja | PASS | behalten | Kritische Graph-Invariante |
| smoke_test_biome_supersampling.py | Teilpixel-Zufall unabhängig von x+y | echt (128) | Ja | PASS | behalten, vorbildlich | Trifft genau die 2026-08-10-Bug-Eigenschaft |
| smoke_test_camera_controls.py | 3D-Kamera-Vektorrechnung gegen echte View-Matrix | n/a | Ja | PASS | behalten | Prüft Ergebnis, nicht wiederholte Codeannahme |
| smoke_test_display_2d.py | Alle 30 2D-Anzeigefunktionen zeichnen wirklich etwas | ausgedacht (48), sachlich begründet | Ja | PASS | behalten | Schließt Daten-Bild-Lücke |
| smoke_test_display_methoden_existieren.py | Jede `hasattr()`-Methode existiert auf min. einer Anzeigeklasse | n/a | Ja | PASS | behalten, foundational | Bewacht häufigste Fehlerklasse im Projekt |
| smoke_test_erosion_field.py | Massenbilanz, Buchhaltung, Determinismus des Erosionskerns | echt (64, MAX_CPU_RESOLUTION) | Ja | PASS | behalten | Technische Garantien, nicht Kalibrierwerte |
| smoke_test_erosion_gpu_contract.py | Statischer Shader/Dispatcher-Vertragsabgleich | n/a | Ja | PASS | behalten | Fängt Fehler vor teurem GPU-Lauf |
| smoke_test_erosion_gpu_parity.py | Echter GPU-Erosionslauf vs. CPU-Referenz | echt (64) | Ja | PASS | behalten, wertvoll | Fing historisch 3 reale GPU-Fehler |
| smoke_test_erosion_hauptschalter.py | EROSION_AKTIV=False → alles null; True-Gegenprobe | echt (128, LOD3) | Ja | PASS | behalten, vorbildlich | Gegenprobe verhindert Schein-Passieren |
| smoke_test_erosion_quality.py | 5 Bildeigenschaften des Erosionsmodells | echt (128, 96, 144 ungültig) | Ja | **FAIL 3/3 Gruppen** | **überarbeiten (NEU)** | Siehe "Neue Befunde" oben #1 |
| smoke_test_erosion_realismus.py | Slope-Area-Gesetz, Monotonie, Konkavität | echt | Ja | **FAIL 1/3 Gruppen** | **überarbeiten (NEU)** | Siehe "Neue Befunde" oben #2 |
| smoke_test_erosionsfilter_baender.py | Bandweise = einteilige Auswertung bitgleich | echt (256/512/1024) | Ja | PASS | behalten | Bitgleichheit schützt Regionseichung |
| smoke_test_export_2048.py | Kategorische Layer nicht interpoliert, Dämpfungsmaske wirkt | echt (256) | Ja | PASS | behalten | Verhindert erfundene Zwischen-Biome |
| smoke_test_fluss_overlay.py | Rasterfunktion + 3D-Verdrahtung | echt (64) | Teilweise | PASS, aber schwach | **überarbeiten** | `verdrahtung_im_3d` prüft nur hasattr/Stringsuche |
| smoke_test_fluss_vorschau.py | Live-Vorschau über gewöhnlichen Anzeigeweg | echt (128) | Ja | PASS | behalten | Verhindert Sonderpfad-3D-Ausfall |
| smoke_test_flussstufen.py | Alle 4 Wasserstufen kommen vor, auch in beiden Biom-Karten | echt (Vollpipeline) | Ja | PASS | behalten | Hätte den 2026-08-24-Bug gefangen |
| smoke_test_geology_3dstack.py | height_delta=0, LOD-Invarianz, Härteverteilung, Seed-Fortpflanzung | echt (9 Kombinationen) | Ja | PASS | behalten | Kein Höhenversatz zwischen Anzeigemodi |
| smoke_test_geology_speed.py | Störungsfeld bitgleich nach Performance-Umbau | echt (256/512/1024) | Ja | PASS | behalten, vorbildlich | Sichert Ergebnis, nicht nur Tempo |
| smoke_test_kontinentform.py | Formregler: Fläche konstant, zusammenhängend, Enden wirken | echt (256) | Ja | PASS | behalten | Fing realen Kalibrierfehler beim Bau |
| smoke_test_kuesten_mesh.py | Küstenvertices/-kanten des Quadtree-Meshs | echt (256/384/512) | **Nein (Prototyp)** | PASS | **überarbeiten** | Testet geparktes, nicht produktives Modul |
| smoke_test_kuesten_naht_kruemmung.py | Höhenmischungs-Realismus an Landzungen | echt (384) | Ja | **FAIL, erwartet** | behalten | Bekannter offener Fehler, vorbildliches Testdesign |
| smoke_test_kuesten_schnitt.py | Konturschnitt-Mesh: Dichtheit, keine Kante über Wasserlinie | echt (256/384/512) | Ja | PASS | behalten | Regressionswächter für 2026-08-22-Wicklungsbug |

## Vollständige Tabelle — Batch B (27 Dateien: kuestengebiete … weltfluesse — Fortsetzung, s. Batch C für den Rest)

| Dateiname | Was wird tatsächlich geprüft | Kartengröße | Echter Codepfad? | Ergebnis | Urteil | Begründung |
|---|---|---|---|---|---|---|
| smoke_test_kuestengebiete.py | Küstenumformung: Wasserlinie/Regionsmittel stehen, Gebietsfläche im Fenster | echt (384) | Ja | PASS | behalten | Deckt 3 dokumentierte historische Bugs ab |
| smoke_test_kuestenprofiltreue.py | Höhenprofile der 27 Archetypen gegen Vorbild-Tabellen | echt (384) | Ja | PASS | behalten | Ergebnis gegen reale Referenzdaten |
| smoke_test_landmarks.py | Eignungskarten, Dominanz-Deckel, Platzierung, Determinismus | synthetisch, angemessen | Ja | PASS | behalten | Blackbox-Prüfung Eignungsfeld+Platzierung |
| smoke_test_layer_2d_3d_parity.py | ~30 Layer: Range-Mapping, Colormap, Skala 2D=3D | synthetisch (24×24) | Ja | PASS (~140 Prüfungen) | behalten (hohe Priorität) | Direkter Wächter der STEHENDEN 2D/3D-REGEL |
| smoke_test_nachtbetrieb.py | Sperrliste, Selbstschutz, Ticket-Commit-Regeln, Morgenbericht-Format | n/a | Ja (Sperrliste echt, Git in Temp-Repos) | PASS (17/17) | behalten | Einziger Test für aktiv genutztes Werkzeug |
| smoke_test_noise_offset_gpu.py | u_offset_x/y GPU==CPU bei mehreren Versätzen | n/a | Ja (echter GPU-Lauf) | PASS (11.2x schneller) | behalten | Belegt GPU/CPU-Parität real |
| smoke_test_parameter_eindeutig.py | Kein Parameterschlüssel zeigt unangemeldet auf zwei Reiter | n/a | Ja | PASS | behalten | Konsistenzwächter gegen Doppelbelegung |
| smoke_test_pipeline_outputs.py | 78 (Knoten,Output)-Paare liefern Daten, GPU+CPU | echt (128, LOD3) | Ja (ist die reale Pipeline) | **FAIL: 7 NUR-NULL + 2 Paritätsverletzungen** | behalten, aktuell rot | 7 bekannt (TESTBERICHT.md), 2 NEU (siehe oben #3) |
| smoke_test_poisson_punkte.py | Determinismus, Mindestabstand, Performance-Regression | n/a | Ja | PASS | behalten | Solider Determinismus-/Regressionstest |
| smoke_test_push_overlays.py | Overlay-Dispatch 2D+3D, unbekannter Name wirft Fehler | n/a | Ja | PASS | behalten | Zweiter Wächter der STEHENDEN 2D/3D-REGEL |
| smoke_test_region_tab.py | Dropdown, Vorschau <0.8s, Regler wirken, Zurücksetzen, 3D-Ansicht | echt (256×160 Vorschau) | Ja | PASS (12/12) | behalten | End-zu-End-GUI-Vertragstest |
| smoke_test_region_vorbild_aehnlichkeit.py | Regionen gegen reale DEM-Vorbilder (Formfaktor, KS-Test, Fraktaldimension) | echte COP30-DEM-Ausschnitte | Ja | PASS (3/3) | behalten | Einzige Kalibrierung gegen reale Referenzformen |
| smoke_test_regionen_fairness.py | Bewohnbare-Fläche-Index je Region gegen Ziel (±25%) | echt (512) | Ja | PASS | behalten | Eigener Prüfwinkel: Spielbalance |
| smoke_test_regionen_welt.py | Hang/Wasser-Fingerabdruck, Nahtprüfung, Auflösungsunabhängigkeit, GPU/CPU | echt (384/512/1024) | Ja | **FAIL (4 Befunde)** | behalten, aktuell rot | NEU, siehe oben #4 |
| smoke_test_regionsfeld.py | Vorschau-Formel-Konsistenz, Land-See-Aufbau je Region | echt | Ja | PASS (6/6) | behalten | Mathematische Konsistenzprüfung |
| smoke_test_regionsregler_wirken.py | Voller Durchgriff Regionsregler → fertige Karte | echt | Ja | PASS (8/8) | behalten | Einziger voller Durchgriffstest |
| smoke_test_reiter_vertrag.py | Shell-Vertrag: 12 Reiterklassen, Viewport/Parameter/Statistik | n/a | Ja | PASS (7/7) | behalten | Architektur-Vertragswächter |
| smoke_test_river_reaches_sea.py | Jeder Flussknoten erreicht das Meer, 0 Sackgassen/Ringe | echt (256/384) | Ja | PASS | behalten | Robuste Blackbox-Invariante |
| smoke_test_seegliederung.py | Seegrad, Zieltiefe, Uferregion, Seeeis-Quote | echt | Ja | PASS | behalten | Konsumiert von biome_generator für Seeeis |
| smoke_test_settlement_clustering.py | Nächste-Nachbar-Abstand je Kultur, keine Klumpung | echt | Ja | PASS | behalten | Eigener statistischer Aspekt |
| smoke_test_settlement_placement.py | Eignungsfaktoren trennscharf, Rang folgt Rauschen | echt (96) | Ja | PASS | behalten | Faktoren-Trennschärfe + Rauschen-vor-Rang |
| smoke_test_settlement_region_grid.py | Roadsite-/Landmark-Verteilung über 9 Regionen | echt | Ja | PASS | behalten (kleine Überarbeitung) | Veralteten Vergleichswert im Text aufräumen |
| smoke_test_settlement_roads.py | Kostenfeld exakt, Gabriel-Graph, Kulturzusammenhalt | synthetisch, angemessen | Ja | PASS | behalten | Präzise Regressionsprüfung exakter Werte |
| smoke_test_settlement_sites.py | Seeweg-Pathfinding, Katalog-Kultur-Konsistenz | synthetisch, angemessen | Ja | PASS | behalten | Deckt historischen Höhen-Deckel-Bug ab |
| smoke_test_settlement_valley_routing.py | A*-Route weicht Grat/Pass aus | 100/256/512 (100 ausgedacht) | Ja | PASS | behalten (kleine Überarbeitung) | 100px durch 32er-Vielfaches ersetzen |
| smoke_test_shader_paths.py | SHADERS_ROOT, 31 Shader existieren, Fremdverzeichnis-Ladbarkeit | n/a | Ja (Prüfungen 2-4) | FAIL Prüfung 1 (Umgebungsartefakt) | behalten | Nur Worktree/Hauptcheckout-Drift, siehe Umgebungsbefund |
| smoke_test_slope_compass_color.py | Hangrichtung → Farbton korrekt | synthetisch (20×10) | Ja | PASS (5/5) | behalten (Überarbeitung: in Suite aufnehmen) | Nicht Teil der automatischen Testsammlung, sollte es sein |

## Vollständige Tabelle — Batch C (24 Dateien: spielkarten … weltfluesse_vektor)

| Dateiname | Was wird tatsächlich geprüft | Kartengröße | Echter Codepfad? | Ergebnis | Urteil | Begründung |
|---|---|---|---|---|---|---|
| smoke_test_spielkarten.py | Kartenzerlegung: Massenspanne, Seitenverhältnis, Determinismus | echt (256/512) | Ja | PASS | behalten | Prüft gegen wörtlich zitierte Nutzervorgaben |
| smoke_test_stadttypen.py | Marktstadt-Wahl, Handelsnetz, Fischersiedlung-Eignung | synthetisch, angemessen | Ja | PASS (9/9) | behalten | Entscheidungsregeln, nicht Maßstab |
| smoke_test_stufen_schalter.py | 3 Abschalt-Häkchen verändern Gelände wirklich | echt (256) | Ja | PASS (5/5) | behalten | Genau die von CLAUDE.md gewarnte Fehlerklasse |
| smoke_test_tab_reihenfolge_kanonisch.py | GENERATOR_TAB_ORDER eine Definitionsstelle, Identität | n/a | Ja | PASS (6/6) | behalten | Billig, trifft wiederkehrendes Muster |
| smoke_test_terrain_erosion_filter.py | ATEF-Filter im alten Pfad (WELTKARTE_AKTIV=False) | echt (128) | **Nein, Umgebungsdrift** | FAIL (AttributeError) | überarbeiten | Hauptcheckout-Drift, siehe Umgebungsbefund |
| smoke_test_terrain_remesh.py | Adaptives 3D-Remesh: Naht dicht, Vertex-Budget, Cache | echt (256/384/512) | Ja | PASS (7/7) | behalten | Genau die 2026-08-12-Lehre umgesetzt |
| smoke_test_terrain_river_network.py | Flussnetz-Skalenentkopplung im alten Pfad | echt (128/256/512) | **Nein, Umgebungsdrift** | FAIL (AttributeError) | überarbeiten | Gleiche Ursache wie erosion_filter |
| smoke_test_terrain_scale_coupling.py | Gully-Größenregler skalenentkoppelt | echt (128/256/512) | Ja | PASS | behalten | Bereits einmal gebrochene Invariante, jetzt grün |
| smoke_test_vektor_kueste.py | Raster- vs. Punktabtaster bitgleich, Determinismus | echt (256/384/512) | Ja | PASS | behalten (Momentaufnahme) | Parallele Sitzung arbeitet aktiv daran |
| smoke_test_wasserbilanz_toleranz.py | Niederschlag = Wasser+Abfluss+Verdunstung ±10% | echt (512, LOD5) | Ja | FAIL +34.62% (bekannt) | behalten | Ticket #70, befristete Dispensation |
| smoke_test_water_drainage_erosion.py | Erosionsformel-Plausibilität + kurzer E2E-Lauf | 64 echt, 48 ausgedacht | Ja | PASS | behalten, schwächster von 3 | Konsolidierungs-Kandidat |
| smoke_test_water_edge_sediment.py | Randmassenverwurf, Härte-Differenzierung, Spawn-Verteilung | 32/64 echt, 3× ausgedacht | Ja | PASS | überarbeiten (klein) | Ausgedachte Größen ersetzen |
| smoke_test_water_erosion_quality.py | Keine Krater, Kanalnetz-Zusammenhang, Sedimentflächen | echt (128, 81920 Partikel) | Ja | PASS | behalten | Umfassendster der 3 Erosionstests |
| smoke_test_water_lake_detection_gpu.py | CPU/GPU-Parität Jump-Flood-Seenerkennung | echt (128) | Ja | PASS, aber zu schwach | überarbeiten | Siehe "Neue Befunde" oben #6 |
| smoke_test_water_pipe_flow.py | Exakte Massenbilanz PipeFlowSimulator, Plateau statt Wachstum | 32 echt, 48 ausgedacht (nebensächlich) | Ja | PASS | behalten | Nicht-verhandelbare physikalische Invariante |
| smoke_test_water_pipeline_full.py | LOD-Übergabe über echten DataLODManager-Pfad | 48 ausgedacht, nebensächlich | Ja | PASS | überarbeiten (nur Doku) | Docstring seit 2026-07-28-Umbau veraltet |
| smoke_test_water_pipeline_order.py | 12 strukturelle Garantien: Reihenfolge, Deadlock-Regression, Feedback | 48 meist, eine Skalenprüfung echt | Ja, gegen echten Scheduler | PASS | behalten, mit Nachdruck | Wertvollste Datei der ganzen Sichtung |
| smoke_test_water_thermal_erosion.py | Böschungswinkel-Mechanik, Massenbilanz, Pro-Zell-Deckel | 32 echt, 2× ausgedacht (nebensächlich) | Ja | PASS | behalten | Eigenständiger Mechanismus, kein Duplikat |
| smoke_test_weather_climatology.py | 17 Klimatologie-Prüfungen an historische Bugreports gebunden | überwiegend 48 ausgedacht | Ja | PASS | behalten, Aufräumhinweis | Wiederkehrende ausgedachte Größe 48 |
| smoke_test_weather_temperature_direktnormierung.py | Temperatur-Jahresmittel auf KLIMA_ZIEL normiert, 9 Regionen | echt (256, LOD6) | Ja | **FAIL 3/36 Kombinationen** | überarbeiten/rot markieren | NEU, siehe oben #5 |
| smoke_test_weather_wind_regions.py | Windmittel auf wind_ziel_map normiert, Nevadin windärmste Region | echt (256, LOD6) | Ja | PASS | behalten | Dokumentiert ehrlich nicht getestete Luv/Lee-Korrelation |
| smoke_test_wege_geometrie.py | Wegband: kein Einsinken, Querprofil, Breite, Klick-Projektion | echt (256/512/1024) | Ja (auch echter Shader) | PASS | behalten | Erfüllt STEHENDE 2D/3D-REGEL |
| smoke_test_wegsuche_schnell.py | Numba-JIT A* punktgenau identisch zur Python-Referenz | synthetisch + echt (256) | Ja | PASS | behalten | Strengste Zusicherung im Batch |
| smoke_test_weltfluesse_vektor.py | Vektorisierte Talschnitzschleife bitgleich zur Schleife | synthetisch + 512 echt | Ja | PASS | behalten | Zitiert CLAUDE.md-Lehre als Testgrund |

---

*Quellen: drei parallele read-only Analyse-Durchläufe dieser Nacht
(Worktrees `agent-a804e3c4be667c89d`, `agent-ac0857383ae470b22`,
`agent-a9bc6ad1e0f3f27ef`), alle Tests tatsächlich per venv-Python
ausgeführt, keine Repo-Datei durch die Analyse selbst verändert.*
