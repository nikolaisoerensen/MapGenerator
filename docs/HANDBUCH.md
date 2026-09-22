# Handbuch MapGenerator

**Was dieses Dokument ist und was nicht.** Der Spezifikationsbaum
`docs/spezifikation/` sagt, was das Programm **soll** (Zielwerte,
Invarianten). Dieses Handbuch sagt, was das
Programm **ist**: wie es aufgebaut ist, wie die Rechenkette läuft, welche
neun Regionen es kennt, was jeder Reiter tut, wie die Karte zum Spiel hin
verlassen wird, und was die Tests prüfen. Baum und Handbuch bleiben
nebeneinander bestehen.

**Belegregel.** Jede harte Behauptung in diesem Handbuch trägt eine Quelle
(Datei, Zeile oder Funktionsname). Eine Behauptung ohne Beleg gilt nicht als
unbewiesen, sondern als falsch — das ist die Lehre aus sieben stillen
Rückfällen, die trotz grüner Tests unbemerkt blieben (siehe `CLAUDE.md`,
Abschnitte „Gruene Tests koennen eine tote Funktion verdecken" und „Wenn alle
Einzelpruefungen gruen sind und das Ergebnis falsch ist").

**Entstehung.** Geschrieben aus `docs/archiv/` und den zum Zeitpunkt der
Erstellung sichtbaren `docs/*.md` heraus (Ticket #52, Nachtbetrieb
2026-09-21). **Wichtiger Vorbehalt:** Ticket #51 („docs/ ins Archiv
schieben") ist auf dem Branch `nacht/2026-09-19` bereits committet, aber
dieser Arbeitsbaum ist von `main` abgezweigt und sieht diesen Branch nicht.
Die Dateiliste unter `docs/`, auf die sich dieses Handbuch stützt, ist also
der alte Stand vor #51, nicht das Ergebnis von #51. **Nach dem Merge von #51
muss ein Mensch gegenlesen, ob sich durch die neue Archiv-Struktur inhaltlich
etwas an den hier beschriebenen Fakten ändert** (reine Verschiebung von
Dateien sollte es nicht, aber das ist ungeprüft).

---

## 1. Aufbau des Programms

Einstiegspunkt `main.py`: `MapGeneratorApp` verwaltet den Fensterwechsel
`MainMenu → Loading → MapEditor` zentral (Docstring `main.py`, Zeilen 1–15).
Die eigentliche Arbeitsoberfläche ist `gui/map_editor_window.py`, Klasse
`MapEditorWindow`.

Verzeichnisse und ihre Rolle:

| Verzeichnis | Rolle |
|---|---|
| `core/` | Die Generatoren selbst — reine Berechnung, kein Qt. 16 Module, z. B. `terrain_generator.py`, `geology_generator.py`, `weather_generator.py`, `water_generator.py`, `erosion_generator.py`, `biome_generator.py`, `settlement_generator.py`, `terrain_weltkarte.py` (die neun Regionen, Abschnitt 5), `spielkarten.py` (die Naht zum Spiel, Abschnitt 7). |
| `managers/` | Koordination: `calculator_graph.py` (die Rechenkette, Abschnitt 3), `generation_orchestrator.py` (Datenfluss, Abschnitt 4), `shader_manager.py` (GPU-Dispatch, siehe `CLAUDE.md`), `parameter_manager.py`, `navigation_manager.py`, `data_lod_manager.py`, `teilschritte.py`. |
| `gui/` | Qt-Oberfläche: `tabs/` (12 Reiter-Klassen, Abschnitt 6), `widgets/` (2D-/3D-Anzeige, u. a. `map_display_2d.py`, `map_display_3d.py`, `adaptive_terrain_mesh.py`), `config/` (Parameter-Defaults, u. a. `value_default.py`). |
| `shaders/` | GLSL-Compute-Shader für den GPU-Pfad (harte Sperre im Nachtbetrieb, siehe `nachtbetrieb/sperrliste.toml`). |
| `nachtbetrieb/` | Werkzeuge des nächtlichen Ticket-Betriebs selbst: `sperre.py`/`sperrliste.toml` (Sperrliste), `branch.py`, `zeitgrenze.py`, `morgenbericht.py`. |
| `tools/` | Bedienung von außen: `nachtlauf.py` (Kommandozeile für den Nachtbetrieb), `test_raenge.py`, `testlauf.py` (Abschnitt 8). |
| `tests/` | 81 Dateien, Konvention `smoke_test_*.py` (Abschnitt 8). |
| `docs/` | Dokumentation; `docs/archiv/` historisch, siehe `docs/archiv/README.md`. |

Das Oberziel des Programms steht in `docs/spezifikation/01_ZIEL.md` §2: reale
Landschaften der Erde nachbilden, über wenige verständliche Regler, ohne dass
ein Reglerstand die Welt zerstören kann.

---

## 2. Wortklärung

- **Reiter (Tab):** ein Abschnitt der Arbeitsoberfläche, z. B. „Terrain" oder
  „Wasser" — in Qt-Begriffen ein `QWidget`, das `MapEditorWindow` in seine
  Reiterleiste einhängt (Abschnitt 6).
- **Rechenkette (Calculator Graph):** die Abfolge der 38 Berechnungsschritte,
  von der Rausch-Erzeugung des Geländes bis zu den Siedlungen — jeder Schritt
  kennt seine Vorgänger und seine Ausgabe (Abschnitt 3).
- **Naht:** die eine Stelle, an der zwei Teile des Programms zusammentreffen
  und sich auf ein gemeinsames Format einigen müssen. Für die Anzeige ist das
  die 2D/3D-Regel aus `CLAUDE.md`; für das fertige Ergebnis ist es
  `core/spielkarten.py` (Abschnitt 7).
- **Waechter/Eichung:** zwei Ränge der Tests. Wächter laufen bei jeder
  Änderung (unter zwei Minuten), Eichung nachts mit beliebiger Dauer
  (Abschnitt 8).

---

## 3. Die Rechenkette (Calculator Graph)

Quelle: `managers/calculator_graph.py`, Klasse `CalculatorSpec` (Zeile 35).
Jeder Knoten trägt einen Namen (`Stufe.Schritt`), eine Stufe, eine Liste
seiner Vorgänger-Knoten und die Namen seiner Ausgaben. `CalculatorDispatcher`
(Zeile 588) und `CalculatorRoundScheduler` (Zeile 434) führen die Knoten in
der durch die Abhängigkeiten erzwungenen Reihenfolge aus.

**38 aktive Knoten** (per `grep -c` gegen `managers/calculator_graph.py`
gegengeprüft, Stand dieser Sitzung), gruppiert in sieben Stufen:

| Stufe | Knoten |
|---|---|
| `terrain` | noise → redistribution → slope, shadow |
| `geology` | layer_thickness, tectonic_displacement → outcrop → intrusions → sediment_overlay, metamorphic_overprint → rock_color, hardness |
| `weather` | temperature, wind, humidity, precipitation |
| `erosion` | hydraulic → slope |
| `water` | lake_detection → flow_network → manning_flow → soil_moisture, evaporation |
| `biome` | preseed_hint, base_classification → super_override → integrate_layers → supersampling, climate_classification |
| `settlement` | suitability → settlements → city_boundary → pathfinding → roadsites, civ_influence → landmarks → landmark_roads → plot_nodes |

**Zwei Knoten sind im Code vorhanden, aber auskommentiert** und laufen
deshalb nicht: `water.erosion_sedimentation` (Zeile 222) und
`water.thermal_erosion` (Zeile 239). Wer diese Stufen sucht, findet sie nicht
im aktiven Graphen — das ist kein Fehlen der Doku, sondern der tatsächliche
Code-Zustand zum Zeitpunkt dieser Sitzung.

Die Reihenfolge ist wichtig: `terrain.redistribution` liefert die
Höhenkarte, von der `geology.tectonic_displacement`, `erosion.hydraulic` und
letztlich jede spätere Stufe abhängen. Ein Knoten läuft erst, wenn alle
seine in der Liste genannten Vorgänger fertig sind (`CalculatorRoundScheduler`).

---

## 4. Datenfluss und Orchestrierung

`managers/generation_orchestrator.py` (1792 Zeilen) ist die Schicht, die den
Calculator Graph mit der GUI verbindet: Reiter lösen über den
`GenerationOrchestrator` eine Neuberechnung aus, der Orchestrator ruft den
`CalculatorDispatcher` mit den geänderten Parametern auf und meldet
Fortschritt und Ergebnis an die Reiter zurück (Signale, kein direkter
Methodenaufruf zwischen Reitern).

Zwei zusätzliche Naht-Stellen im Datenfluss:

- **GPU/CPU:** `managers/shader_manager.py`, `GPUWorker` — jede
  GPU-Operation fängt ihren eigenen Fehler ab und fällt auf den CPU-Pfad
  zurück. Das ist eine bewusste Entscheidung (Verfügbarkeit ohne GPU), aber
  jeder Rückfall braucht laut `CLAUDE.md` eine laute Logzeile, weil er sonst
  von einem Erfolg nicht zu unterscheiden ist.
- **2D/3D:** dieselbe Änderung wird für `MapDisplay2D` und `MapDisplay3D`
  gebaut (`gui/widgets/map_display_2d.py`, `map_display_3d.py`), meist ohne
  neuen GLSL-Code über `rasterize_*_rgba()` → `update_overlay_data()` als
  Alpha-Textur auf das Gelände gelegt. `tests/smoke_test_display_methoden_existieren.py`
  bewacht das: einseitige Anzeigemethoden müssen dort namentlich als
  `NUR_EINE_ANZEIGE` (begründet) oder `FEHLT_IM_3D` (Schuld) eingetragen sein.

---

## 5. Die neun Regionen

Quelle: `core/terrain_weltkarte.py`, `REGIONEN` (Zeile 276 ff.), bestätigt
gegen `docs/spezifikation/10_REGIONEN.md` Teil A (dort am 2026-09-17 unter
Ticket #43 von einem 20 Einzel-Landschaften umfassenden Katalog auf diese
neun berichtigt).

Drei Gruppen zu je drei Regionen, jede mit Namen, Zielvolk, Zielwerten
(Höhe, Relief, Formgröße, Rauheit, Wasseranteil, Flächenfaktor,
Küstenform, Temperatur, Niederschlag, Wind, Hangtrockenheit, Talform):

| Gruppe | Regionen |
|---|---|
| NORD | Clonagh (Kelten, sanfte Wellen), Skerrheim (Wikinger, ein Hauptfjord), Morobora (Slawen, flaches Hochland) |
| MITTE | Estrande (Franken, Küstenebene mit Ästuar), Nevadin (Alemannen, Trogtäler/Grate), Nebelrode (Sachsen, dichte Zertalung) |
| SÜD | Samarcia (Andalusier, Trockentäler), Macchia (Italiener, Küstengebirge), Thalassia (Byzantiner, Archipel) |

Diese neun Parametersätze steuern `weltfeld()` in `core/terrain_weltkarte.py`
und werden von `tests/smoke_test_regionen_welt.py` gegen feste Sollhänge und
Wasseranteile geprüft — laut `CLAUDE.md` der empfindlichste Wächter für
Geländeform im ganzen Projekt (fand z. B., dass die Küsten-Archetypen
Samarcia von Ziel 6.5 auf gemessen 10.6 verschoben hatten). Deshalb steht
`core/terrain_weltkarte.py` in der Sperrliste auf Stufe „warnung", nicht
„sperre": echte Geländearbeit findet dort statt, aber jede Berührung
erscheint im Morgenbericht.

**Zu unterscheiden von `docs/regionen/`:** dieses Verzeichnis enthält 21
reale Landschafts-*Vorbilder* (Sahara, Atacama, Alpen, Amazonas, ...) für
`tests/smoke_test_region_vorbild_aehnlichkeit.py` — das ist ein anderer,
größerer Katalog realer Referenzen zum Ähnlichkeitsabgleich, nicht dieselbe
Liste wie die neun spielbaren `REGIONEN`. Diese Unterscheidung stand in
keiner der gelesenen Archivdateien explizit und wurde erst beim
Gegenchecken gegen den Code sichtbar (siehe Abschnitt 9, Runde 3).

---

## 6. Die Reiter

Quelle: `gui/tabs/` (12 Klassen) und der Vertrag, den
`tests/smoke_test_reiter_vertrag.py` beschreibt.

| Reiter (Klasse) | Datei |
|---|---|
| KontinentTab | `kontinent_tab.py` |
| RegionTab | `region_tab.py` |
| OverviewTab | `overview_tab.py` |
| TerrainTab | `terrain_tab.py` |
| GeologyTab | `geology_tab.py` |
| WeatherTab | `weather_tab.py` |
| ErosionTab | `erosion_tab.py` |
| WaterTab | `water_tab.py` |
| RiverTab | `river_tab.py` |
| BiomeTab | `biome_tab.py` |
| SettlementTab | `settlement_tab.py` |
| SettlementRegionalTab | `settlement_regional_tab.py` |

**Der Vertrag lebt in der Shell, nicht im Reiter.**
`MapEditorWindow._add_successful_tab()` zerlegt jeden Reiter in drei
Pflicht-Attribute: `tab.viewport_widget` (Karte, 2D/3D-Stack),
`tab.parameter_widget` (Regler), `tab.statistics_widget` (Kennzahlen) — so
dokumentiert in `gui/tabs/base_tab.py`, Zeilen 6–13. Fehlt eines dieser drei
Attribute, wirft die Shell erst **nachdem** die Reiterbeschriftung schon
angelegt wurde: die Folge ist ein doppelter Reitername und eine Verschiebung
aller nachfolgenden Reiter um eins (Befund 2026-08-26, Docstring
`tests/smoke_test_reiter_vertrag.py`). Ein Reiter-Test, der nur den Reiter
für sich prüft (wie ursprünglich `smoke_test_region_tab.py`), sieht diesen
Fehler nicht — er lebt in der Verbindung zur Shell, nicht im Reiter selbst.

Die meisten Reiter erben von `BaseMapTab` (2D/3D-Kartenanzeige inklusive).
`KontinentTab` und `RegionTab` sind einfache `QWidget`, ohne Kartenanzeige
im 2D/3D-Sinn — dort ist die Fragestellung eine andere (Kontinent-Vorschau
bzw. Regionswahl).

---

## 7. Die Naht zum Spiel

Quelle: `core/spielkarten.py` (470 Zeilen), Docstring Zeilen 1–56.

Das Programm erzeugt eine zusammenhängende Weltkarte, ein tatsächliches
Spiel braucht aber einzelne, handliche, näherungsweise quadratische
Spielkarten. `core/spielkarten.py` ist die Stelle, die diesen Übergang
herstellt: Sie zerlegt die Weltkarte in **neun** zusammenhängende Vielecke
mit je etwa gleich viel Landmasse.

Verfahren (Nutzer-Vorgabe 2026-08-13, `docs/OFFENE_PUNKTE.md` 5.15):

- **Power-Diagramm (gewichtetes Voronoi)** statt Kostenfeld-Zerlegung: die
  Zellen sind konvex und damit garantiert kompakt genug für eine quadratische
  Darstellung (Bedingung a der Vorgabe).
- **Saatpunkte auf den Siedlungsschwerpunkten**, nicht frei gewählt: dadurch
  liegt jede Kartengrenze automatisch mittig zwischen zwei
  Siedlungsclustern, nicht mitten durch eine Stadt (Bedingung b der Vorgabe).
- **Kapazitätsausgleich über das Zusatzgewicht `lambda_i`**, nicht über
  Verschieben der Saatpunkte: `argmin_i (‖p − s_i‖² − lambda_i)`, iterativ
  angepasst, bis alle Zellen etwa gleich viel Landmasse tragen. Ein
  Verschieben der Saatpunkte (Lloyd-Relaxation) hätte Bedingung (b) wieder
  verletzt.
- **Küstenmeer vom Grad 1 zählt zur Hälfte als Landmasse** (Gewichtsfeld 1.0
  Land, 0.5 küstennahe See, 0.0 offene See) — Nutzer-Vorgabe: „der erste
  seegrad an kueste zaehlt auch als 50% landmasse, da hier viel passiert".

Geprüft von `tests/smoke_test_spielkarten.py`. Das ist die einzige Stelle im
Programm, die aktiv ein spielbares Kartenformat erzeugt — der
`managers/export_manager.py`, der vom Docstring her ebenfalls wie eine
Export-Naht aussieht, ist vollständig auskommentiert (Kopfzeile im Code:
„DERZEIT AUSKOMMENTIERT -> WIRD IN ZUKUNFT VIELLEICHT REVIVED") und damit
kein aktiver Code-Pfad — eine Verwechslungsgefahr, die beim Gegenchecken
aufgefallen ist (Abschnitt 9, Runde 2).

---

## 8. Tests

81 Dateien unter `tests/`, Konvention `smoke_test_*.py` plus
`tests/toleranzen.toml` (Zahlenwerte, keine Testlogik) und `tests/__init__.py`.

**Zwei Ränge**, eingeführt in Ticket #46 (Commit `1be88ea`,
`nacht(#46): Testlaeufe in Waechter und Eichung getrennt`):

- **Wächter** (27 Dateien): laufen bei jeder Änderung, unter zwei Minuten
  Gesamtlaufzeit.
- **Eichung** (53 Dateien): laufen nachts, beliebige Dauer.

Einstufung je Datei mit Begründung steht in `tools/test_raenge.py`
(`EINSTUFUNG`-Dict) und stützt sich laut Commit-Nachricht auf
`docs/TESTBESTAND_BEWERTUNG.md` sowie auf konkrete, in `CLAUDE.md`
dokumentierte Vorfälle — nicht auf pauschale Kriterien wie Laufzeit allein.
`tools/testlauf.py --rang {waechter,eichung}` ruft
`test_raenge.dateien_im_rang()` auf; eine Datei ohne Einstufung löst bewusst
`RangUnbekannt` aus statt stillschweigend übergangen zu werden (Vorbild:
`SperrlisteKaputt` in `nachtbetrieb/sperre.py`, Abschnitt 1).

**Bewusste Ausnahme:** `tests/smoke_test_regionen_welt.py` bleibt trotz
80.8 s Laufzeit in der Eichung, nicht im Wächter-Budget von zwei Minuten —
mit der ausdrücklichen Begründung, dass er trotzdem der empfindlichste
Wächter für Geländeform im Projekt ist (Commit-Nachricht `1be88ea`).

Ein Teil der Tests (GPU-Shader-Pfade, s. `smoke_test_erosion_gpu_parity.py`)
läuft headless über eine `QGuiApplication` ohne sichtbares Fenster — Details
und die Falle mit `QT_QPA_PLATFORM=offscreen` (deaktiviert den echten
GL-Kontext, GPU-Tests fallen dann still auf CPU zurück) stehen in
`CLAUDE.md`, Abschnitt „Compute shaders CAN be run headlessly".

---

## 9. Verfahren dieses Handbuchs (fünf Durchgänge)

1. **Schreiben** — aus `docs/archiv/` und den zum Zeitpunkt sichtbaren
   `docs/*.md` (dieser Text, erste Fassung).
2. **Gegenchecken gegen den Code** — durchgeführt während des Schreibens:
   `REGIONEN` in `core/terrain_weltkarte.py` gegen die in
   `docs/SPEZIFIKATION.md` §2 behauptete Neun-Regionen-Liste geprüft
   (Übereinstimmung: Namen, Gruppen NORD/MITTE/SÜD); die 38 Knotenzahl der
   Rechenkette per `grep -c` gegen `managers/calculator_graph.py`
   nachgezählt (nicht nur aus einem Dokument übernommen); der
   Reiter-Vertrag (`viewport_widget`/`parameter_widget`/`statistics_widget`)
   gegen `gui/tabs/base_tab.py` und `tests/smoke_test_reiter_vertrag.py`
   geprüft; `managers/export_manager.py` als toter Code identifiziert
   (vollständig auskommentiert), nachdem er zunächst als Kandidat für „die
   Naht zum Spiel" in Betracht gezogen wurde — verworfen zugunsten von
   `core/spielkarten.py`, das tatsächlich aktiver Code ist und explizit auf
   eine Nutzer-Vorgabe verweist.
3. **Fehlende Informationen** (im Archiv nicht dokumentiert, aber für das
   Handbuch nötig — durch Code-Lektüre ergänzt):
   - Dass `docs/regionen/` (21 reale Vorbilder) und die neun `REGIONEN` in
     `terrain_weltkarte.py` zwei verschiedene, leicht verwechselbare Kataloge
     sind, stand in keiner gelesenen Archivdatei; nur aus dem Code (Ordner
     vs. Modul) und dem zugehörigen Test (`smoke_test_region_vorbild_aehnlichkeit.py`)
     erschlossen.
   - Dass zwei Knoten der Rechenkette (`water.erosion_sedimentation`,
     `water.thermal_erosion`) im Code zwar vorhanden, aber auskommentiert
     sind, steht in keinem gelesenen Dokument — nur im Quelltext sichtbar.
   - Die genaue Formulierung des Reiter-Vertrags
     (`_add_successful_tab()`, drei Pflicht-Attribute) stand nicht in einem
     Übersichtsdokument, sondern ausschließlich im Docstring eines
     einzelnen Tests.
   - Dass `managers/export_manager.py` vollständig toter Code ist, ist eine
     reine Code-Tatsache ohne Entsprechung in der gelesenen Dokumentation.
   - **Nicht ermittelt, weil außerhalb des Zeitbudgets:** eine
     vollständige Durchsicht aller 38 Rechenknoten-Implementierungen
     Zeile für Zeile (nur Namen, Reihenfolge und Stufenzugehörigkeit wurden
     gegengeprüft, nicht jede einzelne Berechnungsformel); ebenso keine
     Live-Prüfung der GUI (das Handbuch beschreibt Code-Struktur, keine
     Bildschirm-Verifikation, siehe `CLAUDE.md` „Verifying GUI changes").
4. **Erneut geschrieben** — die obigen Korrekturen (Regionen-Verwechslung,
   auskommentierte Wasser-Knoten, `export_manager.py` als toter Code) sind
   bereits in Abschnitt 3, 5 und 7 dieser Fassung eingearbeitet, nicht als
   separater zweiter Text.
5. **Erneut gegengecheckt** — vor Abschluss dieser Fassung wurden die
   Kernzahlen ein zweites Mal gegen den Code gestellt: 38 aktive
   `CalculatorSpec`-Einträge (`grep -c`), 81 Testdateien
   (`ls tests | wc -l` analog, Konvention `smoke_test_*.py`), 12
   Reiter-Klassen in `gui/tabs/`, 9 `REGIONEN`-Einträge in drei Gruppen. Alle
   vier Zahlen stimmen mit den in diesem Text verwendeten überein.

---

## 10. Offene Punkte für den Nutzer

- **`CLAUDE.md` verweist noch nicht auf dieses Handbuch.** `CLAUDE.md` steht
  in `nachtbetrieb/sperrliste.toml` auf Stufe „sperre" (Eintrag
  „Anweisungen an den Agenten") und darf im Nachtbetrieb nicht verändert
  werden. Der Verweis auf `docs/HANDBUCH.md` als erste Anlaufstelle muss von
  Hand ergänzt werden.
- **Abgleich nach dem Merge von #51 nötig.** Dieses Handbuch wurde gegen den
  Dokumentenstand *vor* der Archiv-Verschiebung aus #51 geschrieben, weil der
  Branch `nacht/2026-09-19` in diesem Arbeitsbaum nicht sichtbar ist. Nach
  dem Merge bitte prüfen, ob sich Dateipfade unter `docs/` geändert haben,
  auf die dieses Handbuch verweist (aktuell: `docs/spezifikation/`,
  `docs/OFFENE_PUNKTE.md`, `docs/TESTBESTAND_BEWERTUNG.md`,
  `docs/regionen/`, `docs/archiv/README.md`).
- **Die 38 Rechenknoten sind hier nur mit Name und Reihenfolge erfasst,
  nicht mit ihrer jeweiligen fachlichen Berechnung.** Wer eine bestimmte
  Formel sucht (z. B. wie `erosion.hydraulic` genau rechnet), findet das nur
  im jeweiligen `core/*.py`-Modul selbst, nicht in diesem Handbuch.
