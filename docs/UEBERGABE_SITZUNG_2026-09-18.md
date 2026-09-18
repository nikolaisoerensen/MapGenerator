# Übergabe Nachtlauf 2026-09-18

Branch `nacht/2026-09-18`, 18 Commits, `main` unberührt, nichts gepusht.
Dieser Text ergänzt den maschinell erzeugten
`nachtbetrieb/laufberichte/morgenbericht.md` (nur Ticketliste + Testbefund)
um drei Dinge, die dort fehlen: **wie man jeden Punkt selbst prüft**, **was
im laufenden Programm jetzt anders aussieht**, und **was noch offen ist,
mit Begründung**.

Zwei Begriffe vorab, die unten oft vorkommen:

- **Ticket / Issue**: dieselbe Sache, zwei Namen. Die Nummer (`#37`) ist die
  GitHub-Issue-Nummer. Ein Commit auf dem Nachtbranch schließt das Issue
  NICHT automatisch — das passiert erst, wenn du es nach dem Review manuell
  auf GitHub schließt oder der Branch nach `main` gemerged wird. Deshalb
  stehen unten unter "erledigt" Tickets, die auf GitHub noch als offen
  angezeigt werden.
- **Naht**: die eine Stelle im Programm, an der mehrere Module dieselbe
  Auskunft abholen müssen, damit sie sich nicht widersprechen — z.B. die
  Liste der Felder, die eine Siedlung nach außen zeigt (`docs/SIEDLUNGEN_ENTWURF.md`).

---

## 1. Was diese Nacht committet wurde (Überblick)

16 nummerierte Tickets + 1 Dokumentkorrektur ohne Nummer + 1 Code-Review-
Nachbesserung ohne Nummer = 18 Commits. Kein Ticket ist an der Zeitgrenze
steckengeblieben, kein Ticket wegen Sperrliste ausgelassen.

| Themenblock | Tickets |
|---|---|
| Flüsse & Wasser | #37, #33, #32 |
| Siedlungen | #34, #35, #72, #84 |
| Terrain-Anzeige & Erosion | #61, #62, #74, #30 |
| Speichern/Laden & GUI | #38, #40 |
| Dokumentation & Prüfwerkzeug | #53, #45, TESTBERICHT-Korrektur |
| Sonstiges | #71 |
| Code-Review-Nachbesserung (kein Ticket) | 4 Funde, ein Commit |

---

## 2. Je Themenblock: was geändert wurde, wie man es prüft, was jetzt anders aussieht

### Flüsse & Wasser

**#37 — Flussnetz als Linienzüge exportieren.** Bisher gab es Flüsse nur als
Höhenfeld (ein Bild mit eingegrabenen Tälern). `TerrainGenerator._weltfluesse()`
(`core/terrain_generator.py`) liefert jetzt zusätzlich `river_lines`: die
echte Mittellinie jedes Flusses als Liste von Koordinatenpunkten, direkt aus
dem Berechnungsgraphen entnommen statt nachträglich aus dem Bild
rekonstruiert.
*Prüfen:* `.venv/Scripts/python.exe tests/smoke_test_flussexport_vektor.py`
— zieht 199 Flusslinien Ende-zu-Ende durch Graph → `TerrainData` →
`DataLODManager` und prüft sie gegen das Höhenfeld.
*Sichtbar anders:* noch nicht in der GUI verdrahtet (kein Regler/Anzeige
dafür) — die Daten stehen jetzt bereit, ein künftiges Ticket kann sie z.B.
für den Godot-Export oder eine Vektor-Anzeige nutzen.

**#33 — Sinuosität als Mäander-Kennzahl.** "Sinuosität" heißt: wie sehr ein
Fluss mäandriert, gemessen als Verhältnis Flusslänge zu Luftlinie (1.0 =
schnurgerade). Neu in `core/fluss_sinuositaet.py`.
*Prüfen:* `tests/smoke_test_fluss_sinuositaet.py` (6 Testgruppen).
*Sichtbar anders:* nichts in der GUI — reine Kennzahl für spätere
Kalibrierung. **Aber:** dieses Modul rekonstruiert die Fluss-Geometrie noch
aus dem Raster-Bild, obwohl #37 (auch diese Nacht) die echten Linienzüge
liefert. Das ist Befund 3 aus dem Code-Review unten (nicht behoben).

**#32 — Seenfläche neu gemessen.** Reine Messung, kein Code geändert: die
alte Behauptung "Seen bedecken effektiv 0%" war falsch — neu gemessen
2.2%/0.9%/0.3% je nach Kartengröße.
*Prüfen:* `tests/smoke_test_seenflaeche_messung.py`.
*Sichtbar anders:* nichts — die Seen waren immer da, nur der Messwert dazu
war falsch.

### Siedlungen

**#34 — Biomkarte fließt jetzt in die Standortwahl für Siedlungen ein.**
Vorher konnte eine Stadt in Wüste, Sumpf oder Eiskappe entstehen, weil die
Eignungsrechnung (`_calc_suitability` in `core/settlement_generator.py`) das
Biom (die Vegetationszone) gar nicht kannte.
*Prüfen:* `tests/smoke_test_settlement_placement.py` — vorher 6 Verstöße
("2 Städte wo nur 1 sein sollte"), jetzt 0.
*Sichtbar anders:* im 2D/3D-Kartenbild entstehen künftig generierte Städte
nicht mehr in offensichtlich ungeeignetem Gelände — messbar: Anteil
Siedlungen in ungeeignetem Biom sank von 29,8 % auf 0,0 % über 5 Testläufe.

**#35 — Die Siedlungsnaht-Feldliste ist jetzt festgeschrieben.** Reines
Testgerüst (`smoke_test_siedlungsnaht_felder.py`), das garantiert, dass die
im Entwurfsdokument versprochenen Felder (Stadtgröße, Kultur, Rang, Kontur,
Wege-Anschlüsse) auch wirklich da sind, statt nur dokumentiert.
*Sichtbar anders:* nichts direkt, aber jede künftige Änderung, die eines
dieser Felder aus Versehen wegließe, würde jetzt sofort auffallen.

**#72 — Kontur der Stadtgrenze als Polygon.** Bisher gab es die Stadtgrenze
nur als Rastermaske (welches Pixel gehört zur Stadt). Jetzt zusätzlich als
Polygon (eine Liste von Eckpunkten) — Voraussetzung dafür, dass später
(#73) berechnet werden kann, wo genau eine Straße die Stadtgrenze
durchquert.
*Prüfen:* `tests/smoke_test_siedlungsnaht_felder.py` (Feld "Kontur der
Stadtgrenze" jetzt grün). Zusätzlich habe ich selbst — siehe Code-Review
unten — einen eigenen Testlauf gefahren, weil dieser bestehende Test die
neue Datenschicht nicht wirklich prüft.
*Sichtbar anders:* noch nicht in der GUI gezeichnet — die Daten liegen
jetzt in `SettlementData.city_boundary_polygons` bereit.

**#84 — Marktstadt-Randkorrektur reparierte den falschen Rang.** Ein Dorf
am Kartenrand, das wegen der Randregel aufgewertet werden musste, sprang
direkt auf "Stadt" statt auf den nächsthöheren erlaubten Rang. Fix in
`_typen_zuweisen()` (`core/settlement_generator.py`, um Zeile 5500).
*Prüfen:* `tests/smoke_test_settlement_placement.py`.
*Hinweis:* derselbe Test zeigt bei manchen Seeds weiterhin 2 statt 1 Stadt
je Kultur — das liegt nachweislich an einer vorbestehenden
Nichtdeterminismus-Eigenschaft der 96px-Testkarte, nicht an diesem Fix
(Baseline-Vergleich ohne den Fix zeigt dieselbe Instabilität).

### Terrain-Anzeige & Erosion

**#61 — Erosionsfilter-Regler verständlich benannt.** Reine Umbenennung von
Klassenattributen im Code (die Regler-Beschriftung in der GUI hieß schon
vorher verständlich) — die Parameter-Schlüssel, die gespeicherte Welten
referenzieren, sind unverändert.
*Sichtbar anders:* nichts im laufenden Programm, nur im Quelltext leichter
zu lesen.

**#62 — Geologie-Querschnitt schneller gezeichnet.** Nicht die Berechnung
war zu langsam, sondern das Zeichnen mit Matplotlib — bei 1024 px dauerte
das Neuzeichnen 188 statt jetzt 123 Millisekunden (-35 %), weil die
Stichprobe für den Plot auf 512 Punkte gekappt wurde (echte Werte, kein
Schätzen).
*Sichtbar anders:* der Geologie-Querschnitt-Tab reagiert beim Verschieben
des Schnitts spürbar flüssiger, das Bild selbst ist unverändert.

**#74 — Erosions-Farbskala kalibriert.** Die Farbskala für die
Erosions-/Ablagerungskarte war auf einen Mindestwert (`vmin=0.5`) geeicht,
die tatsächlichen Werte lagen aber bei 0.2-0.3 — die Karte war praktisch
einfarbig.
*Sichtbar anders:* Erosions- und Sedimentationskarte zeigen jetzt
tatsächlich Farbverläufe, **in 2D und 3D gleichermaßen**, weil beide über
dieselbe Naht (`CanvasSettings.CANVAS_2D`) eingefärbt werden.

**#30 — CPU-Erosion vermessen, keine Konsequenz gezogen.** Reine Messung:
CPU-Vollauf bei 1024 px braucht 8-10 Minuten gegen 8-16 Sekunden auf der
GPU (Faktor 37-65). Dieser Pfad wird im Normalbetrieb aber gar nicht
genommen, weil eine Grenze (`MAX_CPU_RESOLUTION=256`) das verweigert.
*Neuer, nicht behobener Nebenbefund:* schlägt die GPU mitten in einem
Erosionslauf fehl, fällt das Programm ungeprüft auf die (sehr langsame)
CPU-Schleife zurück, ohne das laut zu melden. Rücksprache mit dir nötig,
bevor das angefasst wird.

### Speichern/Laden & Datei-Menü

**#38 — `welt_backen()`/`welt_laden()` als einzige Speicher-Naht.** Vorher
gab es mehrere Stellen im Code, die Weltdaten einzeln in Dateien
schrieben/lasen. Jetzt läuft alles über genau diese zwei Funktionen.
*Prüfen:* `tests/smoke_test_welt_io_roundtrip.py` (bitgenauer Rundlauf),
`tests/smoke_test_export_2048.py` (Godot-Export unverändert).

**#40 — Datei-Menü an echtes Speichern/Laden angeschlossen.** Die Menüpunkte
"Speichern"/"Öffnen"/"Exportieren" waren bisher Attrappen ohne Wirkung.
*Prüfen:* `tests/smoke_test_dateimenue_welt_io.py` (9/9 Gruppen, inkl.
Fehlerfall als Dialog statt Absturz).
*Sichtbar anders:* Datei-Menü speichert/lädt jetzt wirklich. **Ein
Live-Rundlauf im echten Programm (speichern → schließen → öffnen) ist noch
nicht von dir bestätigt** — das kann kein Test ersetzen.

### Dokumentation & Prüfwerkzeug

**#53 — Harte Dokumentbehauptungen maschinell gegen den Code geprüft.**
Ein neuer Test (`tests/smoke_test_dokumentbehauptungen.py`) sucht in den
Dokumenten nach Sätzen, die eine konkrete, prüfbare Zahl behaupten (z.B.
"der Berechnungsgraph hat 38 Knoten") und vergleicht sie automatisch mit
dem Code. Fand dabei 5 Stellen, an denen die Doku "38" behauptete, während
der Code eine andere Zahl hatte — korrigiert.

**#45 — Testbestand gesichtet und bewertet.** Alle 78 vorhandenen
Testdateien (nicht 70, wie im ursprünglichen Ticket vermutet) wurden
gelesen und ausgeführt. Ergebnis in `docs/TESTBESTAND_BEWERTUNG.md`: 68
grün, 6 neue Befunde — genau diese 6 Befunde sind jetzt als eigene,
frische GitHub-Issues angelegt (**#75 bis #85**, siehe Abschnitt 3).

### Code-Review-Nachbesserung (kein eigenes Ticket)

Nach Abschluss aller Tickets lief ein eigener Code-Review-Durchgang gegen
`main` (8 unabhängige Prüfwinkel). 4 kleine, sichere Funde direkt behoben,
Commit `2e30700`:

1. **Absturz in der Live-Flussvorschau** (`gui/tabs/river_tab.py`): rief
   `_weltfluesse()` noch mit der alten Anzahl Rückgabewerte auf, seit #37
   sind es einer mehr — jeder Reglerzug im Fluss-Tab wäre mit einem Absturz
   quittiert worden. *Geprüft:* `tests/smoke_test_fluss_vorschau.py`,
   komplett grün.
2. **Doppelter Anzeige-Pfad** (`gui/map_editor.py`): eine Funktion baute
   sich einen eigenen, zweiten Weg, um nach dem Laden einer Welt alle
   Tabs neu zu zeichnen, statt den bereits vorhandenen (`_refresh_all_displays()`)
   zu benutzen — genau das Muster, das laut CLAUDE.md schon dreimal zu
   lautlos fehlenden 3D-Overlays geführt hat.
3. **Doppelte Flächen-Skalierung** (`core/settlement_generator.py`):
   dieselbe Zahl wurde an zwei Stellen unabhängig berechnet, statt die
   bereits vorhandene wiederzuverwenden.
4. **Datenverlust bei Stadtgrenzen-Polygonen**: das Ergebnis von #72
   (`city_boundary_polygons`) erreichte nie `SettlementData` oder die
   Speicherschicht (`DataLODManager`) — jeder Aufruf von
   `get_settlement_data()` oder `welt_backen()` hätte das Feld leer
   zurückgegeben. *Geprüft mit einem eigenen Testlauf* durch den echten
   Siedlungsgenerator: 37 Siedlungen mit Polygon, unverändert nach
   Rundreise durch den Manager.

---

## 3. Was noch offen ist, und warum

### 3.1 Frisch entdeckt durch die Testbestand-Sichtung (#45), noch nicht angefasst

Das sind 9 neue Issues (**#75–#83, #85**), alle heute (18.09.) automatisch
aus den Testbefunden von #45 angelegt. Kein Code dafür geändert.

| # | Befund | Einordnung |
|---|---|---|
| #85 | Julitemperatur-Jahresmittel bei 3 von 36 Region/Seed-Kombinationen knapp (0.74-0.85K) außerhalb der ±0.7K-Toleranz | Kalibrierungsfrage, keine Fehlfunktion |
| #83 | `settlement.landmark_roads` liefert auf der Testkarte keine Daten | vermutlich Folgefehler von #81 (siehe dort) |
| #82 | `settlement.plot_nodes`/`plots` liefert keine Daten, obwohl Eingaben vorhanden sind | zusammen mit einer WARNING "7 von 39 Stadtkernen ohne Voronoi-Nachbarn" — vermutlich dieselbe Ursache wie eine bekannte, nicht konvergierende Physik-Schleife |
| #81 | `settlement.landmarks`/`landmark_list` liefert keine Daten, obwohl Eingaben vorhanden sind | noch nicht lokalisiert, wahrscheinliche Ursache für #83 |
| #80 | `settlement.pathfinding`/`sea_roads` liefert keine Daten | erst zu klären, ob das für diese Testkarte überhaupt korrekt wäre |
| #79 | `height_delta` wird vom Pipeline-Test fälschlich als Fehler gemeldet | **Fehlalarm, kein echter Bug** — der Wert ist laut Modul-Dokumentation absichtlich immer Null. Nur der Test muss das als Sonderfall kennen. Schnell zu schließen. |
| #78 | Dieselbe Landschaft sieht bei 512px und bei 1024px messbar unterschiedlich aus (Korrelation 0.94 statt gefordert >0.98) | strukturell, vermutete Ursache: eine Rauschkoordinaten-Skalierung in `core/terrain_weltkarte.py` |
| #77 | An Regionsgrenzen ist das Gelände 82% steiler als im Regionsinneren, erlaubt sind höchstens 25% | sitzt in der Übergangs-/Blendfunktion zwischen benachbarten Regionen, nicht in den Regionen selbst |
| #76 | Küsten-Archetypen (aus einer früheren Nacht) haben die Region Macchia/Thalassia aus ihrer Eichung geschoben | bereits als bekanntes Muster in CLAUDE.md dokumentiert ("Geländeänderungen verstimmen zuerst die Regionseichung") |
| #75 | Erosion trifft mehrere Zielwerte nicht (Kanalnetz zu kurz, zu wenig ebene Flächen, Schwelle wirkt verkehrt herum, Schrittzahl skaliert zu steil) | Ursache noch ungeklärt, größter Einzelbefund dieser Nacht |

**Das sind die größeren/riskanteren Funde, die laut Nachtbetriebs-Regel
nicht selbst angefasst werden — sie brauchen deine Einschätzung, in
welcher Reihenfolge sie etwas wert sind.**

### 3.2 Jetzt entsperrt

**#73 — Anschlusspunkte der Wege an der Stadtgrenze.** War bisher blockiert,
weil die Kontur der Stadtgrenze (#72) fehlte. Die ist jetzt da — #73 kann
als nächstes angegangen werden.

### 3.3 Grundsatz-/Rechercheticket (keine Coding-Tickets, brauchen erst eine Entscheidung von dir)

- **#66** — welche Kennzahlen eine Landschaft überhaupt beschreiben sollen
- **#67** — reale Vorbildorte je Region beschaffen (Referenzmaterial)
- **#68** — ob die Geologie eine Erzwahrscheinlichkeit hergeben soll
- **#69** — Rohstoffkategorien, Stoffe, Fertigungszweige (Spielinhalt, nicht Technik)

Diese vier sind mit `wayfinder:research`/`wayfinder:grilling` markiert —
das sind Recherche-/Diskussionstickets, kein Auftrag an einen Code-Agenten.

### 3.4 Test-Infrastruktur aufräumen (#46–#54, alle noch offen außer #45/#53)

Zusammenhängender Themenblock aus der großen Testbestand-Aktion: Bandgrenzen
als versionierte Daten festschreiben (#47), Seedführung vereinheitlichen
(#48), Wächter- von Eichungsläufen trennen (#46), die zehn bekannten roten
Tests einzeln aufschlüsseln (#49 — **das ist die Quelle der 9 neuen Issues
oben**), Referenzbild-Vergleich aufsetzen (#50), stille Rückfälle laut
machen (#54), Dokumente ins Archiv schieben und daraus das Handbuch
schreiben (#51, #52).

### 3.5 Dokument-Zusammenführung

**#43, #44, #13** — Spezifikation und Sollbeschreibung sollen zu einer
Datei verschmelzen und auf die neun Regionen umgeschrieben werden. Reine
Doku-Arbeit, kein Codepfad betroffen.

### 3.6 Wegekosten (Pfadfindung für Siedlungen)

**#41, #42** — Wegekostenstaffelung nach Wegart kalibrieren, Brücken und
Uferwege in die Kostenrechnung aufnehmen. Setzt auf dem bestehenden
A*-Wegenetz auf, noch nicht begonnen.

### 3.7 Ältere, grundsätzliche Punkte

- **#29** — die neun Regionsparametersätze (Rauheit, Wasseranteil,
  Küstenform je Region) sollen aus `core/terrain_weltkarte.py` in eine
  eigene Datei wandern. Das ist der Grund, warum diese Datei aktuell auf
  der Sperrliste mit Stufe "Warnung" steht (jede Berührung wird im
  Morgenbericht vermerkt) — sobald #29 erledigt ist, wird daraus laut
  `nachtbetrieb/sperrliste.toml` eine harte Sperre auf die alte Datei.
- **#3** — von Git generierte/lokale Dateien, die schon in `.gitignore`
  stehen, aus der Versionsverwaltung entfernen. Aufräumarbeit ohne
  Funktionsänderung.

---

## 4. Empfehlung für die Reihenfolge

1. **#79 zuerst schließen** — ist kein Bug, sondern ein Fehlalarm im Test;
   kostet fast nichts.
2. **#75 (Erosion verfehlt mehrere Zielwerte) vorziehen** — größter neuer
   Einzelbefund, betrifft die Geländequalität direkt sichtbar im Bild.
3. **#81 vor #83** angehen — #83 (leere `landmark_roads`) hängt vermutlich
   an #81 (leere `landmark_list`); beide zusammen als ein Strang behandeln
   statt getrennt.
4. **#73** ist jetzt entsperrt und technisch klein (baut auf #72 auf) —
   guter nächster Ticket-Kandidat für einen kommenden Nachtlauf.
5. **#40 (Datei-Menü) im laufenden Programm selbst einmal
   speichern→schließen→öffnen** — das ist der einzige Punkt aus dieser
   Nacht, den kein automatischer Test ersetzen kann.

Kein Merge nach `main`, kein Push — das bleibt wie vereinbart dein manueller
Schritt (`python tools/nachtlauf.py stand`, dann gezielt mergen).
