"""
Path: tools/test_raenge.py

Teilt den Testbestand unter tests/smoke_test_*.py in zwei Raenge (Ticket #46).

WAECHTER: laeuft bei JEDER Aenderung, Budget < 2 Minuten insgesamt. Enthaelt
NICHT die schnellsten Dateien, sondern die EMPFINDLICHSTEN - fuer jede Datei
hier muss ein realer Fehler nennbar sein, den sie schon gefangen hat oder
faengen wuerde. Drei Fehlerklassen sind namentlich in CLAUDE.md dokumentiert
und muessen abgedeckt sein:

  STILLER_RUECKFALL    - eine GPU-Operation faellt lautlos auf CPU zurueck,
                          ein hasattr()-Zweig trifft nie, ein try/except
                          schluckt einen echten Fehler und liefert ein
                          plausibles falsches Ergebnis.
  GELAENDEVERSTIMMUNG   - eine Aenderung an core/terrain_weltkarte.py
                          (weltfeld()) verschiebt Hang/Wasseranteil weg von
                          den festen Regionszielen.
  FEHLENDE_ANZEIGE      - ein Feature ist in 2D sichtbar, in 3D nicht (oder
                          umgekehrt) - meist die hasattr()-Weiche aus der
                          STEHENDEN REGEL in CLAUDE.md.

EICHUNG: laeuft nur nachts, beliebige Dauer. Alles, was Zahlen gegen ein
festes Zielband haelt (Kalibrierung), plus alles, was fuer jede Aenderung zu
teuer oder zu spezialisiert ist.

JEDE Datei unter tests/smoke_test_*.py MUSS hier eingetragen sein - das wird
von tools/testlauf.py beim Sammeln erzwungen (RANG_UNBEKANNT), nach demselben
Prinzip wie nachtbetrieb/sperre.py: lieber laut scheitern als eine Datei
lautlos in der Luecke verschwinden lassen (siehe CLAUDE.md, "Jeder stille
Rueckfall auf einen Ersatzpfad braucht eine laute Logzeile").

Grundlage der Einstufung: docs/TESTBESTAND_BEWERTUNG.md (Ticket #45, alle 78
damaligen Dateien einzeln gelesen und ausgefuehrt) plus eigene Messung dieser
Nacht fuer alle hier als WAECHTER gefuehrten Dateien (siehe Begruendung je
Eintrag) und fuer die zwei seither neu hinzugekommenen Dateien
(smoke_test_seenflaeche_messung.py, smoke_test_regionen_naht_schnell.py).

Aufruf (Beispiel, siehe tools/testlauf.py fuer die eigentliche Ausfuehrung):
    .venv/Scripts/python.exe tools/testlauf.py --rang waechter
    .venv/Scripts/python.exe tools/testlauf.py --rang eichung
"""

WAECHTER = "waechter"
EICHUNG = "eichung"
RAENGE = (WAECHTER, EICHUNG)


class RangUnbekannt(Exception):
    """Eine Testdatei ist keinem Rang zugeordnet - siehe Docstring oben."""


# {Dateiname: (Rang, Begruendung)}. Begruendung ist bewusst je Datei einzeln
# geschrieben, nicht pauschal - Ticket #46 verlangt das ausdruecklich.
EINSTUFUNG = {

    # ---------------------------------------------------------------
    # WAECHTER - 27 Dateien, gemessene Gesamtlaufzeit siehe SITZUNGSLOG.
    # ---------------------------------------------------------------

    "smoke_test_display_methoden_existieren.py": (WAECHTER,
        "Bewacht die haeufigste Fehlerklasse des Projekts (FEHLENDE_ANZEIGE): "
        "jede hasattr()-Anzeigemethode muss auf mindestens einer der beiden "
        "Displayklassen existieren. Fing real den leeren Reiter 'Siedlungen "
        "Regional' am 2026-08-13. 1.1 s."),
    "smoke_test_anzeige_register_3d.py": (WAECHTER,
        "FEHLENDE_ANZEIGE: jeder 3D-Anzeigemodus muss vollstaendig im "
        "Register stehen. Fing beim eigenen Bau einen realen Fehler "
        "(fehlender Flussreiter). 1.3 s."),
    "smoke_test_biome_overlays_3d.py": (WAECHTER,
        "FEHLENDE_ANZEIGE: BiomeTab.apply_overlays() muss 2D UND 3D "
        "erreichen. Fing real einen Fehler am 2026-09-16. 1.5 s."),
    "smoke_test_layer_2d_3d_parity.py": (WAECHTER,
        "Direkter Wächter der STEHENDEN REGEL 2D=3D (Range-Mapping, "
        "Colormap, Skala) ueber rund 140 Einzelpruefungen an synthetischen "
        "24x24-Layern. 1.1 s."),
    "smoke_test_push_overlays.py": (WAECHTER,
        "Zweiter Wächter derselben STEHENDEN REGEL: Overlay-Dispatch in "
        "2D UND 3D, ein unbekannter Overlay-Name muss einen Fehler werfen "
        "statt lautlos nichts zu tun (STILLER_RUECKFALL-Muster). 1.1 s."),
    "smoke_test_wege_geometrie.py": (WAECHTER,
        "FEHLENDE_ANZEIGE: die Wege sind laut CLAUDE.md der eine Fall, wo "
        "echte 3D-Geometrie statt einer Overlay-Textur noetig ist, weil sie "
        "beim Zoomen scharf bleiben und anklickbar sein muessen (kein "
        "Einsinken, Querprofil, Breite, Klick-Projektion, echter Shader). "
        "0.8 s."),
    "smoke_test_fluss_vorschau.py": (WAECHTER,
        "FEHLENDE_ANZEIGE: prueft die Live-Vorschau ueber den GEWOEHNLICHEN "
        "Anzeigeweg statt eines Sonderpfads - verhindert genau den Bug-Typ "
        "'Sonderpfad faellt in 3D lautlos aus'. 3.5 s."),
    "smoke_test_flussstufen.py": (WAECHTER,
        "Haette den dokumentierten 2026-08-24-Befund (Fluesse der Stufen "
        "'river'/'grand_river' kamen nie vor, weil eine Perzentilschwelle "
        "mit absoluten Faktoren multipliziert wurde) gefangen: prueft, dass "
        "alle vier Wasserstufen tatsaechlich vorkommen, auch in beiden "
        "Biom-Karten. 5.5 s."),
    "smoke_test_erosion_gpu_contract.py": (WAECHTER,
        "STILLER_RUECKFALL: statischer Abgleich Shader-Uniforms/Dispatcher "
        "gegen den tatsaechlichen GLSL-Code, faengt Namens-/Typfehler VOR "
        "einem teuren GPU-Lauf (genau die Fehlerklasse aus den drei "
        "2026-08-Shader-Debugrunden in CLAUDE.md). 1.1 s."),
    "smoke_test_erosion_gpu_parity.py": (WAECHTER,
        "STILLER_RUECKFALL: echter GPU-Erosionslauf gegen CPU-Referenz, "
        "laut CLAUDE.md 'Working example' fuer GPU-Tests und hat historisch "
        "3 reale GPU-Fehler gefangen. 1.4 s."),
    "smoke_test_shader_paths.py": (WAECHTER,
        "STILLER_RUECKFALL, woertlich: dieselbe Datei, die am 2026-07-30 "
        "den SHADERS_ROOT-Pfadfehler nach dem Verschieben von "
        "shader_manager.py aufdeckte (jede GPU-Operation faellt sonst "
        "lautlos und ohne Fehlermeldung auf CPU zurueck). Zum "
        "Messzeitpunkt ROT, aber nachweislich nur wegen der bekannten "
        "Worktree/Hauptcheckout-Pfaddivergenz (siehe "
        "docs/TESTBESTAND_BEWERTUNG.md), nicht wegen eines neuen Fehlers - "
        "bleibt trotzdem im Waechter, weil genau das seine Aufgabe ist. "
        "1.4 s."),
    "smoke_test_erosion_hauptschalter.py": (WAECHTER,
        "STILLER_RUECKFALL, Gegenprobe-Muster: EROSION_AKTIV=False muss "
        "ALLES auf null setzen, True ist die Gegenprobe - verhindert ein "
        "Schein-Passieren, bei dem ein Test nur deshalb gruen ist, weil der "
        "gepruefte Zweig nie ausgefuehrt wird. 3.5 s."),
    "smoke_test_stufen_schalter.py": (WAECHTER,
        "STILLER_RUECKFALL, laut Bewertung wortwoertlich 'genau die von "
        "CLAUDE.md gewarnte Fehlerklasse': prueft, dass drei "
        "Abschalt-Haekchen das Gelaende TATSAECHLICH veraendern, nicht nur "
        "einen ignorierten Parameter setzen. 4.5 s."),
    "smoke_test_nachtbetrieb.py": (WAECHTER,
        "STILLER_RUECKFALL auf der Meta-Ebene: prueft die Sperrliste, den "
        "Selbstschutz und die Commit-Regeln, die diesen Nachtlauf selbst "
        "absichern - eine kaputte Sperrliste waere der teuerste denkbare "
        "stille Rueckfall, weil sie den gesamten Schutzzaun abschaltet. "
        "Einziger Test fuer ein Werkzeug, das jede Nacht aktiv benutzt "
        "wird. 4.4 s."),
    "smoke_test_stille_rueckfaelle.py": (WAECHTER,
        "STILLER_RUECKFALL direkt am Namen: AST-Scan ueber die Hotspot- "
        "Dateien, der jeden neuen stillen except-Block gegen eine feste "
        "Allowlist prueft - genau die Fehlerklasse, die diesen ganzen "
        "Rang begruendet, hier als eigener Waechter statt nur als Lehre "
        "in CLAUDE.md. Nachtrag bei der #46-Pruefung eingestuft, weil "
        "das Worktree des #46-Agenten vor dieser (#54-)Datei von main "
        "abzweigte. 0.5 s."),
    "smoke_test_biome_preseed.py": (WAECHTER,
        "STILLER_RUECKFALL: prueft, dass CALCULATOR_GRAPH keinen Zyklus "
        "zwischen Feuchte und Biom enthaelt - eine kritische "
        "Graph-Invariante, deren Verletzung sich sonst als scheinbar "
        "plausibles, aber falsches Endergebnis zeigen wuerde. 0.9 s."),
    "smoke_test_water_pipe_flow.py": (WAECHTER,
        "STILLER_RUECKFALL: exakte Massenbilanz des PipeFlowSimulator ist "
        "laut Bewertung eine 'nicht-verhandelbare physikalische "
        "Invariante' - jede Abweichung waere ein stiller Rechenfehler, "
        "keine kosmetische Ungenauigkeit. 1.1 s."),
    "smoke_test_adaptive_terrain_mesh.py": (WAECHTER,
        "STILLER_RUECKFALL, woertlich die Datei aus der CLAUDE.md-Lehre "
        "'Gruene Tests koennen eine tote Funktion verdecken': eine falsche "
        "Vorbedingung (2^n+1 statt der echten 2er-Potenz-Kartengroessen "
        "dieses Projekts) liess das adaptive Mesh wochenlang unbemerkt nie "
        "laufen, stiller Rueckfall aufs alte Gleichmaessig-Gitter. Test "
        "laeuft inzwischen auf echten Groessen (128/1024). 0.7 s."),
    "smoke_test_export_2048.py": (WAECHTER,
        "STILLER_RUECKFALL-Variante: verhindert, dass kategorische Layer "
        "(Biome) beim Export lautlos interpoliert werden und dadurch "
        "erfundene Zwischen-Biome entstehen, die wie ein plausibles "
        "Ergebnis aussehen. 1.5 s."),
    "smoke_test_regionen_naht_schnell.py": (WAECHTER,
        "GELAENDEVERSTIMMUNG, eigens fuer dieses Ticket gebaut als "
        "schneller Ersatz fuer die Nahtfrage aus "
        "smoke_test_regionen_welt.py (siehe deren Eichung-Eintrag unten "
        "fuer die Entscheidung) - reproduziert dessen Nahtpruefung "
        "(Schritt 3) wortgleich auf rohem weltfeld() bei 256px/1 Seed. "
        "Im Bau nachgewiesen: findet dieselbe reale, aktuell bestehende "
        "Naht-Regression wie die volle Datei (1.55 gegen 1.62 "
        "Ueberschreitung). 13.7 s."),
    "smoke_test_kontinentform.py": (WAECHTER,
        "GELAENDEVERSTIMMUNG: Formregler-Wirkung auf core/terrain_weltkarte.py "
        "direkt (Flaeche bleibt konstant, Kontinent bleibt zusammenhaengend, "
        "die Regler-Enden wirken sichtbar) - fing laut Bewertung bereits "
        "einen realen Kalibrierfehler beim eigenen Bau. 8.1 s."),
    "smoke_test_terrain_scale_coupling.py": (WAECHTER,
        "GELAENDEVERSTIMMUNG: der Gully-Groessenregler war laut Bewertung "
        "BEREITS EINMAL skalenentkoppelt (eine Invariante, die schon "
        "einmal gebrochen war) und ist jetzt gruen - genau der Fall, bei "
        "dem ein Ruecktritt der Reparatur unbemerkt bliebe, liefe der Test "
        "nicht bei jeder Aenderung. 15.2 s."),
    "smoke_test_kuestengebiete.py": (WAECHTER,
        "GELAENDEVERSTIMMUNG: prueft core/terrain_weltkarte.py direkt an "
        "der Stelle (_kuesten_umformen()), die den dokumentierten "
        "Samarcia/Morobora-Nahtfehler ausgeloest hat - Wasserlinie und "
        "Regionsmittel muessen stehen bleiben, Gebietsflaeche im Zielfenster. "
        "Deckt laut Bewertung 3 dokumentierte historische Bugs ab. 6.4 s."),
    "smoke_test_parameter_eindeutig.py": (WAECHTER,
        "Guenstiger Konsistenzwaechter: kein Parameterschluessel darf "
        "unangemeldet auf zwei Reiter zeigen - eine Doppelbelegung waere "
        "ein stiller Konfigurationsfehler, der sich erst als falscher "
        "Reglerwert zeigt. 0.1 s."),
    "smoke_test_tab_reihenfolge_kanonisch.py": (WAECHTER,
        "Guenstiger Konsistenzwaechter: GENERATOR_TAB_ORDER hat genau eine "
        "Definitionsstelle - trifft ein wiederkehrendes Muster (verstreute "
        "Doppeldefinitionen), das sich sonst als schwer nachvollziehbare "
        "Reiter-Reihenfolge zeigt. 0.3 s."),
    "smoke_test_reiter_vertrag.py": (WAECHTER,
        "Architektur-Vertragswaechter: alle 12 Reiterklassen muessen "
        "Viewport/Parameter/Statistik-Vertrag einhalten - eine Verletzung "
        "waere in jedem einzelnen betroffenen Reiter ein stiller "
        "Funktionsausfall statt eines Absturzes. 3.1 s."),
    "smoke_test_slope_compass_color.py": (WAECHTER,
        "Schnell, deterministisch, headless-faehig, prueft reale "
        "Funktionslogik (Hangrichtung -> Farbton) - laut Bewertung "
        "ausdruecklich als fehlend in der automatischen Suite "
        "bemaengelt ('sollte aufgenommen werden'); mit diesem Ticket "
        "aufgenommen. 1.0 s."),
    "smoke_test_water_pipeline_order.py": (WAECHTER,
        "STILLER_RUECKFALL: laut Bewertung woertlich die 'wertvollste "
        "Datei der ganzen Sichtung' - 12 strukturelle Garantien gegen den "
        "ECHTEN Scheduler (Reihenfolge im Graph, Deadlock-Regression, "
        "Feedback-Schleifen). Ein Verstoss haette sich sonst als "
        "plausibel aussehendes, aber falsch sortiertes Pipeline-Ergebnis "
        "gezeigt. Mit gemessenen 2.2 s guenstig genug fuers "
        "Waechterbudget - urspruenglich faelschlich als 'zu teuer' "
        "eingeschaetzt und der Eichung zugedacht, nach Messung "
        "korrigiert."),

    # ---------------------------------------------------------------
    # EICHUNG - alle uebrigen Dateien.
    # ---------------------------------------------------------------

    "smoke_test_adaptive_mesh_vectorized.py": (EICHUNG,
        "Performance-Refactor-Aequivalenz (vektorisiert vs. schleifenbasiert), "
        "kein Bezug zu einer der drei Waechter-Fehlerklassen, 7.5 s Kosten "
        "ohne entsprechenden Zusatznutzen fuer jeden Commit."),
    "smoke_test_archetyp_verteilung.py": (EICHUNG,
        "Verteilungskalibrierung ueber 512px + 16 zusaetzliche Karten "
        "(Saatstationen- vs. Kuestenlaengenanteil je Archetyp) - vorbildlich "
        "im Testdesign (Zwei-Enden-Vergleich nach der 2026-08-24-Lehre), "
        "aber ein Kalibrierungsergebnis mit Zielband, kein Every-Commit-Fall, "
        "und zu teuer fuers Waechterbudget."),
    "smoke_test_biome_supersampling.py": (EICHUNG,
        "Trifft zwar konkret die 2026-08-10-Bug-Eigenschaft "
        "(Teilpixel-Zufall abhaengig von x+y), ist aber schmal genug "
        "(ein einzelner Supersampling-Mechanismus), um im nightly-Lauf "
        "ausreichend abgesichert zu sein; Waechterbudget wird durch die "
        "drei benannten Fehlerklassen bereits ausgefuellt."),
    "smoke_test_camera_controls.py": (EICHUNG,
        "Prueft reale 3D-Kamera-Vektorrechnung gegen die View-Matrix, aber "
        "die Bewertung nennt keinen konkreten historischen Fehlerfund - "
        "ohne nennbaren realen Bug bleibt sie in der Eichung statt das "
        "Waechterbudget zu belegen."),
    "smoke_test_display_2d.py": (EICHUNG,
        "Prueft, dass alle 30 2D-Anzeigefunktionen ueberhaupt etwas "
        "zeichnen - wichtig, aber durch smoke_test_layer_2d_3d_parity.py "
        "und smoke_test_push_overlays.py im Waechter bereits fuer die "
        "STEHENDE REGEL abgedeckt; diese Datei preuft eine breitere, "
        "teurere Flaeche (30 Funktionen) fuer denselben Fehlertyp."),
    "smoke_test_erosion_field.py": (EICHUNG,
        "Massenbilanz/Determinismus des Erosionskerns - technische "
        "Garantie, aber durch smoke_test_water_pipe_flow.py (Waechter) "
        "bereits fuer dieselbe Fehlerklasse (STILLER_RUECKFALL bei "
        "Massenbilanz) abgedeckt."),
    "smoke_test_erosion_quality.py": (EICHUNG,
        "Haelt 5 Bildeigenschaften des Erosionsmodells gegen feste "
        "Zielbaender - klassische Kalibrierung, aktuell in 3 von 3 "
        "Gruppen ROT (siehe docs/TESTBESTAND_BEWERTUNG.md, 'Neue Befunde' "
        "#1) und mit 47.5 s zu teuer fuers Waechterbudget."),
    "smoke_test_erosion_realismus.py": (EICHUNG,
        "Haelt das Slope-Area-Gesetz gegen einen Zielkorridor aus "
        "docs/SPEZIFIKATION.md - Kalibrierung, aktuell ROT (siehe "
        "'Neue Befunde' #2) und mit 102.8 s die teuerste Einzeldatei "
        "der ganzen Suite."),
    "smoke_test_erosionsfilter_baender.py": (EICHUNG,
        "Bandweise = einteilige Auswertung bitgleich bei 256/512/1024px - "
        "schuetzt die Regionseichung, ist aber mit gemessenen 49.0 s "
        "deutlich zu teuer fuers Waechterbudget; die GELAENDEVERSTIMMUNG-"
        "Klasse ist im Waechter bereits durch "
        "smoke_test_regionen_naht_schnell.py, smoke_test_kontinentform.py "
        "und smoke_test_kuestengebiete.py abgedeckt."),
    "smoke_test_fluss_overlay.py": (EICHUNG,
        "Laut Bewertung strukturell schwach: der 3D-Verdrahtungsteil "
        "prueft nur per hasattr()/Quelltext-Stringsuche, ohne die Funktion "
        "wirklich aufzurufen - strukturell dieselbe Schwaeche wie die in "
        "CLAUDE.md dokumentierte hasattr-Falle, nur eine Ebene tiefer "
        "versteckt. Als Waechter fuer FEHLENDE_ANZEIGE ungeeignet, bis das "
        "behoben ist (eigener Uebersichtspunkt, nicht Teil dieses Tickets)."),
    "smoke_test_geology_3dstack.py": (EICHUNG,
        "Prueft Hoehenversatz zwischen Anzeigemodi ueber 9 Kombinationen - "
        "inhaltlich FEHLENDE_ANZEIGE-nah, aber die Klasse ist im Waechter "
        "bereits durch fuenf andere Dateien abgedeckt; Kosten/Nutzen "
        "spricht hier fuer die Eichung."),
    "smoke_test_geology_speed.py": (EICHUNG,
        "Sichert ein Performance-Refactor-Ergebnis bitgleich ab (11.2 s), "
        "kein Bezug zu einer der drei Waechter-Fehlerklassen."),
    "smoke_test_kuesten_mesh.py": (EICHUNG,
        "Testet laut Bewertung ein GEPARKTES PROTOTYP-MODUL "
        "(gui/widgets/kuesten_mesh.py), das produktiv seit "
        "KUESTEN_SCHNITT_AKTIV nicht mehr genutzt wird "
        "(dead-code-Befund) - fuer den Waechter ungeeignet, weil ein "
        "Fehlschlag hier keinen produktiven Pfad betrifft."),
    "smoke_test_kuesten_naht_kruemmung.py": (EICHUNG,
        "Bekannter offener Kalibrierfehler an core/vektor_kueste.py "
        "(Hoehenmischungs-Realismus) - haelt Werte gegen ein Zielband, "
        "gehoert damit zur Eichung; bleibt bewusst rot dokumentiert."),
    "smoke_test_kuesten_schnitt.py": (EICHUNG,
        "Regressionswaechter fuer den 2026-08-22-Wicklungsbug im "
        "Konturschnitt-Mesh - wertvoll, aber Mesh-spezifisch statt einer "
        "der drei benannten Klassen; bleibt zugunsten des Budgets in der "
        "Eichung."),
    "smoke_test_kuestenprofiltreue.py": (EICHUNG,
        "Haelt Hoehenprofile der 27 Kuesten-Archetypen gegen reale "
        "Vorbild-Tabellen - klassische Kalibrierung gegen externe "
        "Referenzdaten."),
    "smoke_test_landmarks.py": (EICHUNG,
        "Blackbox-Pruefung von Eignungsfeld/Platzierung, kein Bezug zu "
        "einer der drei Waechter-Fehlerklassen."),
    "smoke_test_noise_offset_gpu.py": (EICHUNG,
        "Belegt GPU/CPU-Paritaet bei Rauschversatz real - inhaltlich "
        "STILLER_RUECKFALL-nah, aber die Klasse ist im Waechter bereits "
        "durch smoke_test_erosion_gpu_contract.py und "
        "smoke_test_erosion_gpu_parity.py abgedeckt."),
    "smoke_test_pipeline_outputs.py": (EICHUNG,
        "Prueft 78 (Knoten,Output)-Paare durch die VOLLE 39-Knoten-Pipeline "
        "(128px, LOD3) - die umfassendste Paritaetspruefung im Bestand, "
        "aber entsprechend teuer, und aktuell mit bekannten, in "
        "docs/TESTBERICHT.md dokumentierten Dispensationen rot; als "
        "Every-Commit-Gate ungeeignet, gehoert in die naechtliche "
        "Vollpruefung."),
    "smoke_test_poisson_punkte.py": (EICHUNG,
        "Determinismus-/Performance-Regressionstest, kein Bezug zu einer "
        "der drei Waechter-Fehlerklassen."),
    "smoke_test_region_tab.py": (EICHUNG,
        "End-zu-End-GUI-Vertragstest (Dropdown, Vorschau, Regler, "
        "Zuruecksetzen, 3D-Ansicht) - breit und wertvoll, aber 8.9 s ohne "
        "konkreten Bezug zu einer der drei benannten Klassen; gehoert in "
        "die naechtliche Vollpruefung."),
    "smoke_test_region_vorbild_aehnlichkeit.py": (EICHUNG,
        "Kalibrierung der Regionsformen gegen echte COP30-DEM-Ausschnitte "
        "(Formfaktor, KS-Test, Fraktaldimension) - 16.4 s, klassische "
        "Eichung gegen externe Referenzdaten."),
    "smoke_test_regionen_fairness.py": (EICHUNG,
        "Haelt den Bewohnbare-Flaeche-Index je Region gegen ein Ziel "
        "(+/-25%) - Kalibrierung, gehoert in die Eichung."),
    "smoke_test_regionen_welt.py": (EICHUNG,
        "ENTSCHEIDUNG (Ticket-Pflichtpunkt): bleibt in der Eichung. "
        "Gemessen in dieser Sitzung 80.8 s (Ticket #45 hatte an einem "
        "anderen Tag 46.4 s gemessen - die Maschine schwankt laut "
        "tools/testlauf.py um Faktor 2-3, siehe SITZUNGSLOG). Allein "
        "diese Datei wuerde das 2-Minuten-Waechterbudget entweder "
        "sprengen oder bei einer einzigen langsamen Ausfuehrung darueber "
        "schieben. Sie ist trotzdem der empfindlichste Waechter fuer "
        "Gelaendeform (neun Regionen gegen feste Hang-/Wasserziele, volle "
        "BaseTerrainGenerator-Pipeline, fuenf Seeds, "
        "Aufloesungsunabhaengigkeit, GPU/CPU-Paritaet) und bleibt deshalb "
        "vollstaendig erhalten - nur eben nachts. ERSATZ fuer die "
        "GELAENDEVERSTIMMUNG-Frage im Waechter: "
        "smoke_test_regionen_naht_schnell.py (neu gebaut, siehe dort), "
        "beantwortet dieselbe Nahtfrage in 13.7 s statt 80.8 s, deckt "
        "aber NICHT die absoluten Hang-/Wasserziele und die "
        "Aufloesungsunabhaengigkeit ab - dafuer bleibt diese Datei "
        "zustaendig."),
    "smoke_test_regionsfeld.py": (EICHUNG,
        "Mathematische Konsistenzpruefung der Vorschau-Formel, kein "
        "konkreter Bezug zu einer der drei Waechter-Fehlerklassen."),
    "smoke_test_regionsregler_wirken.py": (EICHUNG,
        "Einziger voller Durchgriffstest Regionsregler->fertige Karte - "
        "wertvoll, aber allgemein statt an einer der drei Klassen "
        "orientiert; gehoert in die naechtliche Vollpruefung."),
    "smoke_test_river_reaches_sea.py": (EICHUNG,
        "Robuste Blackbox-Invariante (jeder Flussknoten erreicht das "
        "Meer), kein konkreter Bezug zu einer der drei Waechter-"
        "Fehlerklassen."),
    "smoke_test_seegliederung.py": (EICHUNG,
        "Haelt Seegrad, Zieltiefe, Uferregion und Seeeis-Quote gegen feste "
        "Werte - Kalibrierungscharakter."),
    "smoke_test_seenflaeche_messung.py": (EICHUNG,
        "Reine MESSUNG ohne Zielband (Ticket #32: druckt Zahlen zur "
        "manuellen Uebernahme in Ticket/docs/TESTBERICHT.md aus, keine "
        "Zusicherung) - kein wiederholbarer Bandwaechter und damit per "
        "Definition kein Waechter-Kandidat."),
    "smoke_test_settlement_clustering.py": (EICHUNG,
        "Eigener statistischer Einzelaspekt (Naechste-Nachbar-Abstand je "
        "Kultur), kein Bezug zu einer der drei Waechter-Fehlerklassen."),
    "smoke_test_settlement_placement.py": (EICHUNG,
        "Eignungsfaktoren-Trennschaerfe, kein Bezug zu einer der drei "
        "Waechter-Fehlerklassen."),
    "smoke_test_settlement_region_grid.py": (EICHUNG,
        "Roadsite-/Landmark-Verteilung ueber 9 Regionen, traegt laut "
        "Bewertung noch einen veralteten Vergleichswert im Ausgabetext "
        "(kleine Aufraeumarbeit, nicht Teil dieses Tickets)."),
    "smoke_test_settlement_roads.py": (EICHUNG,
        "Praezise Regressionspruefung exakter Werte (Kostenfeld, "
        "Gabriel-Graph), kein Bezug zu einer der drei Waechter-"
        "Fehlerklassen."),
    "smoke_test_settlement_sites.py": (EICHUNG,
        "Deckt einen historischen Hoehen-Deckel-Bug ab, ist aber "
        "spezialisiert genug (Seeweg-Pathfinding, Katalog-Kultur-"
        "Konsistenz), um im nightly-Lauf ausreichend abgesichert zu "
        "sein."),
    "smoke_test_wege_wasser_kosten.py": (EICHUNG,
        "Praezise Regressionspruefung exakter Werte (Furt-/Bruecken-"
        "kosten, Uferweg-Rabatt im Kostenfeld aus #42), kein Bezug zu "
        "einer der drei Waechter-Fehlerklassen - gleiches Muster wie "
        "smoke_test_settlement_roads.py. Nachtrag bei der #46-Pruefung "
        "eingestuft, weil das Worktree des #46-Agenten vor dieser "
        "(#42-)Datei von main abzweigte. 1.1 s."),
    "smoke_test_settlement_valley_routing.py": (EICHUNG,
        "A*-Routenqualitaet, traegt laut Bewertung noch eine ausgedachte "
        "Testgroesse (100px statt eines 32er-Vielfachen) - kleine "
        "Aufraeumarbeit, nicht Teil dieses Tickets."),
    "smoke_test_spielkarten.py": (EICHUNG,
        "Kartenzerlegung gegen woertlich zitierte Nutzervorgaben, kein "
        "Bezug zu einer der drei Waechter-Fehlerklassen."),
    "smoke_test_stadttypen.py": (EICHUNG,
        "Entscheidungsregeln (Marktstadt-Wahl, Handelsnetz), kein Bezug "
        "zu einer der drei Waechter-Fehlerklassen."),
    "smoke_test_terrain_erosion_filter.py": (EICHUNG,
        "Testet laut Bewertung den ALTEN Pfad (WELTKARTE_AKTIV=False) - "
        "kein produktiver Codepfad mehr, aktueller Fehlschlag ist reine "
        "Worktree/Hauptcheckout-Pfaddivergenz (Umgebungsartefakt, siehe "
        "docs/TESTBESTAND_BEWERTUNG.md); als Waechter fuer einen toten "
        "Pfad ungeeignet."),
    "smoke_test_terrain_remesh.py": (EICHUNG,
        "Setzt die 2026-08-12-Lehre (adaptives Mesh auf echten "
        "Kartengroessen) um, ist aber Mesh-spezifisch und die "
        "STILLER_RUECKFALL-Klasse ist im Waechter bereits durch "
        "smoke_test_adaptive_terrain_mesh.py (derselbe historische Fall) "
        "abgedeckt."),
    "smoke_test_terrain_river_network.py": (EICHUNG,
        "Testet laut Bewertung denselben ALTEN Pfad wie "
        "smoke_test_terrain_erosion_filter.py, gleicher Grund fuer den "
        "aktuellen Fehlschlag (Umgebungsartefakt), kein produktiver "
        "Codepfad."),
    "smoke_test_vektor_kueste.py": (EICHUNG,
        "Laut Bewertung ausdruecklich als MOMENTAUFNAHME einer parallel "
        "laufenden Sitzung zu verstehen, nicht als dauerhafte Bestaetigung "
        "- fuer einen taeglichen Waechter ungeeignet."),
    "smoke_test_wasserbilanz_toleranz.py": (EICHUNG,
        "Kalibrierungstoleranz (Niederschlag = Wasser+Abfluss+Verdunstung "
        "+/-10%), aktuell bekannt +34.62% abweichend mit befristeter "
        "Dispensation (Ticket #70) - klassischer Eichungsfall."),
    "smoke_test_water_drainage_erosion.py": (EICHUNG,
        "Laut Bewertung der schwaechste von drei sich ueberschneidenden "
        "DropletErosionSystem-Tests (Duplikat-Cluster mit "
        "water_edge_sediment/water_erosion_quality) - Konsolidierungs-"
        "Kandidat, kein eigenstaendiger Waechter-Bedarf."),
    "smoke_test_water_edge_sediment.py": (EICHUNG,
        "14.1 s, Teil desselben Duplikat-Clusters wie "
        "water_drainage_erosion/water_erosion_quality; die Massenbilanz-"
        "Klasse ist im Waechter bereits durch smoke_test_water_pipe_flow.py "
        "abgedeckt."),
    "smoke_test_water_erosion_quality.py": (EICHUNG,
        "7.4 s, umfassendster der drei DropletErosionSystem-Tests, aber "
        "Kalibrierungscharakter (Kanalnetz-Zusammenhang, Sedimentflaechen "
        "gegen Zielband) - gehoert zur selben Gruppe wie "
        "erosion_quality/erosion_realismus in der Eichung."),
    "smoke_test_water_lake_detection_gpu.py": (EICHUNG,
        "Laut Bewertung explizit ZU SCHWACHE Zusicherung (prueft nur "
        "'nicht komplett leer', nicht 'ungefaehr gleiche Flaeche/Anzahl' "
        "trotz gemessener CPU/GPU-Abweichung 6 vs. 3 bzw. 4 vs. 2 Seen) - "
        "fuer den Waechter erst geeignet, nachdem die Zusicherung "
        "verschaerft wurde (eigener Uebersichtspunkt, nicht Teil dieses "
        "Tickets)."),
    "smoke_test_water_pipeline_full.py": (EICHUNG,
        "Prueft LOD-Uebergabe ueber den echten DataLODManager-Pfad, "
        "traegt laut Bewertung nur einen veralteten Docstring "
        "(Doku-Korrektur, nicht Teil dieses Tickets) - kein konkreter "
        "Bezug zu einer der drei Waechter-Fehlerklassen."),
    "smoke_test_water_thermal_erosion.py": (EICHUNG,
        "Boeschungswinkel-Mechanik mit eigener Massenbilanz - "
        "eigenstaendiger Mechanismus, aber dieselbe Fehlerklasse "
        "(Massenbilanz-STILLER_RUECKFALL) ist im Waechter bereits durch "
        "smoke_test_water_pipe_flow.py abgedeckt."),
    "smoke_test_weather_climatology.py": (EICHUNG,
        "17 Klimatologie-Pruefungen an historische Bugreports gebunden, "
        "aber mit festen Zielwerten (Kalibrierungscharakter) und 14.3 s "
        "Kosten bei ueberwiegend ausgedachter Testgroesse 48px."),
    "smoke_test_weather_temperature_direktnormierung.py": (EICHUNG,
        "Haelt Temperatur-Jahresmittel gegen KLIMA_ZIEL fuer 9 Regionen - "
        "klassische Kalibrierung, aktuell 3 von 36 Kombinationen ausserhalb "
        "der 0.7-K-Toleranz."),
    "smoke_test_weather_wind_regions.py": (EICHUNG,
        "Haelt Windmittel gegen wind_ziel_map - klassische Kalibrierung."),
    "smoke_test_wegsuche_schnell.py": (EICHUNG,
        "Numba-JIT-A* punktgenau identisch zur Python-Referenz - strenge, "
        "aber technische Zusicherung ohne Bezug zu einer der drei "
        "Waechter-Fehlerklassen."),
    "smoke_test_weltfluesse_vektor.py": (EICHUNG,
        "Vektorisierte Talschnitzschleife bitgleich zur Schleife - "
        "Performance-Refactor-Absicherung, kein Bezug zu einer der drei "
        "Waechter-Fehlerklassen."),
}


def rang(datei):
    """Rang einer einzelnen Testdatei. Wirft RangUnbekannt statt None
    zurueckzugeben - ein unbekannter Rang ist ein Fehler in dieser Liste,
    kein Grund, die Datei stillschweigend zu ueberspringen."""
    eintrag = EINSTUFUNG.get(datei)
    if eintrag is None:
        raise RangUnbekannt(
            "%s ist keinem Rang zugeordnet - in tools/test_raenge.py "
            "EINSTUFUNG eintragen (waechter oder eichung, mit "
            "Begruendung)." % datei)
    return eintrag[0]


def begruendung(datei):
    eintrag = EINSTUFUNG.get(datei)
    if eintrag is None:
        raise RangUnbekannt(
            "%s ist keinem Rang zugeordnet." % datei)
    return eintrag[1]


def dateien_im_rang(alle_dateien, gewuenschter_rang):
    """Filtert eine Dateiliste auf einen Rang. Prueft dabei NEBENBEI die
    Vollstaendigkeit der Einstufung fuer alle uebergebenen Dateien - wer
    filtert, bekommt die Luecke sofort gemeldet statt spaeter eine Datei
    zu vermissen."""
    if gewuenschter_rang not in RAENGE:
        raise ValueError("unbekannter Rang %r, erwartet einer von %r"
                         % (gewuenschter_rang, RAENGE))
    fehlend = [d for d in alle_dateien if d not in EINSTUFUNG]
    if fehlend:
        raise RangUnbekannt(
            "%d Testdatei(en) ohne Einstufung in tools/test_raenge.py: %s"
            % (len(fehlend), ", ".join(sorted(fehlend))))
    return [d for d in alle_dateien if EINSTUFUNG[d][0] == gewuenschter_rang]


if __name__ == "__main__":
    # Kleine Selbstpruefung ohne Testlauf: zaehlt beide Raenge und meldet,
    # falls sich EINSTUFUNG und der tatsaechliche Dateibestand auseinander
    # gelebt haben (in beide Richtungen).
    import os
    import sys

    wurzel = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests = os.path.join(wurzel, "tests")
    echte_dateien = set(f for f in os.listdir(tests)
                        if f.startswith("smoke_test_") and f.endswith(".py"))
    eingestufte_dateien = set(EINSTUFUNG)

    fehlend = sorted(echte_dateien - eingestufte_dateien)
    verwaist = sorted(eingestufte_dateien - echte_dateien)
    # Nur ueber bekannte Dateien zaehlen: rang() wirft RangUnbekannt fuer
    # jede Datei aus 'fehlend' - die Meldung dazu soll unten als Text
    # erscheinen, nicht als Traceback vor dem ersten print().
    bekannt = echte_dateien - set(fehlend)
    waechter = sorted(d for d in bekannt if rang(d) == WAECHTER)
    eichung = sorted(d for d in bekannt if rang(d) == EICHUNG)

    print("%d Testdateien insgesamt, %d Waechter, %d Eichung"
          % (len(echte_dateien), len(waechter), len(eichung)))
    if fehlend:
        print("FEHLT in EINSTUFUNG (%d): %s"
              % (len(fehlend), ", ".join(fehlend)))
    if verwaist:
        print("VERWAIST in EINSTUFUNG, Datei existiert nicht mehr (%d): %s"
              % (len(verwaist), ", ".join(verwaist)))
    sys.exit(1 if (fehlend or verwaist) else 0)
