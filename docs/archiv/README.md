# Archiv

Hier liegt, was **historischen Wert hat, aber nicht mehr gilt**. Nichts in
diesem Ordner beschreibt den heutigen Zustand des Programms. Wer wissen will,
wie etwas heute funktioniert, liest den Code oder `docs/OFFENE_PUNKTE.md`.

Abgelegt am 2026-08-12; erweitert am 2026-09-21 (Ticket #51) und am
2026-09-23 (Ticket #44, Zusammenführung der Spezifikation zum Baum
`docs/spezifikation/`).

| Datei | Was es war | Warum es hier liegt |
|---|---|---|
| `session_handover_2026-07-08.md` | Übergabe der Terrain-/Geologie-Sitzung | Die beschriebenen Aufgaben sind erledigt; die Übergabe ist abgeschlossen. |
| `session_handover_2026-07-22_terrain_review.md` | Übergabe nach dem Terrain-Review | dito |
| `session_review_2026-07-22_geology.md` | Prüfbericht Geologie | Befunde sind in `OFFENE_PUNKTE.md` aufgegangen. |
| `session_design_2026-07-22_geology_3dstack_concept.md` | Entwurf des 3D-Schichtstapels | Umgesetzt; der Code ist die Wahrheit. |
| `tickets.xlsx` | Tabelle aus dem alten Kanban-Board | Ersetzt durch `OFFENE_PUNKTE.md`. |
| `descriptor_STAND_2025-09.py` | Beschreibung aller Skripte und Methoden | **Stand 2025-09-02, elf Monate alt.** Kennt weder `terrain_weltkarte.py` noch `calculator_graph.py` noch `adaptive_terrain_mesh.py` — beschreibt also eine Architektur, die es nicht mehr gibt. Wurde nirgends importiert. Lag bis zum 2026-08-12 mit 281 KB in der Projektwurzel und war dort aktiv irreführend. |
| `2026-08-12_UEBERGABE.md` | Übergabe/Gesamtstand-Dokument | Stand 2026-08-12. Abgelöst durch die beiden neueren Sitzungsübergaben und danach `docs/SITZUNGSLOG.md`. |
| `2026-08-16_UEBERGABE_SITZUNG.md` | Sitzungsübergabe (Remesh) | Stand 2026-08-16. Sitzung liegt abgeschlossen in der Vergangenheit; Folgearbeit steht in `OFFENE_PUNKTE.md` 6.30–6.33. |
| `2026-08-24_UEBERGABE_SITZUNG.md` | Sitzungsübergabe (Küsten-/Fluss-Fehler) | Stand 2026-08-24. Die drei beschriebenen Fehler sind laut `docs/SITZUNGSLOG.md` behoben; das Dokument war ohnehin als Einstiegspunkt für genau eine Folgesitzung gedacht. |
| `2026-07-29_SPEZIFIKATION.md` | Oberziel, Komponentenziele, Invarianten | Angelegt 2026-07-29. `docs/SOLLBESCHREIBUNG.md` ersetzt §1/§2 davon, sobald sie fertig ist (siehe deren §Einleitung); bis dahin bleibt sie hier als Referenz nachlesbar, ist aber nicht mehr die tägliche Arbeitsgrundlage. |
| `2026-08-25_AUFRAEUMPLAN.md` | Plan: Programmaufbau und die nächsten acht Ziele | Angelegt 2026-08-25. Was daraus erledigt ist, steht in `docs/SITZUNGSLOG.md`; was offen ist, in `docs/OFFENE_PUNKTE.md` — der Plan selbst ist damit ein Zwischenstand, keine laufende Quelle mehr. |
| `2026-08-04_INTEGRATIONSPLAN.md` | Plan: Regionenwelt im Labor (zweite Fassung) | Stand 2026-08-04. Trägt seit 2026-08-12 selbst den Vermerk "UMGESETZT — Planungsarchiv, keine Vorgabe mehr". |
| `2026-08-27_PRUEFLISTE_LIVE.md` | Prüfliste für alles, was headless nicht geht | Stand 2026-08-27. Bezog sich auf einen bestimmten Entwicklungsstand der GUI; einzelne Punkte sind seither erledigt oder überholt, ohne dass das im Dokument nachgeführt wurde. |
| `2026-08-24_FLUESSE_UND_WASSER.md` | Plan: Flüsse und Wasserverteilung, Blöcke 1–5 | Stand 2026-08-24. Die dort geplanten Blöcke sind laut `docs/SITZUNGSLOG.md` abgearbeitet. |
| `2026-08-24_ANZEIGE_UND_SEEN.md` | Plan: Anzeige im 3D und Binnenseen, Blöcke A–C | Stand 2026-08-24. Ebenfalls laut `docs/SITZUNGSLOG.md` umgesetzt. |
| `2026-09-16_SOLLBESCHREIBUNG.md` | Gerüst: Zweck, Oberziel, Qualitätsmaßstab, Rohstoffe, Eignungsfelder | Stand 2026-09-16. **Aufgegangen in `docs/spezifikation/01_ZIEL.md`** (Ticket #44). Die Sollbeschreibung war das Gerüst, aus dem der Baum gewachsen ist; sie selbst ist damit erledigt. |
| `2026-09-01_KUESTENMODELL.md` | Funktionsweise der Küste: Profil, Reichweite, Archetypen | Stand 2026-08-18, in Teilen schon am 2026-08-24 überholt (eigener Warnkasten). **Aufgegangen in `docs/spezifikation/11_GELAENDE.md`** (Ticket #44): alte §3→Abschnitt 5, §4→6, §5→7, §7→9, §8→10. |
| `2026-09-01_KLIMA_UND_SEE.md` | Klimavorgaben und Seegliederung (Seegrad-Voronoi, Tiefentabelle) | Stand 2026-08-07, umgesetzt laut eigenem Vermerk 2026-08-12. **Auf zwei Dateien aufgeteilt** (Ticket #44): die Seegliederung (alte §2) steht in `docs/spezifikation/12_WASSER.md` Abschnitt 7, der Klimateil in `docs/spezifikation/13_KLIMA_UND_BIOME.md` (alte §0→Abschnitt 1, §1→2, §4→4, Sonnenstand→5). |
| `2026-09-16_BIOME_MATRIX.md` | Die 15 Grundbiome, Superbiome, Vereinfachung des Wetters | Stand 2026-08-07, umgesetzt und gemessen laut eigenem Vermerk 2026-08-12. **Aufgegangen in `docs/spezifikation/13_KLIMA_UND_BIOME.md`** (Ticket #44). |
| `2026-09-22_SIEDLUNGEN_ENTWURF.md` | Entwurf: Ortsgrößen, Eignungsfeld, Wegenetz | Stand 2026-08-06, mit dem Nutzer abgestimmt. **Aufgegangen in `docs/spezifikation/14_SIEDLUNGEN.md`** (Ticket #44): alte §2→2, §4→5 (mit §4.1–4.7→5.1–5.7), §6.1→7. |
| `2026-09-01_KULTUREN_UND_ORTE.md` | Auswahllisten: Kulturen, Landmarken, Roadsites (Zeitschnitt 932 n. Chr.) | Stand 2026-08-06. **Aufgegangen in `docs/spezifikation/14_SIEDLUNGEN.md`** (Ticket #44); die Kataloge selbst stehen als Daten im Code, nicht mehr in der Spezifikation. |
| `2026-09-21_SPEC_OVERLAYS.md` | Spezifikation: Overlays bekommen eine Naht (`BaseMapTab._push_overlays`) | Angelegt 2026-09-14. **Aufgegangen in `docs/spezifikation/15_ANZEIGE.md`** (Ticket #44). Die Naht ist gebaut und wird von `tests/smoke_test_push_overlays.py` bewacht. |

## Was NICHT hier liegt, obwohl es alt aussieht

Zwei Dateien in `docs/` sind ebenfalls veraltet, bleiben aber draußen, weil
Teile davon weitergelten — sie tragen jeweils oben einen Warnkasten — UND
weil sie aus echtem Programmcode heraus referenziert werden
(`core/*.py`, `managers/*.py`), nicht nur aus anderen Dokumenten:

* **`backlog.md`** — Archiv, aber drei GPU-Punkte daraus sind noch offen und
  nach `OFFENE_PUNKTE.md` 7.9–7.11 übernommen.
* **`generation_pipeline_dependencies.md`** — Knotenzahl und Orchestrator-
  Beschreibung sind überholt, die Datenflüsse selbst stimmen weiterhin.

Ebenfalls draußen, aber aus einem anderen Grund: `docs/OFFENE_PUNKTE.md` (die
einzige, aktiv geführte Aufgabenliste), `docs/SITZUNGSLOG.md` und
`docs/TESTBERICHT.md` (laufende Protokolle) sowie `docs/SOLLBESCHREIBUNG.md`
und `docs/NACHTBETRIEB.md` (Anweisung an den Agenten, keine Ablage). Zu jeder
dieser fünf Dateien wurde beim Aufräumen 2026-09-21 (Ticket #51) einzeln
entschieden, sie zu behalten — siehe der zugehörige Commit für die
Begründung je Datei.
