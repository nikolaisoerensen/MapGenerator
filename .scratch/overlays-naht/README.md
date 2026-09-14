# Overlays bekommen eine Naht — neun Tickets

Aufgeteilt am 2026-09-14 aus `docs/SPEC_OVERLAYS.md`. Ein Ticket je Datei in
`issues/`, nummeriert in Abhaengigkeitsreihenfolge (Blockierer zuerst).

**Angelegt auf GitHub am 2026-09-14** als Issues #4 bis #12 in
`nikolaisoerensen/MapGenerator`, mit Label und **echten
GitHub-Abhaengigkeiten** (`blocked_by`) — nicht nur Textverweisen. Die
Dateien hier sind die Quelle, die Issues sind die Arbeitsansicht.

**Arbeitsweise: die Front abarbeiten** — jedes Ticket, dessen Blockierer alle
fertig sind, darf beginnen. Das ist hier kein reiner Strang: nach #8 laufen
drei Tickets parallel.

```
01/#4 Rasterfunktionen (Vorarbeit) ─┐
                                    │
02/#5 Biome-Reiter ─┐               │
                    ├→ 04/#7 Waechter ┴→ 05/#8 Naht ─┬→ 06/#9  Siedlungen ─┐
03/#6 Uebersicht  ─┘                                 ├→ 07/#10 Regional   ─┼→ 09/#12 Aufraeumen
                                                     └→ 08/#11 Fluesse    ─┘
```

| Datei | Issue | Ticket | Blockiert von |
|---|---|---|---|
| `01-…` | [#4](https://github.com/nikolaisoerensen/MapGenerator/issues/4) | Rasterfunktionen in ein eigenes Modul *(Vorarbeit, streichbar)* | — |
| `02-…` | [#5](https://github.com/nikolaisoerensen/MapGenerator/issues/5) | Biome-Reiter zeigt Siedlungen und Fluesse wieder in 3D | — |
| `03-…` | [#6](https://github.com/nikolaisoerensen/MapGenerator/issues/6) | Uebersichts-Reiter — loeschen oder bauen | — |
| `04-…` | [#7](https://github.com/nikolaisoerensen/MapGenerator/issues/7) | Waechtertest prueft Erreichbarkeit statt Namensexistenz | #5, #6 |
| `05-…` | [#8](https://github.com/nikolaisoerensen/MapGenerator/issues/8) | Die Naht, mit dem Biome-Reiter als erstem Nutzer | #7, #4 |
| `06-…` | [#9](https://github.com/nikolaisoerensen/MapGenerator/issues/9) | Siedlungs-Reiter umgestellt | #8 |
| `07-…` | [#10](https://github.com/nikolaisoerensen/MapGenerator/issues/10) | Regional-Reiter umgestellt — und der Fehlerschlucker raus | #8 |
| `08-…` | [#11](https://github.com/nikolaisoerensen/MapGenerator/issues/11) | Fluss-Reiter umgestellt | #8 |
| `09-…` | [#12](https://github.com/nikolaisoerensen/MapGenerator/issues/12) | Aufraeumen — alte Weichen weg, Namensraeume ins Register | #9, #10, #11 |

**#5 und #6 haengen an nichts und koennen sofort beginnen.** #5 behebt einen
Fehler, der heute im Arbeitsverzeichnis steht, und ist am Programm sofort
vorfuehrbar.

## Label

Am 2026-09-14 angelegt, als Standard fuer das ganze Repo gedacht, nicht nur
fuer dieses Buendel. Je Issue **genau eine Art**, **genau ein `bereich:`**,
dazu beliebig viele Marker.

* **Ablauf:** `ready-for-agent`
* **Art:** `fehler` · `umbau` · `funktion` · `test` · `doku`
* **Bereich:** `bereich:anzeige` · `bereich:gelaende` · `bereich:wasser` ·
  `bereich:wetter` · `bereich:biome` · `bereich:siedlungen` ·
  `bereich:pipeline`
* **Marker:** `sichtpruefung` (nur am laufenden Programm zu bestaetigen) ·
  `stiller-ausfall` (Ersatzpfad, der von Erfolg nicht zu unterscheiden ist) ·
  `vorarbeit`

Die beiden ersten Marker sind projektspezifisch und verdienen ihren Platz:
`sichtpruefung` hat eine eigene Datei (`docs/PRUEFLISTE_LIVE.md`), und
`stiller-ausfall` ist die Fehlerklasse, die hier nachweislich fuenfmal
durchgekommen ist — Shaderpfade nach dem Dateiumzug, adaptives Netz,
Geologie-Dispatch, `impact_matrix`, und jetzt die Overlays.

## Nicht in diesem Buendel

Die fuenf Eintraege der Schuldliste tatsaechlich in 3D bauen (Regionsgitter,
Stadtgrenzkontur, Hoehenlinien, Regionsfarben, Parzellengrenzen). Diese neun
Tickets bauen den *Mechanismus* und machen die Schuld sichtbar und laut; sie
loesen sie nicht ein. Begruendung in `docs/SPEC_OVERLAYS.md`, Abschnitt
"Out of Scope".

Ebenfalls nicht: Kandidaten 2 bis 5 des Architekturberichts vom 2026-09-14 —
die GPU/CPU-Naht, der Regionskatalog als Wert, der `DataLODManager`, die vier
Schreibweisen je Parameter. Eigene Spezifikationen.
