# Die Spezifikation des MapGenerators

**Was das hier ist.** Das Soll des Programms — was gelten *soll*, aufgeteilt auf
zehn kleine Dateien statt einer mit 1700 Zeilen. Wer eine Frage hat, liest genau
eine davon, nicht alles.

**Was das hier nicht ist.** Kein Bericht über den Ist-Zustand. Was das Programm
heute *tut*, steht in `docs/HANDBUCH.md`; was gerade offen ist, in
`docs/OFFENE_PUNKTE.md`; was in welcher Sitzung passiert ist, in
`docs/SITZUNGSLOG.md`. Diese Trennung ist der Grund, warum der Baum nicht wieder
auf 1700 Zeilen wächst.

---

## Welche Datei beantwortet welche Frage

| Frage | Datei | Zeilen |
|---|---|---:|
| Wofür ist das Programm da, und was gehört nicht hinein? | [01_ZIEL.md](01_ZIEL.md) | 216 |
| Was ist bei **jeder** Änderung zu prüfen, egal welches Thema? | [02_INVARIANTEN.md](02_INVARIANTEN.md) | 144 |
| Wie wird gearbeitet, womit gemessen, was ist ein guter Test? | [03_ARBEITSREGELN.md](03_ARBEITSREGELN.md) | 174 |
| Welchen Zielwert muss Region X erreichen? | [10_REGIONEN.md](10_REGIONEN.md) | 324 |
| Wie soll die Küste geformt sein, wie das Rauschgelände? | [11_GELAENDE.md](11_GELAENDE.md) | ~220 |
| Woher bekommt ein Fluss sein Wasser, wie tief ist die See? | [12_WASSER.md](12_WASSER.md) | ~244 |
| Welche Temperatur, welcher Niederschlag, welches Biom? | [13_KLIMA_UND_BIOME.md](13_KLIMA_UND_BIOME.md) | 281 |
| Wo liegen Orte, welche Kultur, welches Wegenetz? | [14_SIEDLUNGEN.md](14_SIEDLUNGEN.md) | ~365 |
| Was muss ein Haken in einem Reiter sichtbar machen? | [15_ANZEIGE.md](15_ANZEIGE.md) | 280 |
| Wo kommt diese Zahl von 2026-07-30 eigentlich her? | [90_MESSPROTOKOLLE.md](90_MESSPROTOKOLLE.md) | 1178 |

**Die Nummerierung** trennt drei Sorten: **0x** gilt überall (Ziel, Invarianten,
Arbeitsregeln), **1x** ist nach Sachgebiet geschnitten (Regionen, Gelände,
Wasser, Klima, Siedlungen, Anzeige), **9x** ist Anhang.

**Der Anhang wird nicht von vorn gelesen.** `90_MESSPROTOKOLLE.md` enthält
fünfzehn datierte Messreihen aus der Erosions- und Flussnetzarbeit vom
2026-07-30 bis 2026-08-04. Man springt hinein, um nachzusehen, woher eine Zahl
stammt — **nichts darin ist normativ**. Wo ein Protokoll noch bindet, ist die
Regel daraus nach 11–15 gewandert.

---

## Die vier Regeln, die den Baum klein halten

1. **Soll, nicht Ist.** Ein Satz, der mit „aktuell", „zurzeit" oder „noch nicht
   umgesetzt" beginnt, gehört nach `docs/OFFENE_PUNKTE.md` — außer er steht
   ausdrücklich unter `## Offene Fragen` am Dateiende.
2. **Keine datierten Messreihen in 01–15.** Eine Messung belegt eine
   Festlegung, ersetzt sie aber nicht. Die Zahl kommt in den Text, das Protokoll
   in den Anhang.
3. **Jeder übernommene Abschnitt trägt eine Herkunftszeile** — kursiv, direkt
   unter dem Abschnitt, mit Quelldatei, Stand und Abschnittstitel. Ohne sie ist
   nach dem nächsten Umbau nicht mehr feststellbar, ob eine Aussage beschlossen
   oder abgeschrieben war.
4. **Zeilenverweise sind ein Stand, keine Zusage.** `datei.py:828` sagt, wo es
   am 2026-09-23 stand. Bei diesem Umbau waren allein in `15_ANZEIGE.md` fünf
   Verweise um vier Zeilen verrutscht. Wer einen Verweis prüft und ihn falsch
   findet, korrigiert ihn — das ist kein Fehlerbericht wert.

---

## Wer diese Festlegungen bewacht

| Test | Prüft |
|---|---|
| `tests/smoke_test_regionen_welt.py` | die neun Regionen gegen `10_REGIONEN.md` Teil A (≈2 min) |
| `tests/smoke_test_display_methoden_existieren.py` | die 2D/3D-Regel aus `02_INVARIANTEN.md` 0.1 und `15_ANZEIGE.md` |
| `tests/smoke_test_push_overlays.py` | die Overlay-Naht `BaseMapTab._push_overlays` |
| `tests/smoke_test_display_2d.py` | alle 30 Darstellungen der 2D-Anzeige |

---

## Woher das kommt

Zusammengeführt am **2026-09-23** aus neun Dokumenten, die dieselbe Sache an
verschiedenen Stellen und teils widersprüchlich beschrieben hatten:

`docs/archiv/2026-07-29_SPEZIFIKATION.md` (1626 Zeilen, davon 1153
Messprotokolle) · `docs/archiv/2026-09-16_SOLLBESCHREIBUNG.md` · `docs/archiv/2026-09-01_KUESTENMODELL.md` ·
`docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md` · `docs/archiv/2026-09-01_KLIMA_UND_SEE.md` ·
`docs/archiv/2026-09-16_BIOME_MATRIX.md` · `docs/archiv/2026-09-22_SIEDLUNGEN_ENTWURF.md` ·
`docs/archiv/2026-09-01_KULTUREN_UND_ORTE.md` · `docs/archiv/2026-09-21_SPEC_OVERLAYS.md`.

**Alle neun liegen seit dem 2026-09-23 in `docs/archiv/`**, mit Datumspräfix und
dem Stand ihres letzten Commits — gelöscht wurde nichts. `docs/archiv/README.md`
sagt ausdrücklich, dass dort nichts mehr gilt: wer eine Herkunftszeile
nachschlägt, findet den Wortlaut, arbeitet aber im Baum weiter.

Widersprüche zwischen den Quellen wurden **nicht** stillschweigend aufgelöst:
Wo zwei Dokumente verschiedene Zahlen nannten, steht im Fließtext der jüngere
Stand und unter `## Offene Fragen` beide Werte mit Fundstelle. Der größte
aufgelöste Widerspruch steht in `01_ZIEL.md` §2 — das Oberziel von 2026-07-29
verlangte eine Welt „ohne Meer", während heute sechs der neun Regionen Wasser
haben und Thalassia bei 65 % liegt.
