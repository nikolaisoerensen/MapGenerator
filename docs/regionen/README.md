# Referenzregionen

Zwanzig reale Landschaften, gegen die der Generator abgenommen wird. Sie sind
der Zielkatalog aus `docs/archiv/2026-07-29_SPEZIFIKATION.md` §2, hier je Region ausgeführt.

**Alle ohne Meer** — Küstenlinien und Ozean sind bewusst ausgeklammert.

## Aufbau je Region

```
NN_name/
  REGION.md      Zielwerte, Koordinaten, Prüfliste, erzielter Stand
  aerial/        Senkrechte Luft-/Satellitenaufnahme, 10-20 km Breite
  fluss/         Flusslauf von oben
  talform/       Querschnitt oder Blick talaufwärts
  klima/         Jahresgang Temperatur und Niederschlag
```

Jeder Unterordner enthält eine Textdatei, die sagt, **was** das Bild belegen
soll. Nicht „ein schönes Bild", sondern eine überprüfbare Eigenschaft: Dichte
des Talnetzes, Zahl der Zusammenflüsse, V gegen U, Breite der Sohle.

## Die Bilder muss der Nutzer ablegen

Ich kann keine Bilder herunterladen. In jedem `REGION.md` stehen dafür die
**Koordinaten** — in Google Earth eingegeben zeigen sie genau den Ausschnitt,
um den es geht, 15 × 15 km, dieselbe Größe wie eine Standardkarte im Programm.

## Wozu die Regionen dienen

Sie sind das Abnahmekriterium, nicht Dekoration. Drei Dinge werden daran
gemessen:

**Verhältnisse statt Absolutwerte.** Ob der Generator 400 oder 700 mm
Niederschlag ausgibt, ist ein Faktor. Ob die Atacama trockener herauskommt als
die Toskana und diese trockener als Guilin, ist die eigentliche Frage. Deshalb
ist der Niederschlag in `REGION.md` **relativ** angegeben, mit dem Erg als 1.0.

**Form gegen Form.** Zu jeder Region gehört ein Querschnitt aus dem Programm,
der neben den echten gelegt wird. Kennzahlen allein haben in diesem Projekt
mehrfach das Falsche gesagt — die 45°-Pyramiden fand keine einzige Zahl, und
`redistribute_power` wurde aus Median und Steigung als Ursache der Nadelgrate
diagnostiziert, was der Querschnitt dann widerlegte.

**Alle gleichzeitig.** Eine Änderung, die Wallis verbessert und Westsibirien
zerlegt, ist keine Verbesserung. Der Sinn von zwanzig Regionen ist, dass
Nebenwirkungen auffallen.

## Warum diese zwanzig

Sie ziehen die Spannweite auf, in der jeder Mechanismus getestet werden muss:

| Achse | Extreme |
|---|---|
| Relief | Westsibirien 30 m gegen Wallis 3800 m |
| Niederschlag | Atacama 2 mm gegen Schottland 2000 mm |
| Temperatur | Tibet −14 °C im Winter gegen Amazonas 26 °C |
| Jahresschwankung | Amazonas 5 K gegen Westsibirien 38 K |
| Talform | Badlands V-dicht gegen Tibet flach gegen Wallis U-glazial |
| Untergrund | Ton (Badlands) gegen Kalk (Dolomiten) gegen Sand (Erg) |
| Gewässer | Tibet endorheisch gegen Kanada seenreich gegen Neuseeland verflochten |

Drei davon sind für offene Baustellen besonders wichtig:

**20 Neuseeland Südalpen** — verflochtene Flüsse auf breiter Schottersohle. Das
ist das Zielbild, das der Nutzer mit Fotos vorgegeben hat.

**16 Badlands** — weicher Ton, extrem dichte V-Verzweigung. Der Härtefall für
die Erosion: hier muss sie viel machen. Zusammen mit **5 Dolomiten** (Kalk,
steile Wände) prüft das die Härte-Kopplung.

**17 Westsibirien** — praktisch kein Relief, träge Mäander, Moor. Der Gegentest
zu allem Gebirgigen, und der Fall, in dem die relief-relative Erosion beweisen
muss, dass sie eine flache Landschaft nicht zerlegt.

## Stand

Die Zielwerte in den `REGION.md` sind mein Vorschlag und als **ZU BESTAETIGEN**
markiert, wo sie eine Einschätzung sind. Sie gehören durchgesehen, bevor gegen
sie kalibriert wird — genau so ist die Temperatur-Klimatologie entstanden, die
heute als einzige Komponente ihre Zielwerte trifft.
