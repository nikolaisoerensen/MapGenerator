# Nachtbetrieb

**Stand 2026-09-16.** Wie ein Agent nachts allein arbeitet, ohne dass morgens
jemand überrascht wird.

Diese Datei steht selbst in der Sperrliste: ein Nachtlauf darf sie nicht
ändern. Sie ist Anweisung an den Agenten, nicht sein Arbeitsmaterial.

---

## 1. Die zwei Regeln

1. **Ein Branch je Nacht, ein Commit je Ticket.** `main` bleibt unberührt, bis
   morgens jemand hinschaut und zusammenführt.
2. **Was in der Sperrliste steht, wird nachts nicht angefasst.** Auch nicht
   "nur kurz", auch nicht "weil der Test sonst rot bleibt".

Beide Regeln werden durchgesetzt, nicht nur aufgeschrieben. `abschliessen`
verweigert den Dienst auf `main`, und es verweigert ihn, sobald eine gesperrte
Datei im Arbeitsstand liegt.

---

## 2. Warum es die Sperrliste gibt

Die teuersten Fehler dieses Projekts waren nie Abstürze. Es waren Änderungen,
die plausibel aussahen und still etwas kaputtmachten:

| Was passierte | Was man sah |
|---|---|
| `shader_manager.py` zog eine Ebene um, `SHADERS_ROOT` zeigte daneben | nichts — jede GPU-Operation fiel still auf CPU zurück |
| Adaptives 3D-Netz verlangte `2^n+1`, das Programm liefert `2^n` | nichts — zehn grüne Tests, die Funktion lief nie |
| numpy 2.5 statt 2.4.6 | nichts — `import numba` scheitert, opensimplex fängt es ab, Faktor 213 langsamer |

Keiner dieser Fälle wird rot. Ein Agent, der nachts allein arbeitet, kann sie
alle auslösen und morgens einen grünen Bericht abliefern. Die Sperrliste ist
der Zaun um genau die Dateien, bei denen das passiert.

**Die Liste:** [`nachtbetrieb/sperrliste.toml`](../nachtbetrieb/sperrliste.toml).
Jeder Eintrag trägt seine Begründung im Klartext — wer nachts vor der Sperre
steht, soll lesen können, warum sie da ist, statt sie für Willkür zu halten.

### Zwei Stufen

* **`sperre`** — der Lauf bricht ab. Die Änderung bleibt uncommittet liegen,
  das Ticket bleibt offen, und die Abbruchnotiz wird hineingeschrieben.
* **`warnung`** — der Lauf geht weiter, aber die Datei steht am Morgen im
  Bericht. Für Dateien, in denen echte Arbeit UND Empfindliches zusammen
  wohnen: `core/terrain_weltkarte.py` beherbergt die neun Regionsparameter,
  aber auch das Gelände selbst. Sobald Ticket #29 die Parameter herausholt,
  wird daraus eine harte Sperre.

### Selbstschutz

Die Sperrliste und ihre Durchsetzung stehen im ersten Eintrag der Liste. Ein
Nachtlauf, der die Sperrliste aufweichen will, bricht an seiner eigenen
Prüfung ab. `lade_sperrliste()` weigert sich zusätzlich zu laden, wenn diese
Einträge fehlen — eine leere Sperrliste sieht von außen aus wie eine erfüllte
Prüfung, und genau das darf sie nicht.

**Einträge entfernt nur der Nutzer.**

---

## 3. Der Ablauf einer Nacht

```bash
python tools/nachtlauf.py starten
```

Legt `nacht/2026-09-16` aus `main` an und wechselt hinein. Liegt der Branch von
gestern noch da, heißt der neue `-b`, dann `-c`. Der alte wird nie
überschrieben — das wäre die Arbeit einer ganzen Nacht.

Startet nur auf sauberem Arbeitsverzeichnis. Sonst wanderte fremde,
uncommittete Arbeit in die Nacht-Commits.

Dann, je Ticket:

```bash
python tools/nachtlauf.py abschliessen 57 "Sperrliste anlegen" --tests "gruen - tests/smoke_test_nachtbetrieb.py"
```

Ein Ticket, ein Commit. Die Nachricht hat immer dieselbe Form:

```
nacht(#57): Sperrliste fuer den Nachtbetrieb anlegen

<was der Agent getan hat, in Prosa>

Ticket: https://github.com/nikolaisoerensen/MapGenerator/issues/57
Tests: gruen — tests/smoke_test_nachtbetrieb.py
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

**Der Teststand steht in der Commit-Nachricht, nicht in einem
Begleitprotokoll.** Drei Gründe: `stand` liest ihn direkt aus `git log`
zurück; er überlebt eine Rücknahme, weil der Commit in der Historie bleibt;
und im Code-Review steht er neben genau der Änderung, für die er behauptet
wurde. Ein Begleitprotokoll wäre eine zweite Wahrheit, die still
auseinanderlaufen kann — und still auseinanderlaufende Wahrheiten sind in
diesem Projekt schon oft genug teuer geworden.

Steht dort `nicht gelaufen`, ist das eine ehrliche Aussage und kein Makel. Ein
falsches *grün* ist der Makel — siehe die Erosionszeile in
`docs/TESTBERICHT.md`, die monatelang behauptete, Erosion sei abgeschaltet,
während `EROSION_AKTIV = True` stand.

---

## 4. Der Morgen

```bash
python tools/nachtlauf.py stand
```

Ein Befehl, und man sieht alles: welcher Branch, welche Tickets, welcher
Teststand je Ticket, und ob die Sperrliste sauber blieb.

```
Branch: nacht/2026-09-16  (gegen main)
2 Ticket(s) abgeschlossen:
  42ee77e0  #101   Erstes Ticket
            Tests: gruen - tests/smoke_test_a.py
  34bb6b66  #102   Zweites Ticket
            Tests: gruen - tests/smoke_test_b.py

Sperrliste: sauber
```

Passt ein Ticket nicht:

```bash
python tools/nachtlauf.py zuruecknehmen 102
```

`git revert` genau dieses einen Commits. Die anderen Tickets bleiben, wie sie
waren — das ist der ganze Grund für *ein Commit je Ticket*. Bei einem
Sammelcommit müsste man von Hand auseinanderklauben, was zusammengehört, und
zwar morgens, ohne zu wissen, was der Agent sich nachts dabei gedacht hat.

Erst danach wird zusammengeführt. Der Code-Review findet auf dem Branch statt,
nicht auf `main`.

---

## 5. Die Befehle

| Befehl | Was er tut |
|---|---|
| `starten` | Nachtbranch aus `main` anlegen, kollisionsfrei benannt |
| `sperre-pruefen [--arbeitsstand]` | prüft den Branch oder den uncommitteten Stand gegen die Sperrliste |
| `abschliessen <nr> "<titel>" [--tests …]` | ein Ticket, ein Commit, mit Sperrprüfung davor |
| `stand` | Commits, Tickets, Testlage, Sperrlage in einem Aufruf |
| `zuruecknehmen <nr>` | genau dieses Ticket zurück, die anderen unberührt |

Rückgabewert `0` heißt in Ordnung, `1` heißt abgebrochen. Ein Steuerskript kann
sich daran halten.

---

## 6. Die Naht

Alles läuft über **`sperre.pruefe(dateien)`**: Pfade hinein, Treffer heraus.
Ob die Pfade aus `git diff`, aus einem Test oder von Hand kommen, ist dieser
Funktion gleich.

> *Naht: die eine Stelle, an der alle Aufrufer dieselbe Auskunft abholen.*

Deshalb kann `tests/smoke_test_nachtbetrieb.py` die Durchsetzung vorführen,
ohne einen Nachtlauf zu starten — und deshalb prüft er die **echte**
Sperrliste, nicht eine ausgedachte. Ein Test mit ausgedachten Daten prüft eine
ausgedachte Situation; genau daran lief das adaptive 3D-Netz monatelang
unbemerkt vorbei.

Für die Branch-Teile baut der Test sich ein Wegwerf-Repository in einem
Temporärverzeichnis. Er committet nie ins echte Projekt.

---

## 7. Dateien

| Datei | Wofür |
|---|---|
| [`nachtbetrieb/sperrliste.toml`](../nachtbetrieb/sperrliste.toml) | wo nachts niemand hinfasst, mit Begründung je Eintrag |
| [`nachtbetrieb/sperre.py`](../nachtbetrieb/sperre.py) | die Durchsetzung dieser Liste |
| [`nachtbetrieb/branch.py`](../nachtbetrieb/branch.py) | Branch, Commit-Format, Stand, Rücknahme |
| [`tools/nachtlauf.py`](../tools/nachtlauf.py) | die Bedienung |
| [`tests/smoke_test_nachtbetrieb.py`](../tests/smoke_test_nachtbetrieb.py) | führt beides je einmal vor, neun Gruppen |

`nachtbetrieb/` ist bewusst **kein Teil des Programms**: es steht nicht in der
`include`-Liste von `pyproject.toml` und wird nie von `core/`, `gui/` oder
`managers/` importiert. Es ist Werkzeug, so wie `tools/`.
