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

## 4. Die Zeitgrenze und die Steckenbleib-Notiz

Ein Ticket darf sich nicht die ganze Nacht nehmen. Sonst steht morgens eine
einzige halbfertige Sache da statt sechs fertiger.

Die Grenze ist aber **keine feste Zahl**. Ein Ticket, dessen Tests allein
schon zwanzig Minuten laufen, kann keine Dreissig-Minuten-Grenze haben — die
waere schon abgelaufen, bevor das erste Mal gemessen wurde. Sie wird deshalb
**gerechnet**:

```
Grenze = 3 Versuche × (300 s Arbeit + 3 × Testlaufzeit)
```

Drei Versuche, weil ein Ticket typischerweise bauen, reparieren und
bestaetigen muss. Der Faktor 3 auf die Tests ist der *Streufaktor*: dieselbe
Last schwankt auf dieser Maschine um Faktor zwei bis drei. Die 300 s Arbeit je
Versuch sind ausdruecklich **geschaetzt**, nicht gemessen — das steht auch so
im Quelltext, damit sie niemand fuer einen Messwert haelt. Darunter liegt eine
Untergrenze von 30 min, darueber ein Deckel von 3 h; die Nacht hat rund acht.

```bash
python tools/nachtlauf.py grenze --tests-dauer 120
```

Der Befehl nennt die Grenze **und ihre Rechnung**. Eine Zahl ohne Begruendung
kann man nicht bestreiten, und was man nicht bestreiten kann, korrigiert man
auch nicht.

### Abgebrochen heisst nicht verworfen

Laeuft die Grenze ab, passiert genau eines: es wird eine **Notiz**
geschrieben. Der Branch bleibt, die halbfertige Arbeit bleibt liegen, das
Ticket bleibt **offen** und wird nicht neu gestartet.

```bash
python tools/nachtlauf.py steckengeblieben 104 "Erosionskanaele"     --gelaufen 34 --tests-dauer 120     --stand "Schwellenlogik umgestellt, Sedimentation noch unberuehrt"     --roter-test tests/smoke_test_erosion_quality.py     --meldung "(c) Kanalnetz (groesste Komponente 45 px, Schwelle > 60)"     --versuch "Schwelle 0.6 auf 0.4 gesenkt - 51 px, reicht nicht"     --versuch "Erosionsschritte verdoppelt - Laufzeit x2, unveraendert"     --vermutung "Nicht die Schwelle, sondern die Reihenfolge: die Sedimentation fuellt die Rinne wieder auf, bevor der naechste Schritt sie vertieft."
```

Fehlt davon etwas, **verweigert das Werkzeug die Notiz** und sagt, was fehlt.
Das ist Absicht:

> „Zeitlimit erreicht" ist keine Notiz, das ist eine Uhr. Wer morgens hier
> weitermacht, faengt sonst bei null an und verliert dieselbe Zeit noch
> einmal.

Verlangt sind vier Dinge: **wo die Arbeit steht**, **was rot ist samt
Fehlermeldung** (der Name eines Tests sagt nicht, woran er scheitert), **was
schon versucht wurde** — damit es morgens niemand ein zweites Mal versucht —
und **die naechste Vermutung**.

Die Notizen liegen als JSON in `nachtbetrieb/laufberichte/` und werden bei
`starten` geloescht: sonst zaehlte der Morgenbericht die von vorgestern mit.

---

## 5. Der Morgen

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

## 6. Der Morgenbericht auf einer Seite

`stand` zeigt, was der Branch enthaelt. Der **Bericht** zeigt, was die Nacht
bedeutet — auf einer Seite, weil ein Bericht, den man scrollen muss, morgens
nicht gelesen wird.

```bash
python tools/nachtlauf.py bericht --testlauf laufberichte/schnelllauf.json     --protokoll laufberichte/nacht.log
```

| Block | Was drinsteht | Woher |
|---|---|---|
| 1 Geschlossen | ein Ticket je Zeile, mit Teststand | `git log` ueber den Nachtbranch |
| 2 Rot | jede rote Testdatei mit ihren Meldungen | `tools/testlauf.py --bericht` |
| 3 Kennzahlen | heute, gestern, Veraenderung in Prozent | derselbe Bericht |
| 4 Stille Rueckfaelle | wo das Programm lautlos auf einen Ersatzpfad ausgewichen ist | die mitgeschriebene Ausgabe |
| Steckengeblieben | die Notizen aus Abschnitt 4 | `nachtbetrieb/laufberichte/` |

**Block 4 ist der Grund, warum es diesen Bericht gibt.** Der teuerste Fehler
dieses Projekts ist nicht der Absturz, sondern der stille Rueckfall: das
adaptive 3D-Netz lief monatelang nicht, die GPU-Erosion faellt mit Faktor 385
weniger Abtrag auf die CPU zurueck — beides lieferte plausible Ergebnisse,
beides blieb gruen. Der Bericht sucht deshalb in der mitgeschriebenen Ausgabe
nach **neun namentlich bekannten Ersatzpfaden**, jeder mit Quellzeile und
einem Satz dazu, was er kostet.

Zwei Dinge sagt der Bericht ausdruecklich, statt sie zu verschweigen:

* Wurde **kein Testlauf abgelegt**, steht dort *„Das ist KEIN gruener Befund —
  es wurde nicht gemessen."*
* Wurden **null Zeilen durchsucht**, steht dort, dass null durchsuchte Zeilen
  nicht null Rueckfaelle heisst.

Denn eine leere Messung, die aussieht wie ein gutes Ergebnis, ist genau der
Fehler, gegen den der ganze Block gebaut ist.

War nichts los, sagt der Bericht das in **einem Satz** und man liest den Rest
nicht.

---

## 7. Die Befehle

| Befehl | Was er tut |
|---|---|
| `starten` | Nachtbranch aus `main` anlegen, kollisionsfrei benannt |
| `sperre-pruefen [--arbeitsstand]` | prüft den Branch oder den uncommitteten Stand gegen die Sperrliste |
| `abschliessen <nr> "<titel>" [--tests …]` | ein Ticket, ein Commit, mit Sperrprüfung davor |
| `stand` | Commits, Tickets, Testlage, Sperrlage in einem Aufruf |
| `zuruecknehmen <nr>` | genau dieses Ticket zurück, die anderen unberührt |
| `grenze --tests-dauer <s>` | wieviel Zeit ein Ticket bekommt — **und die Rechnung dazu** |
| `steckengeblieben <nr> …` | Ticket sauber abbrechen: Notiz schreiben, sonst nichts anfassen |
| `bericht [--testlauf …] [--protokoll …]` | der Morgenbericht auf einer Seite |

Rückgabewert `0` heißt in Ordnung, `1` heißt abgebrochen. Ein Steuerskript kann
sich daran halten.

---

## 8. Die Nähte

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

Die Zeitgrenze hat dieselbe Form: alles läuft über **`zeitgrenze.grenze(s)`**
und **`zeitgrenze.notiz(steckenbleib)`**. Beide **rechnen nur** — sie rufen
kein `git` auf, schreiben nichts fest und brechen nichts ab. Genau das ist der
saubere Abbruch: wer nichts anfasst, kann auch nichts verlieren. Deshalb kann
der Test einen abgelaufenen Lauf mit einer **gestellten Uhr** vorführen, ohne
eine halbe Stunde zu warten — ein Test, der wirklich wartet, wird abgeschaltet,
und dann prüft niemand mehr den Abbruch.

Block 4 des Berichts läuft über **`morgenbericht.zaehle_rueckfaelle(texte)`**:
Text hinein, Treffer und **die Zahl der durchsuchten Zeilen** heraus. Die
zweite Zahl ist nicht schmückend — ohne sie wäre „keine Rückfälle gefunden"
nicht von „nichts durchsucht" zu unterscheiden.

Und weil ein Wächter, der ins Leere zeigt, schlimmer ist als keiner, prüft der
Test zusätzlich, dass **jedes der neun Muster im Quelltext wirklich noch
vorkommt**. Verschwindet eine Logzeile bei einem Umbau, schlägt er fehl,
statt ab dann für immer null Rückfälle zu melden.

---

## 9. Dateien

| Datei | Wofür |
|---|---|
| [`nachtbetrieb/sperrliste.toml`](../nachtbetrieb/sperrliste.toml) | wo nachts niemand hinfasst, mit Begründung je Eintrag |
| [`nachtbetrieb/sperre.py`](../nachtbetrieb/sperre.py) | die Durchsetzung dieser Liste |
| [`nachtbetrieb/branch.py`](../nachtbetrieb/branch.py) | Branch, Commit-Format, Stand, Rücknahme |
| [`tools/nachtlauf.py`](../tools/nachtlauf.py) | die Bedienung |
| [`nachtbetrieb/zeitgrenze.py`](../nachtbetrieb/zeitgrenze.py) | Zeitgrenze rechnen, Steckenbleib-Notiz erzwingen und ablegen |
| [`nachtbetrieb/morgenbericht.py`](../nachtbetrieb/morgenbericht.py) | die vier Blöcke, die neun Rückfallmarken |
| `nachtbetrieb/laufberichte/` | Notizen und Kennzahlen eines Laufs; nicht versioniert |
| [`tests/smoke_test_nachtbetrieb.py`](../tests/smoke_test_nachtbetrieb.py) | führt alles je einmal vor, fünfzehn Gruppen |

`nachtbetrieb/` ist bewusst **kein Teil des Programms**: es steht nicht in der
`include`-Liste von `pyproject.toml` und wird nie von `core/`, `gui/` oder
`managers/` importiert. Es ist Werkzeug, so wie `tools/`.
