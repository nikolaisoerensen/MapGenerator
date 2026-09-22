# 15 — Anzeige und Overlays

Was gelten soll, wenn ein Haken in einem Reiter etwas auf der Karte sichtbar
macht: dass er in 2D und 3D dasselbe bewirkt, wo das entschieden wird und wie es
geprüft wird. Nicht hier: wie eine Karte gerechnet wird (eigene Kapitel) und wie
ein Overlay aussieht. Jeder Block trägt seinen Umsetzungsstand.

---

## Wortklärung

Die englischen Entwurfsbegriffe der Quelle bleiben stehen, damit sie nicht mit
„Komponente" oder „Schnittstelle" verwechselt werden — je einmal erklärt:

* **module** — ein Stück Programm, das eine Sache allein verantwortet; hier das
  Overlay-Register in `gui/tabs/base_tab.py`.
* **interface** — das Wenige, das ein Nutzer des Moduls kennen muss; hier genau
  eine Methode, `_push_overlays([...])`, und der Werttyp `Overlay`.
* **implementation** — alles dahinter, das man ändern darf, ohne dass ein
  Reiter es merkt: welche Zeichenfunktion für welchen Namen gerufen wird.
* **seam** (Naht) — die eine Stelle, an der alle Reiter dieselbe Auskunft
  abholen; hier `BaseMapTab`, weil dort schon die Skalarlayer durchlaufen.
* **adapter** — das Zwischenstück, das einen Namen in einen konkreten Aufruf
  übersetzt; je Overlay einer für 2D und einer für 3D.
* **deep** / **shallow** — ein tiefes module verbirgt viel hinter wenig
  interface; ein flaches reicht die Arbeit fast unverändert weiter.
* **leverage** — Hebelwirkung: eine Änderung wirkt in allen sechs Reitern.
* **locality** — Zusammenliegen: 2D-Weg und 3D-Weg eines Overlays stehen in
  derselben Zeile des Registers, nicht in vier Dateien.

---

## Stehende Regel: was in 2D sichtbar ist, gehört auch in 3D

Jede Anzeige, jeder Haken, jedes Overlay wird in **derselben Änderung** für 2D
und 3D gebaut. Nicht „erst 2D, 3D später" — später kommt nicht.

Der Grund ist ein Mechanismus, kein Vergessen: ruft ein Reiter eine
Anzeigemethode über `hasattr(display, "...")` auf und gibt es sie nur auf
`MapDisplay2D`, trifft die Weiche in der 3D-Ansicht nie zu und der Code tut
lautlos gar nichts — keine Fehlermeldung, keine Warnung. So verschwanden
2026-08-24 die Flüsse (`overlay_river_generations`), 2026-08-25 das Flussnetz
und die Siedlungen im Biome-Reiter, und ein viertes Mal am 2026-09-14. Dreimal
davon wurde als behoben verbucht.

In 3D wird ein Overlay fast immer **ohne neuen GLSL-Code** gebaut: über die
`rasterize_*_rgba()`-Funktionen auf eine RGBA-Textur und per
`update_overlay_data(bereich, name, rgba)` als Alpha-Haut auf das Gelände, dazu
`set_layer_visibility(...)` zum Ein- und Ausschalten. Echte Geometrie lohnt nur,
wo die Textur an ihre Grenze stößt — so bei den Wegen
(`gui/widgets/wege_geometrie.py`, scharf beim Zoomen und anklickbar).

**Stand: umgesetzt und bewacht.** Jede Anzeigemethode, die es nur auf einer der
beiden Klassen gibt, muss in `tests/smoke_test_display_methoden_existieren.py`
eingetragen sein — in `NUR_EINE_ANZEIGE` (einseitig und richtig so, `:153`) oder
in `FEHLT_IM_3D` (einseitig und eine Schuld, `:199`). Die Schuldliste darf nur
schrumpfen; kommt etwas Neues dazu, schlägt der Test fehl.

*Herkunft: `CLAUDE.md` (Projektwurzel), Abschnitt „STEHENDE REGEL: was in 2D
sichtbar ist, gehoert auch in 3D"; Nutzervorgabe 2026-08-25.*

---

## Festlegungen

Die stehende Regel wurde viermal gebrochen, weil sie eine Gedächtnisleistung
verlangt, wo jeder Reiter die Weiche selbst baut — zum Zeitpunkt der
Spezifikation 43 `hasattr`-Abfragen in `gui/`. Die Festlegungen drehen deshalb
daran, **wer entscheidet**: ein Reiter meldet an, *was* gezeigt werden soll; ein
einziges deep module entscheidet, *wie*.

### 1. Die Naht liegt in `BaseMapTab` — es ist die bestehende

Entscheidung des Nutzers vom 2026-09-14: **keine neue seam nach außen.** Die
Verantwortung „Anzeigedaten ans richtige Display schicken" liegt schon bei
`BaseMapTab` (`_push_data_to_current_display()` für Skalarlayer) und wird um
Vektor-Overlays erweitert, statt daneben eine zweite Stelle aufzumachen. Das
gesamte interface eines Reiters ist
`self._push_overlays([Overlay("siedlungen", sichtbar=..., daten=...), ...])`.

**Stand: umgesetzt.** `gui/tabs/base_tab.py:828` (`_push_overlays`), direkt
neben `_push_data_to_current_display()` (`:689`).

### 2. Ein Overlay ist ein Wert, kein Aufruf

Ein Overlay trägt Name, sichtbar ja/nein, Daten — nicht aber, *wie* gezeichnet
wird; genau deshalb kann ein Reiter es nicht mehr falsch machen.
`sichtbar=False` ist ein vollwertiger Zustand, kein Weglassen: erst dadurch kann
eine liegengebliebene 3D-Textur abgeräumt werden.

**Stand: umgesetzt.** `gui/tabs/base_tab.py:76` (`@dataclass(frozen=True) class
Overlay`); Abräumen bei `sichtbar=False` in `:160` (`set_layer_visibility`) und
`:187` (`clear_river_overlay()`).

### 3. Hinter der Naht liegt ein Register, kein `if`

Ein internes module — privat zur implementation von `BaseMapTab`, aber mit
eigenen Tests — hält je Overlay-Namen beide Wege: den **2D-adapter** (welche
`overlay_*()`-Methode mit welchen Argumenten) und den **3D-adapter** (welche
`rasterize_*_rgba()`-Funktion, unter welchem `(bereich, layername)` die Textur
abgelegt wird). Ein Name ohne Eintrag ist ein Fehler, keine stille Auslassung.
Nur hier stehen beide Dialekte nebeneinander — darum geht es bei locality.

**Stand: umgesetzt, mit zwei Einträgen.** `gui/tabs/base_tab.py:194`
(`_OVERLAY_REGISTER`) kennt `"siedlungen"` und `"fluesse"`; die Schuldliste ist
bewusst nicht dabei (siehe Abgrenzung).

### 4. Immer beide Anzeigen bedienen, nie `current_view` fragen

Die zentrale Regel dieses Kapitels. Das Modul schiebt an *beide* Anzeigen,
unabhängig davon, welche sichtbar ist — die 3D-Anzeige existiert je Reiter
immer, auch während 2D sichtbar ist. Wer sie nur bei `current_view == "3d"`
füllt, zeigt beim Umschalten ein leeres Bild oder erzwingt ein Neuerzeugen.
`current_view` darf in dieser implementation **nicht vorkommen**; das ist
nachprüfbar und wird nachgeprüft.

**Stand: umgesetzt und maschinell bewacht.** `gui/tabs/base_tab.py:860-863`
(zwei unbedingte Zweige); Quelltexttest `tests/smoke_test_push_overlays.py:244-253`.

### 5. Ein fehlender 3D-Weg wird laut

Drei Zustände, drei Reaktionen — vorher waren alle drei dasselbe Nichts:

| Zustand | Reaktion |
|---|---|
| Overlay-Name unbekannt | Fehler, mit Namen |
| Overlay angemeldet, 3D-Weg fehlt, **nicht** in der Schuldliste | Fehler |
| Overlay angemeldet, 3D-Weg fehlt, **in** der Schuldliste | läuft, schreibt eine WARNING je Anlass |

Das folgt der Projektlehre, die zweimal teuer war (Shaderpfade nach dem
Dateiumzug, adaptives Netz): jeder stille Rückfall braucht eine laute Logzeile.

**Stand: Zeile 1 umgesetzt, Zeilen 2 und 3 beschlossen, aber offen.** Der
unbekannte Name wirft (`gui/tabs/base_tab.py:854-858`, `ValueError`) — beim Aufruf,
nicht beim Programmstart. Die Zustände 2 und 3 gibt es im Register gar nicht:
ein Eintrag hat immer beide adapter. Die Restschuld steht weiter außerhalb in
`FEHLT_IM_3D` (`tests/smoke_test_display_methoden_existieren.py:199`), ihre
Weichen laufen direkt im Reiter (`gui/tabs/settlement_regional_tab.py:318`,
`:331`, `:343`, `:357`) — dort warnt immerhin der umgebende Block laut statt
per `logger.debug` (`:368`).

### 6. Die beiden Namensräume werden zusammengeführt

`_LAYER_NAME_MAP_3D` übersetzt die Skalarlayer (`"temp_map"` →
`"temperature"`); die 3D-Layernamen der Vektor-Overlays (`"uebersicht"`,
`"wegbaender"`) standen als blanke Zeichenketten in den Reitern. Beide
Zuordnungen sollen in das Register aus Punkt 3 ziehen.

**Stand: beschlossen, aber offen.** `_LAYER_NAME_MAP_3D` steht weiter getrennt
neben dem Register (`gui/tabs/base_tab.py:620`), und
`tests/smoke_test_layer_2d_3d_parity.py:78` prüft gegen diese Tabelle statt
gegen das Register. Umgesetzt ist nur der Tippfehler-Teil: ein unbekannter
`layer_name` meldet sich laut über `rendering_error`
(`gui/widgets/map_display_3d.py:999`, `:1045`). Die Zusammenlegung selbst galt
als zu riskant für einen Nachtlauf (`docs/OFFENE_PUNKTE.md`, Punkt 14.7).

### 7. Was sich nicht ändert

* **Die Anzeigeklassen bleiben, wie sie sind.** `MapDisplay2D` lernt nichts
  Neues, `MapDisplay3D` auch nicht. Die verworfene Alternative — der 3D-Klasse
  dieselben `overlay_*`-Methoden zu geben — hätte neun Methoden und vier neue
  Rasterfunktionen gekostet und wäre nur am laufenden Programm prüfbar gewesen.
* **Die `rasterize_*_rgba()`-Funktionen bleiben und werden weiterverwendet.**
  **Wege bleiben echte Bandgeometrie** (`gui/widgets/wege_geometrie.py`), keine
  Textur — das Register kennt sie als eigenen 3D-adapter.
* **Skalarlayer bleiben bei `_push_data_to_current_display()`.** Beide Methoden
  stehen nebeneinander und teilen sich das Register, verschmelzen aber nicht.

**Stand: eingehalten, mit einer bewussten Abweichung.** Die fünf
`rasterize_*_rgba()` liegen nicht mehr in der 2D-Anzeige, sondern in einem
neutralen Modul (`gui/widgets/overlay_rasterizer.py:77` ff.), damit das Register
nicht auf das 2D-Modul zeigt. Wege als reine Geometrie ist im Wächtertest
begründet (Eintrag `overlay_roads` in `NUR_EINE_ANZEIGE`,
`tests/smoke_test_display_methoden_existieren.py:168`).

### 8. Der vorzeitige Ausstieg im Biome-Reiter fällt weg

`BiomeTab.apply_overlays()` stieg in der ersten Zeile aus, wenn die Ansicht
nicht 2D war — die 3D-Zweige darunter konnten nie laufen. Das wird nicht einzeln
repariert, sondern durch die Umstellung auf `_push_overlays()` behoben; der
Kommentarblock darüber beschrieb eine Behebung, die nie gewirkt hatte.

**Stand: umgesetzt.** `gui/tabs/biome_tab.py:531`.

### 9. Die Ausnahme im Wächtertest wird neu begründet

Der Eintrag für `overlay_settlements` in `NUR_EINE_ANZEIGE` berief sich auf
unerreichbaren Code — der Test war grün *wegen* des toten Codes. Die Begründung
muss auf das Register zeigen, nicht auf eine Reiterzeile.

**Stand: umgesetzt.** `tests/smoke_test_display_methoden_existieren.py:160-167`
zeigt auf den Register-Eintrag `"siedlungen"`.

### 10. Reihenfolge des Umbaus

Ein Reiter zuerst, vollständig, dann Sichtprüfung, dann die anderen — zuerst
**Biome** (dort sitzt der Fehler, beide Overlays haben ihren Rasterweg), danach
Siedlungen, Regional, Fluss.

**Stand: umgesetzt für alle vier Reiter.** `gui/tabs/biome_tab.py:531`,
`gui/tabs/settlement_tab.py:728`, `gui/tabs/settlement_regional_tab.py:301`,
`gui/tabs/river_tab.py:425` und `:436`. Im Regional-Reiter läuft nur das
Siedlungs-Overlay über das Register; die vier übrigen Weichen sind Restschuld
(Punkt 5).

*Herkunft: `docs/SPEC_OVERLAYS.md` (Stand 2026-09-14), Abschnitt „Implementation
Decisions"; die Einleitung oben aus „Problem Statement" und „Solution".*

---

## Prüfregeln

Geprüft wird **beobachtbares Verhalten am interface**: welche Aufrufe bei
welcher Anzeige ankommen — nicht, wie das Modul intern entscheidet. Der Gewinn:
die Zeichenentscheidung ist dadurch **ohne Qt und ohne OpenGL** prüfbar; genau
weil das vorher fehlte, blieben die vier Vorfälle unbemerkt.

| # | Regel | Stand |
|---|---|---|
| 1 | Das Overlay-Modul gegen mitschreibende Attrappen: je angemeldetem Overlay erzeugt `sichtbar=True` in **beiden** Protokollen einen Eintrag, `sichtbar=False` in beiden das Abräumen, ein unbekannter Name wirft. Kein Qt. | **umgesetzt** — `tests/smoke_test_push_overlays.py:114`, `:138`, `:162`, `:180`, `:202` |
| 2 | `current_view` kommt in der neuen implementation nicht vor (Quelltexttest). Klingt grob, trifft aber genau die Ursache aller vier Fälle. | **umgesetzt** — `tests/smoke_test_push_overlays.py:244-253` |
| 3 | **Erreichbarkeit statt Existenz:** der Wächter fragt nicht nur, ob eine Methode existiert, sondern ob der Zweig, der sie ruft, überhaupt laufen kann. Ein vorzeitiges `return`, das alle 3D-Zweige abschneidet, muss fehlschlagen. Ohne diesen Punkt wiederholt sich der Fall. | **umgesetzt** — `tests/smoke_test_display_methoden_existieren.py:337` (`run_zweige_sind_erreichbar`) |
| 4 | Die Schuldliste darf nur schrumpfen, und eine Begründung darf nicht auf Reiter-Zeilen zeigen. | **teils umgesetzt** — Schrumpfen und veraltete Einträge werden geprüft (`tests/smoke_test_display_methoden_existieren.py:266`, `:272`); dass eine Begründung nicht auf eine Reiterzeile zeigt, ist redaktionell erfüllt, aber nicht maschinell geprüft |
| 5 | Echte Kartengrößen (128/256/512/1024), nicht ausgedachte — die Lehre aus dem adaptiven Netz, wo zehn grüne Tests eine Funktion prüften, die im Betrieb nie lief. | **teils umgesetzt** — nur 256 (`tests/smoke_test_push_overlays.py:105`, `tests/smoke_test_biome_overlays_3d.py:63`) |

Vorbilder im Repo: `tests/smoke_test_display_methoden_existieren.py` (der
Wächter, wird erweitert statt ersetzt — seine Trennung zwischen
`NUR_EINE_ANZEIGE` und `FEHLT_IM_3D` bleibt), `tests/smoke_test_display_2d.py`
(alle 30 2D-Darstellungen) und `tests/smoke_test_pipeline_outputs.py` (fährt
denselben Ablauf zweimal und meldet jeden Unterschied — dieselbe Form wie
„denselben Zustand über beide adapter fahren und vergleichen").

**Nur am laufenden Programm prüfbar bleibt, ob die Textur richtig *aussieht*.**
Die Tests belegen, dass beide adapter mit vergleichbaren Daten gerufen werden;
ob die RGBA-Haut im 3D an der richtigen Stelle sitzt, sieht nur der Nutzer —
dafür je umgestelltem Reiter ein Eintrag in
`docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md`.

*Herkunft: `docs/SPEC_OVERLAYS.md` (Stand 2026-09-14), Abschnitt „Testing
Decisions".*

---

## Abgrenzung

Nicht Gegenstand dieses Kapitels:

* **Die fünf Einträge der Schuldliste tatsächlich in 3D bauen.** Der
  Mechanismus macht die Schuld sichtbar, löst sie aber nicht ein. Wo ein
  Rasterweg existiert (Regionen), kann er mitgenommen werden; wo einer fehlt
  (Regionsgitter, Stadtgrenzkontur, Höhenlinien), ist das eigene Arbeit.
* **Die 3D-Anzeige um `overlay_*`-Methoden erweitern** (verworfen, siehe
  Festlegung 7) und **Wege als Textur** (bleiben Geometrie).
* **`gui/tabs/overview_tab.py`.** Damals vollständig tot (vier Aufrufe auf
  Methoden, die es nirgends gibt); inzwischen ist der tote Composite-Teil
  entfernt (`gui/tabs/overview_tab.py:10-15`).
* **Die doppelte Auspackung von `DisplayWrapper`** — manche Weichen fragen den
  Wrapper, manche das innere Objekt. Wird berührt, nicht aufgeräumt.
* **Leistung.** Kein Ziel; nur das Umschalten zwischen 2D und 3D darf nicht
  spürbar langsamer werden als vorher.

*Herkunft: `docs/SPEC_OVERLAYS.md` (Stand 2026-09-14), Abschnitt „Out of
Scope".*

---

## Offene Fragen

1. **Sind Parzellengrenzen in 3D überhaupt sinnvoll?** Unentschieden — tausende
   Parzellen könnten Pixelmatsch werden (`docs/SPEC_OVERLAYS.md`, „Out of
   Scope"; `tests/smoke_test_display_methoden_existieren.py:208`).
2. **Wie sieht der Zustand „angemeldet, aber ohne 3D-Weg" im Register aus?**
   Festlegung 5 verlangt für ihn Fehler bzw. WARNING; das Register kennt nur
   Einträge mit beiden adaptern (`gui/tabs/base_tab.py:194`). Unklar, ob die
   Schuldliste dorthin wandert oder im Wächtertest bleibt.
3. **„Fehler beim Start" oder beim Aufruf?** Festlegung 5 sagt „beim Start",
   umgesetzt ist ein `ValueError` beim ersten `_push_overlays()`-Aufruf
   (`gui/tabs/base_tab.py:854-858`). Eine Startprüfung ist nirgends festgelegt.
4. **Welche Kartengrößen sind Pflicht?** Prüfregel 5 nennt vier, die Tests fahren
   eine (256); ob alle vier je Test verlangt sind, ist nicht festgelegt.
