# Invarianten — bei JEDER Änderung zu prüfen

**Was das hier ist.** Die Prüfliste, die für **jede** Änderung am Programm
gilt, unabhängig vom Thema. Sie ist nicht nach Modulen sortiert, sondern
**nach Ursachen, die in diesem Projekt schon zugeschlagen haben**. Jeder
Punkt steht hier, weil er einmal Geld gekostet hat.

*Herkunft: `docs/archiv/2026-07-29_SPEZIFIKATION.md` (Stand 2026-07-29), §4
„Invarianten — bei JEDER Änderung zu prüfen", sowie
`docs/SOLLBESCHREIBUNG.md` (Stand 2026-09-16), Abschnitt „8. Stehende Regeln,
die weiter gelten".*

---

## 0. Die zwei stehenden Regeln

Sie stehen vor allem anderen, weil sie am häufigsten gebrochen wurden.

### 0.1 Was in 2D sichtbar ist, wird in DERSELBEN Änderung auch in 3D gebaut

Nicht „erst 2D, 3D später" — später kommt nicht, und der Nutzer muss es
erneut anfordern. Dreimal derselbe Fehler nach demselben Muster: ein Reiter
ruft eine Anzeigemethode über `hasattr(display, "...")` auf; gibt es sie nur
auf `MapDisplay2D`, trifft die Weiche in der 3D-Ansicht nie zu und **der Code
tut lautlos gar nichts**.

Bewacht von `tests/smoke_test_display_methoden_existieren.py`, Gruppe
`einseitige_sind_begruendet`. Wie man es in 3D baut, steht in
[15_ANZEIGE.md](15_ANZEIGE.md).

### 0.2 Jeder stille Rückfall braucht eine laute Logzeile

Ein `try/except` mit CPU-Rückfall, eine `if geeignet: ... else: ...`-Weiche,
ein `.get(key, default)` — **alles, was im Fehlerfall trotzdem ein plausibles
Ergebnis liefert, ist von Erfolg nicht zu unterscheiden.** Diese Fehlerklasse
ist hier nachweislich fünfmal durchgekommen; jedes Mal lieferte der
Ersatzpfad ein plausibles Ergebnis, und jedes Mal blieben alle Tests grün.

---



## 1. CPU und GPU liefern dasselbe

- Jeder Rechenweg mit GPU-Pfad hat einen Paritätstest gegen die CPU.
- **Eine Änderung an einem Pfad ist erst fertig, wenn der andere mitgezogen
  ist.** Am 2026-07-29 wurde die Rinnenbreite nur im CPU-Pfad eingebaut und
  über die GPU gemessen: vier Varianten kamen bitgleich heraus, die Änderung
  lief nie.
- Konstanten werden vom CPU-Code an den Dispatcher **durchgereicht**, nicht
  doppelt gepflegt (`smoke_test_erosion_gpu_contract`).

## 2. Der geänderte Code wird im Messlauf ausgeführt

Der teuerste Fehlertyp dieses Projekts. Drei Fälle an einem Tag:

| Fall | Was schiefging |
|---|---|
| Breitengrad im Pre-Biome | in `set_active_parameters` gesetzt, gelesen wird `data_lod_manager.get_map_latitude()` |
| Talsohlen-Term | auf flachem Testgelände gemessen, dort ist der Term identisch null |
| Rinnenbreite | CPU-Code geschrieben, GPU-Pfad gemessen |

**Regel:** vor der Auswertung belegen, dass der neue Zweig betreten wurde —
Zähler, Log, oder ein Ergebnis, das ohne die Änderung unmöglich wäre.

## 3. Masse und Bilanz

- Jeder Pass, der Material oder Wasser bewegt, **verteilt um** statt zu
  erzeugen. Der Glättungspass der Erosion schrieb den geglätteten Wert zurück
  und war dadurch der einzige nicht erhaltende Pass; auf Umverteilen
  umgestellt lag die Drift bei 0.000e+00.
- Einheiten und **Zeitbasen** vor dem Vergleich angleichen. Weather rechnet in
  Jahren, Water in 1800 simulierten Sekunden — eine Bilanz aus zwei Uhren
  ergab scheinbar +3517 % Leck.

## 4. Absolute Konstanten sind verdächtig

Viermal derselbe Fehler: eine absolute Größe, wo eine relative hingehört.

| Fall | falsch | richtig |
|---|---|---|
| Regen | pro Schritt | pro Sekunde |
| Konvergenz | pro Schritt | pro Sekunde |
| Glättungsschwelle | absolut | relativ zur Nachbardifferenz |
| Sedimentkapazität | 1 m fest | Bruchteil des Reliefs |

**Regel:** jede neue Konstante mit Einheit muss beantworten, gegen *was* sie
bemessen ist.

## 5. Reihenfolge und Abhängigkeiten

- Der Knotengraph (`CALCULATOR_GRAPH`) ist die einzige Wahrheit über
  Reihenfolge. Jede Liste von Generatoren wird daraus **abgeleitet**, nie von
  Hand geführt. Fünf handgepflegte Listen haben je einen Deadlock oder eine
  fehlende Invalidierung verursacht.
- Der Generator-Baum ist eine Vergröberung und **darf Kreise haben**, auch wenn
  der Knotengraph keine hat. Jede Rekursion darüber braucht Zyklusschutz.
- Wer eine Größe liest, deklariert sie als Kante. Settlement las `biome_map`
  ohne Kante und benutzte je nach Thread-Timing eine Ersatzkarte.

## 6. Anzeige und Bedienung

Nach jeder Änderung an Daten oder Skalen zu prüfen — **nur in der laufenden App
sichtbar**:

- Ändert sich die Karte, ohne dass die Farbskala mitgeht? Passen die Bereiche
  in `gui_default.py layer_ranges` noch zu den tatsächlichen Werten? (Bei der
  Erosion lag einmal die ganze Karte in der hellsten Stufe: Daten da, Skala
  blind.)
- Sind **2D und 3D** dieselbe Skala und derselbe Layer?
  (`smoke_test_layer_2d_3d_parity`)
- Funktioniert **jeder** Tabwechsel, und laden die Grafiken dabei neu?
- Aktualisieren sich Tabs, wenn ein **fremder** Generator geländeformend
  gerechnet hat? (`_TERRAIN_FORMING_GENERATORS`)
- Ist die 3D-Darstellung maßstabstreu? (15 km × 15 km × 4 km → 10 : 10 : 2.67
  Render-Einheiten)
- Sind neue Ausgaben in **allen** Registern eingetragen — Anzeigemodi,
  Layer-Namen, Farbbereiche, Export?

## 7. Regler

- Jeder Regler hat eine **sichtbare Wirkung**. Drei tote Water-Slider gab es
  monatelang.
- Kein Reglerstand erzeugt ein unbrauchbares Ergebnis. Wo das droht, muss die
  Größe an eine andere gekoppelt werden.
- Der Name sagt, was passiert. `SMOOTHING` steuerte eine *Schwelle*: kleine
  Werte bedeuteten aggressives Glätten, das Gegenteil der Erwartung.
- Abhängige Defaults ziehen mit. Ändert sich das Relief, müssen die
  Erosions-Vorgaben dazu passen, ohne dass der Nutzer nachstellt.

---

## Woran diese Liste hängt

* Die Reihenfolgeregeln der Arbeit stehen in
  [03_ARBEITSREGELN.md](03_ARBEITSREGELN.md) — dort auch die Messfallen und
  die vier Fragen, die bei jeder Änderung zu beantworten sind.
* Die Prüfpunkte aus 4.6 sind nur **am laufenden Programm** sichtbar; welche
  Anzeige es überhaupt gibt und wie sie zu bauen ist, steht in
  [15_ANZEIGE.md](15_ANZEIGE.md).
* `tests/smoke_test_regionen_welt.py` ist der empfindlichste Wächter für die
  Geländeform (neun Regionen gegen feste Sollhänge und Wasseranteile, rund
  zwei Minuten). Er gehört in die Prüfliste **jeder** Änderung an
  `weltfeld()` — siehe [10_REGIONEN.md](10_REGIONEN.md).
