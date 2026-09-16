# Testbericht — Stand 2026-08-27

Erzeugt mit `tools/testlauf.py`, jede Datei als eigener Prozess.

**MESSWARNUNG:** diese Maschine schwankt um Faktor 2–3. Dieselbe feste
Referenzlast maß zwischen 0,80 s und 1,91 s ohne erkennbare Fremdlast.
Laufzeiten sind Größenordnungen, keine Messwerte.

Diese Datei zeigt den **jetzigen** Stand und wurde dafür neu geschrieben.
Die ausführlichere Fassung vom 24.08. mit der Herleitung der Längenquote
steht in der Versionsverwaltung (`git show HEAD:docs/TESTBERICHT.md`); die
fachliche Begründung je Befund steht in `docs/SITZUNGSLOG.md`.

## Das Wichtigste in fünf Zeilen

* **60 von 70 Testdateien bestehen**, Gesamtlaufzeit **1153 s (19,2 Min)**.
  Nach den beiden Korrekturen unten sind es **62 von 70**.
* **Ein Fehlschlag war neu und selbst verursacht** — `layer_2d_3d_parity`
  fand, dass die vier neuen Terrain-Karten in 3D **ohne Farbtafel** liefen.
  Behoben. Das ist der wichtigste Befund dieses Laufs.
* **Ein Fehlschlag war ein FEHLER IM TEST, nicht im Code** —
  `archetyp_verteilung` maß die Auflösung seiner eigenen Testkarte.
  Geklärt durch zwei Enden gegeneinander, behoben.
* **Sieben Fehlschläge sind bekannt und standen schon im Bericht vom
  24.08.** — die drei Erosionsbefunde, `regionen_welt`, `pipeline_outputs`,
  `settlement_placement`, `weather_temperature_direktnormierung`.
* **Kein einziger Absturz.** Alle Fehlschläge sind Zusicherungen, die eine
  Kennzahl gegen einen Zielwert halten.

---

## 1. Der neue Befund: vier Karten ohne Farbtafel in 3D (behoben)

```
[28/70] smoke_test_layer_2d_3d_parity.py                        FAIL(1)
          [FAIL] terrain/hinterland_height: Range-Mapping vorhanden
          [FAIL] terrain/river_order: Range-Mapping vorhanden
          [FAIL] terrain/river_water: Range-Mapping vorhanden
```

**Was passiert war.** Die vier neuen Karten (`river_water`, `river_order`,
`hinterland_height`, `voronoi_map`) bekamen am 26.08. ihre Farbskalen in
`CanvasSettings.CANVAS_2D["layer_ranges"]`. Damit sah die Sache erledigt
aus. Die 3D-Ansicht liest diese Tabelle aber **nicht direkt** — sie geht
über `_LAYER_RANGE_KEY_MAP` in `gui/widgets/map_display_3d.py`, und dort
fehlten alle vier.

**Warum es unsichtbar war.** `_colorize_layer()` findet keinen `range_key`,
fällt auf Auto-Skalierung ohne Farbtafel zurück und zeichnet **trotzdem
ein plausibles Bild** — nur mit anderen Farben und anderer Skala als 2D.
Kein Absturz, keine Warnung. Das ist derselbe stille Rückfall, den
CLAUDE.md an drei anderen Stellen beschreibt, und er hat sich hier zum
vierten Mal wiederholt.

**Gefunden hat es kein Mensch und kein Blick auf das Bild**, sondern der
Test, der genau diese Doppelregistrierung bewacht. Ohne ihn wäre es erst
beim Vergleich zweier Screenshots aufgefallen — oder gar nicht.

Behoben; `smoke_test_layer_2d_3d_parity.py` ist wieder 4/4 grün.

## 2. Geklärt: Vendée war ein Artefakt des Tests (behoben)

```
[ 4/70] smoke_test_archetyp_verteilung.py                       FAIL(1)
          [FAIL] jeder Archetyp im Mittel innerhalb 16 Punkten
                 - Vendee-Straende -16.7P
```

**Gemessen wurden zwei Enden der Kette bei zwei Auflösungen, je 8 Karten:**

| Vendée-Strände | Saatanteil | Längenanteil |
|---|---:|---:|
| 384 px (55 m/px) | −5,0 | **−20,1** |
| 768 px (28 m/px) | −4,9 | **−0,2** |

Das Saatende ist bei beiden Auflösungen gleich — **die Zuordnung stimmt.**
Der Verlust entsteht danach und verschwindet bei feiner Karte vollständig.
Die Gegenprobe liefert Bretagne-Klippen, Vendées Regionsnachbar: +11,4 bei
384 px, −5,8 bei 768 px. Die beiden tauschen genau das, was Vendée fehlt.

**Ursache:** `MIN_SEGMENT_M` (750 m) ist eine *absolute* Länge. Vendée hat
mit 0,18 km die kürzeste Reichweite der drei Atlantik-Typen; bei 55 m/px
fallen seine Zonen darunter und werden in `_segmente_schliessen()` in den
längeren Nachbarn eingeschmolzen. Der Test hat die Auflösung **seiner
eigenen Testkarte** gemessen, nicht die Verteilung.

**Was daraus wurde.** Der Test prüft jetzt beide Enden mit zwei Grenzen:
den Saatanteil scharf (10 Punkte; schlechtester gemessener Wert 5,2 über 24
Archetypen) und den Längenanteil weich (22 Punkte). Der Fehler, für den es
diese Datei gibt — ein Archetyp, der systematisch ausfällt — liegt 25 bis 50
Punkte daneben und fällt durch **beide** Grenzen; die weiche Längengrenze
verliert also nichts.

Auf der echten Karte (1024 px, 21 m/px) ist das Einschmelzen noch geringer
als bei 768 px. Der Effekt ist damit **kein Problem der erzeugten Karte**,
sondern eine Eigenschaft grober Testauflösungen. Notiert in
`docs/OFFENE_PUNKTE.md`.

## 3. Bekannt und unverändert

| Test | Befund | Stand |
|---|---|---|
| `erosion_field` | Farbskala `[0.5, 300]`, typischer Wert 0,22 | Farbskalenproblem, kein Rechenproblem — seit 24.08. |
| `erosion_gpu_parity` | Export 38,5 gegen 0,1 m auf der CPU | **echter Paritätsbruch**, Faktor 385, seit 24.08. |
| `erosion_quality` | Kanalnetz 45 px statt > 60, Ebenen 7,2 % statt 15–55 % | **ungeklärt** — frühere Erklärung ("Erosionskette abgeschaltet") war falsch, siehe unten |
| `regionen_welt` | 4 Befunde, u. a. Macchia Hang 18,6 statt 14,5 | die Küsten-Archetypen verstimmen die Regionseichung — in CLAUDE.md beschrieben |
| `pipeline_outputs` | 5 Befunde | seit 24.08. |
| `settlement_placement` | 6 Befunde, u. a. 2 Städte statt 1 je Kultur | seit 24.08. |
| `weather_temperature_direktnormierung` | 3 Befunde, Skerrheim 7,86 statt 8,60 K | seit 24.08. |

**Die drei Erosionsbefunde sind offen und nicht erklärt.** Hier stand bis
zum 16.09.2026 das Gegenteil: sie seien kein Zufall, weil die Erosionskette
bewusst abgeschaltet sei. Das war **doppelt falsch** (Ticket #28 und #63
haben das unabhaengig voneinander gefunden und behoben): erstens laeuft die
Erosion seit dem 27.08.2026 im Betrieb mit - `EROSION_AKTIV = True` steht in
[`gui/config/value_default.py:1088`](../gui/config/value_default.py#L1088)
und ist die einzig gueltige Aussage zum Schalter. Zweitens stimmte schon die
Zahl nicht: die Tabelle oben listet nur **drei** Erosionsbefunde
(`erosion_field`, `erosion_gpu_parity`, `erosion_quality`), keine vier.

Der Satz hat real etwas angerichtet: solange er dort stand, sah jeder Leser
drei gewollte Fehlschläge statt drei ungeklärter, und niemand fasste sie an.
Der schwerste darunter, der Paritätsbruch mit Faktor 385 zwischen GPU und
CPU, stand damit scheinbar vor einer Reaktivierung statt mitten im laufenden
Betrieb. `erosion_field` und `erosion_gpu_parity` haben eigene, vom Schalter
unabhaengige Erklaerungen; der tatsaechliche Grund fuer `erosion_quality`
bleibt **ungeklaert** - das ist eine offene Aufgabe, kein erledigter oder
bewusster Zustand.

Eine falsche Erklärung ist schlimmer als keine: keine Erklärung lädt zum
Nachsehen ein, eine falsche schliesst die Frage.

## 4. Was neu grün ist

Acht Testdateien sind seit dem 24.08. dazugekommen, alle grün:

```
smoke_test_anzeige_register_3d.py     2.2s
smoke_test_fluss_vorschau.py          neu, 8/8
smoke_test_kontinentform.py           6.4s
smoke_test_kuestengebiete.py         13.5s
smoke_test_region_tab.py             11.3s
smoke_test_regionsfeld.py             1.8s
smoke_test_regionsregler_wirken.py    7.1s
smoke_test_reiter_vertrag.py          5.0s
smoke_test_stufen_schalter.py         (in erosion_hauptschalter aufgegangen)
```

`smoke_test_reiter_vertrag.py` ist der Wächter für den Fehler vom 26.08.,
bei dem ein abstürzender Reiter die ganze Leiste verschob — er zählt die
fertige Reiterleiste durch, nicht nur die Importe.

## 5. Die größte Lücke bleibt dieselbe

**Die 3D-Ansicht ist von keinem Test erfasst, der wirklich zeichnet.**
`layer_2d_3d_parity` prüft die Registrierung und die Farbtafel — also
genau die Buchhaltung, die diesmal den Fehler fand. Ob das Bild danach
richtig aussieht, prüft nach wie vor niemand außer dir.
