# 11 — Gelände und Küste

Sollform des Geländes am Übergang von Land zu Wasser: die Küste als Vektorbeschreibung, ihre Profile, die Mischung mit dem Rauschgelände,
die beiden Abtastungen der Höhenfunktion und der Schnitt des 3D-Netzes entlang der Nullkontur. Flüsse, Erosion, Seen und Meerestiefe stehen
in `12_WASSER.md`. Beschrieben ist das Ziel, nicht der Bauzustand.

## 1. Grundgedanke und die zwei Koordinaten

Das Gelände an der Küste ist eine **Wellenform**, die vom Wasser landeinwärts läuft, sich **entlang** der Küste langsam verändert, und mit
dem Rauschgelände **gemischt** statt es zu ersetzen. Alles wird in zwei Koordinaten ausgedrückt:

| | |
|---|---|
| **d** | Abstand zur Wasserlinie, in Metern, landeinwärts positiv |
| **s** | Position **entlang** der Küste (Bogenlänge), in Metern |

Die Küste ist eine Vektorbeschreibung, keine Rasteroperation. Sie ersetzt gedanklich `_kuesten_umformen()` in `core/terrain_weltkarte.py`,
das dieselbe Aufgabe auf Pixelmasken löst und dadurch auflösungsabhängig ist: 34.7 m statt
11.9 m mittlere Abweichung zwischen 256 und 512 px.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „0. Der Grundgedanke in einem Satz".*

## 2. Konturen, Stationen, Sweep

**Die Küstenlinie ist eine Polylinie, kein Pixelband.** Sie kommt aus Marching Squares auf der Nullhöhe (`kuestenlinien()` in
`gui/widgets/kuesten_mesh.py`), das **zwischen** den Pixeln interpoliert; darin liegt die Rasterfreiheit, die alles Weitere erbt. Auf der
Linie stehen **Stationen** in festem Meterabstand (100 m); jede trägt Position und Bogenposition *s*, ihre Region, die lokale Rohgeländehöhe
(5×5-Mittel) als Referenz und den zugeteilten **Küstentyp** samt Profil. **Der Sweep** wird nicht als Geometrie ausgewertet, sondern als
Feld: ein KDTree über dicht abgetastete Linienpunkte liefert zu jedem Punkt *d* und *s*, daraus werden die Profilwerte interpoliert —
geometrisches Sweepen scheidet aus, weil sich die Querprofile in engen Buchten innen schneiden (Offsetkurven-Problem). Geschlossene Konturen
müssen **zyklisch** umlaufen, sonst entsteht an der Startstelle einer Insel eine Naht.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „1. Konturen und Sweeps".*

## 3. Profil in Metern, feste Zonen, Archetypen

Das Profil ist **in Metern** angegeben, nicht normiert: `MESS_PROFIL_M_JE_ARCHETYP` hält 19 Stellen alle 50 m bis 900 m. **Die Höhe steht im
Profil** — keine getrennte Zielhöhe, kein Katalogboden, keine Streckung auf eine Regions-Reichweite. Die Zonen sind fest, nicht je Region
verschieden:

| Zone | |
|---|---|
| 0–350 m landeinwärts | volles Profil |
| 350–500 m landeinwärts | Übergang |
| 0 bis −200 m | Auslauf ins Meer |

Es gibt **27 Archetypen, jeder aus seiner eigenen Vorbildküste** (`tools/archetyp_vorbilder.py`); der Küsteneinfluss wirkt nur **innerhalb
der eigenen Landmasse** (`ndimage.label`). Begründung: die frühere Streckung zerstörte das Verhältnis, um das es geht — die
Weissmeer-Flachküste stand bei 187 m statt 15–30 m. Maßstab ist `tests/smoke_test_kuestenprofiltreue.py`.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „ACHTUNG: Abschnitte zu Reichweite und Profilform gelten nicht mehr".*

## 4. Küstenabschnitte: Segmente statt Dauermischung

Zusammenhängende Stationen gleichen Typs bilden ein **Segment**. **Im Inneren eines Segments ist der Typ rein**; nur an den Segmentgrenzen
liegt eine **Übergangszone fester Breite**, in der die beiden Nachbartypen überblenden — nicht „zwischen je zwei Stationen interpolieren",
denn dann wäre jeder Punkt eine Mischung.

| Größe | Vorschlag | Bedingung |
|---|---|---|
| Stationsabstand | 100 m | kleiner als der kleinste unterscheidbare Abschnitt |
| Übergangsbreite | 200–300 m | |
| Mindest-Segmentlänge | 3–4× Übergangsbreite | sonst wird der Typ nie rein sichtbar |

Läufe unter der Mindestlänge werden in den Nachbarn eingeschmolzen; eine kleine Insel bekommt so von selbst genau einen Typ. Die
Überblendkurve ist **smoothstep-artig, nicht linear** — linear gäbe einen Knick in der Ableitung, sichtbar als Kante längs der Küste. Die
**Typenzahl je Region** folgt der Küstenlänge: `max(1, Küstenlänge ÷ Ziellänge je Typ)`.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „2. Küstenabschnitte: Segmente statt Dauermischung".*

## 5. Formklassen: Insel, Halbinsel, Festland

Die Zuordnung kommt aus einer **Formanalyse**, nicht aus einer Handliste: die Landfläche wird schrittweise nach innen versetzt
(`shapely.buffer(-d)` in Stufen von 25–50 m).

| Ereignis | Bedeutung |
|---|---|
| Teilfläche **verschwindet** bei Tiefe *k* | dieses Stück ist 2·*k* dick |
| Teilfläche **spaltet sich** bei Tiefe *k* | dort liegt ein Hals der Breite 2·*k* |
| Teilfläche **bleibt** | kontinentaler Kern |

Daraus je Küstenabschnitt drei Zahlen: **Hinterlandtiefe**, **Halsbreite**, **Zugehörigkeit**. Der Katalog ist nach **Tiefenklasse**
sortiert — ein Abschnitt mit 50 m Hinterland kann eine tiefe Fjordwand nicht ziehen; damit wird die Maßstabsfrage eine Katalogspalte statt
ein Sonderfall im Code. Kleinteiligkeit ist häufig, aber flächenmäßig nebensächlich (0.8 %): sie darf wenig Rechenzeit kosten.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „3. Formklassen: Insel, Halbinsel, Festland".*

## 6. Kleine Inseln zur Mitte schließen

Die Wellenform zu **stauchen** ergäbe eine Miniaturlandschaft, ein Modell der Insel ihrer selbst; sie **abzuschneiden** ergäbe einen
Kegelstumpf. Richtig ist ein **erzwungenes Schließen zur Mitte**: jeder Punkt braucht neben *d* eine zweite Zahl, **d_max**, die größte
Küstenentfernung *seiner Landmasse*.

```
t = d / d_max        t = 0 an der Küste,  t = 1 auf der Medialachse
h = h_welle(d) · (1 − σ(t))  +  h_kamm · σ(t)
```

σ(t) setzt erst spät ein (etwa ab t = 0.6) und erreicht bei t = 1 den Wert 1; `h_kamm` ist der Wert, den die Wellenform bei *d_max* hätte.
Das Ergebnis nähert sich der Medialachse **mit Steigung null** — eine glatte Kuppe statt einer Spitze, ohne Stauchung. **d_max wird je
Landmasse geführt**: für kleine Inseln exakt, für das Festland grob, wo es ohnehin nicht greift (d_max 2750 m). **Zusätzlich muss die
Wellenform selbst mit Steigung null enden**, sonst entsteht auch mit Schließen ein Grat — eine Bedingung an den Katalog, **beim Laden zu
prüfen**.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „4. Kleine Inseln: nicht stauchen, sondern zur Mitte schließen".*

## 7. Wo zwei Küsten sich treffen

Betroffen sind die **zwei Seiten einer Insel oder Halbinsel** und **benachbarte Abschnitte an einem Hals**, beide mit derselben Antwort.
**Nicht: der nächste gewinnt** — das ist eine Voronoi-Aufteilung, und auf der Medialachse kippt die Zuordnung von einer Seite zur anderen
(gemessen an einer Landzunge: 292 m Höhensprung auf der Achse). **Sondern: gewichtet mischen** über die *k* nächsten Küstenabschnitte, zwei
reichen fast immer, drei an Ecken:

```
h = Σᵢ wᵢ · zielᵢ · welleᵢ(dᵢ / reichweiteᵢ)  /  Σᵢ wᵢ
wᵢ = (1 − dᵢ / reichweiteᵢ)²
```

Derselbe Querschnitt sinkt damit auf 3.9 m. Entscheidend ist die Form: ein **gewichteter Mittelwert** ist zwischen kleinstem und größtem
Beitrag beschränkt, Beiträge mehrerer Küsten können sich konstruktiv **nicht aufsummieren** — strukturell ausgeschlossen, nicht bloß
unwahrscheinlich.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „5. Zwei Küsten, die sich treffen".*

## 8. Der Katalog der Wellenformen

**Gespeichert werden Kontrollpunkte** (8–16), nicht die dichte Abtastung: die Datei hält die *Absicht*, die Arbeitsdarstellung wird beim
Laden daraus gerechnet. Format JSON im Repo, mit Formatversion. Neben der Höhenkurve trägt eine Wellenform den Kanal `t_norm` — die
**Rauschtransparenz** (Abschnitt 11). **Vier Formfamilien** ergaben sich aus 19 vermessenen Vorbildküsten:

| Familie | Vorbilder | Kennzeichen |
|---|---|---|
| **Treppe** | Moher, Rügen, Kap Kaliakra | Sprung in < 250 m, dann Plateau |
| **Rampe** | Lofoten, Geiranger, Stromboli | gleichmäßig über die volle Strecke |
| **Flach** | Stockholmer Schären | 2–36 m, kein Anstieg, nur Rauschen |
| **Buckel** | Milos, Capri, Santorini | steigt an und fällt wieder |

Eine einzige monoton sättigende Kurve der Bauart `ziel · (1 − e^(−d/skala))` kann davon nur „Rampe" ausdrücken.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „6. Die Vektoren selbst".*

## 9. Eine Höhenfunktion, zwei Abtaster

Es gibt genau **eine** Funktion `hoehe(x, y, …)`, die beliebige Fließkommakoordinaten nimmt.

| | wird abgetastet auf | ergibt |
|---|---|---|
| **A** | — | Basisgelände, ohne Küstenformung |
| **B** | Pixelmitten | Rasterkarte für Hydrologie, Biome, Siedlung, Export |
| **Mesh** | freie Vertexpositionen | die sichtbare Geometrie |

Der Unterschied zwischen B und Mesh ist **ausschließlich die Abtastdichte**, kein Modellunterschied; mit demselben Parameter liefern beide
Abtaster **bitgleiche** Ergebnisse. **Der eine erlaubte Unterschied** ist die Mindest-Anstiegsstrecke: das Raster bekommt 2·mpp (feiner kann
ein Gitter eine Klippe nicht zeigen), das Mesh einen kleinen festen Wert. Genau dieser Parameter erzeugt die Rastertreppe — er muss benannt
und einstellbar sein, nicht implizit; an freien Punkten steht damit Höhendetail (p99 29–79 m), das B nicht hergibt.

**Die Küstenlinie legt A fest, nicht die Welle.** Die Welle formt nur *dahinter*; sonst verschöbe jede Formänderung den Wasseranteil je
Region und entwertete die Regionseichung. Zugesichert und geprüft: **0 Pixel** wechseln durch die Küstenformung die Seite. A darf den alten
`_kuesten_umformen()`-Pass **nicht** mehr enthalten, sonst formen die Vektoren eine bereits geformte Küste ein zweites Mal.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „7. Die zwei Heightmaps".*

## 10. Das Mesh, befreit vom Gitter

Die Höhe vom Raster zu lösen genügt nicht — die **Linie** muss es auch, sonst sitzen die Vertices weiter auf Gitterpunkten. **Verfahren:**
das Gitter wird zellweise entlang der Nullkontur **geschnitten**; wo die Linie eine Zellkante kreuzt, entsteht ein Vertex an der
interpolierten Stelle, und die Zelle wird so zerlegt, dass die Küstenlinie eine echte Dreieckskante wird. Das kommt **ohne Constrained
Delaunay** aus: auf einer gemeinsamen Kante rechnen beide Nachbarzellen denselben Schnittpunkt aus denselben zwei Eckhöhen, die Naht ist per
Konstruktion dicht. Der Umlauf `c00 e0 c10 e1 c11 e2 c01 e3` liefert Land- und Seeteilfläche direkt; nur der **Sattel** (diagonale Ecken
gleicher Seite) ist mehrdeutig und wird über den Mittelwert der vier Ecken entschieden. Zielwerte: nahezu alle Konturvertices liegen
**nicht** auf einer Pixelecke, Höhen exakt 0, keine Kante an mehr als zwei Dreiecken, Fläche exakt erhalten, Mehrkosten rund +3 % Dreiecke.
Umgesetzt in `gui/widgets/kuesten_schnitt.py`, geprüft von `tests/smoke_test_kuesten_schnitt.py`.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „8. Das Mesh, befreit vom Gitter".*

## 11. Rauschgelände und Küste zusammen

**Die Küste ersetzt das Rauschen nicht, sie überblendet es** — wieviel durchkommt, sagt die Wellenform über ihren Kanal `t_norm`: `h =
mische( rauschen(p), küstenbeitrag(p), t_norm )`. Eine Klippenwand ist glatt (Transparenz 0), das Hinterland voll strukturiert (Transparenz
1); als fester quadratischer Verlauf wirkt die Küste „aufgeklebt", als gestaltbarer Kanal „gewachsen", und der Kanal ist das Ziel. **Zur
Seeseite** braucht es keine glatte Fortsetzung: die Küstenfunktion läuft aus (`d < 0` ⇒ kein Beitrag), die Meerestiefe kommt aus einer
eigenen, nicht-iterativen Formel über den Seegrad (`_seetiefe_aus_archetyp()`, siehe `12_WASSER.md`). **Zu verhindern ist ein
Wiederanstieg** hinter der Küste — eine Stufe, die vom Wasser weg erst fällt und dann wieder steigt: eine Monotoniebedingung, kein
Glättungsproblem.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „9. Wie Rauschgelände und Küste zusammengehen".*

## 12. Varianz — damit nicht jede Insel gleich aussieht

**Nicht ein zufälliger Ausschnitt aus den Messdaten:** 2 km Moher auf 200 km Küste wäre dutzendfache Wiederholung, und Wiederholung sieht
man. **Sondern drei Sorten Variation:** (1) **Typwahl** je Insel/Abschnitt, aus dem Seed; (2) **Skalare** innerhalb der gemessenen Spannen
(Moher: Zielhöhe 141–185 m); (3) **Änderungsrate längs** aus der Messung, 20 m/km im Median. Erzeugt wird die Längsvariation entweder
autoregressiv aus Änderungsrate und Autokorrelationslänge oder aus einer je Typ gespeicherten **normierten Längs-Signatur** (Zielhöhe über
die Abschnittslänge, 0..1), auf die Länge gestreckt.

*Herkunft: docs/KUESTENMODELL.md (Stand 2026-08-18), Abschnitt „10. Varianz — damit nicht jede Insel gleich aussieht".*

## Was nicht mehr gilt

Aufgehoben am **2026-08-24** (Warnkasten in `docs/KUESTENMODELL.md`, Zeilen 5–30) — nicht wieder einbauen:

* **Profilform normiert auf 0..1, auf die Reichweite gestreckt** → Profil in Metern, `MESS_PROFIL_M_JE_ARCHETYP`, §3.
* **Zielhöhe aus `hinterland × überhöhung + Sockel`, Katalogwert als Boden** → entfällt, die Höhe steht im Profil, §3.
* **Reichweite je Region (110–349 m), Plateau bei 60 %** → feste Zonen 0–350 m / 350–500 m / 0 bis −200 m, §3.
* **Ein Formprofil je Region auf drei Archetypen** → 27 Archetypen aus je eigener Vorbildküste, §3.
* **Küsteneinfluss beliebig weit** → begrenzt auf die eigene Landmasse (`ndimage.label`), §3.

## Offene Fragen

1. **Quelle Zeilen 224–226:** die Mischformel rechnet mit `zielᵢ`, `reichweiteᵢ` und einer auf `dᵢ/reichweiteᵢ` normierten
   Welle — alles drei ist seit 2026-08-24 aufgehoben. Abschnitt 7 gibt sie unverändert wieder; wie die Gewichte in der
   Meter-Darstellung lauten, sagt die Quelle nicht.
2. **Quelle Zeile 10 gegen 223–233:** der Warnkasten nennt als weitergeltend die „Max-Normalisierung beim Mischen mehrerer
   Küsten", der Abschnitt selbst beschreibt einen **gewichteten Mittelwert**.
3. **Quelle Zeilen 192–194:** dass das Schließen zur Inselmitte auf dem Festland nichts tut, ist über „Reichweite höchstens
   700 m" begründet; mit den festen Zonen (max. 500 m) bleibt es vermutlich richtig, ist aber unbelegt.
4. **Quelle Zeilen 200–203:** die Bedingung „Wellenform endet mit Steigung null" war an die normierte Achse (u = 1) gebunden;
   in der Meter-Darstellung ist offen, an welcher Stelle (350 m, 500 m, 900 m) sie zu prüfen ist.
5. **Quelle Zeile 73** nennt als Stationsinhalt „Wellenform, Reichweite, Zielhöhe" — zwei davon gibt es nicht mehr; welche
   Felder eine Station heute trägt, ist aus `core/vektor_kueste.py` zu bestätigen.
