# Das Küstenmodell — Funktionsweise

Stand 2026-08-18, **überholt in wesentlichen Teilen am 2026-08-24**.

> ## ACHTUNG: Abschnitte zu Reichweite und Profilform gelten nicht mehr
>
> Am 2026-08-24 wurde das Modell umgebaut. Wer die Küste verstehen will,
> muss die folgenden Unterschiede kennen; alles andere in diesem Dokument
> (die zwei Koordinaten d und s, die Konturextraktion, die
> Max-Normalisierung beim Mischen mehrerer Küsten) gilt weiter.
>
> **Was sich geändert hat:**
>
> | bis 2026-08-18 | seit 2026-08-24 |
> |---|---|
> | Profilform normiert auf 0..1, dann auf die Regions-Reichweite **gestreckt** | Profil in **METERN**, `MESS_PROFIL_M_JE_ARCHETYP`, 19 Stellen alle 50 m bis 900 m |
> | Zielhöhe aus `hinterland × überhöhung + Sockel`, Katalogwert als Boden | Die Höhe **steht im Profil**. Zielhöhe, Katalogboden und Streckung entfallen |
> | Reichweite je REGION (110–349 m), Plateau bei 60 % | Feste Zonen: 0–350 m volles Profil, 350–500 m Übergang, 0 bis −200 m ins Meer |
> | Ein Formprofil je Region, auf drei Archetypen aufgeteilt | **27 Archetypen, jeder aus seiner eigenen Vorbildküste** (`tools/archetyp_vorbilder.py`) |
> | Küsteneinfluss beliebig weit | Nur **innerhalb der eigenen Insel** (`ndimage.label`) |
>
> **Warum:** die Streckung zerstörte genau das Verhältnis, um das es geht.
> Die Weissmeer-Flachküste stand bei 187 m, ihre Vorlage sagt 15–30 m.
> Seit dem Umbau trifft sie auf 3 m genau, der Median über 20 Gruppen
> liegt bei 2.6 m.
>
> **Wo der aktuelle Stand steht:** `core/vektor_kueste.py` (der Code ist
> ausführlich kommentiert, inklusive der widerlegten Ansätze),
> `docs/SITZUNGSLOG.md` Eintrag 2026-08-24 Teil 2, und
> `tests/smoke_test_kuestenprofiltreue.py` als Maßstab.

Entwurf, noch nicht vollständig gebaut — was steht und was
fehlt, sagt jeder Abschnitt am Ende.

Dieses Dokument beschreibt, wie die Küste entsteht: als **Vektorbeschreibung**
statt als Rasteroperation. Es ersetzt gedanklich `_kuesten_umformen()` in
`core/terrain_weltkarte.py`, das dieselbe Aufgabe auf Pixelmasken löst und
dadurch auflösungsabhängig ist (gemessen: 34.7 m mittlere Abweichung zwischen
256 und 512 px gegenüber 11.9 m ohne diesen Pass).

---

## 0. Der Grundgedanke in einem Satz

Das Gelände an der Küste ist eine **Wellenform**, die vom Wasser landeinwärts
läuft, sich **entlang** der Küste langsam verändert, und mit dem Rauschgelände
**gemischt** statt es zu ersetzen.

Daraus folgen zwei Koordinaten, in denen alles ausgedrückt wird:

| | |
|---|---|
| **d** | Abstand zur Wasserlinie, in Metern, landeinwärts positiv |
| **s** | Position **entlang** der Küste (Bogenlänge), in Metern |

Alles Weitere ist eine Funktion dieser beiden.

---

## 1. Konturen und Sweeps

**Die Küstenlinie ist eine Polylinie, kein Pixelband.** Sie kommt aus Marching
Squares auf der Nullhöhe (`kuestenlinien()` in `gui/widgets/kuesten_mesh.py`) —
das Verfahren interpoliert **zwischen** den Pixeln, die Punkte liegen also auf
Bruchkoordinaten. Das ist die Quelle der Rasterfreiheit; alles danach erbt sie.

Auf dieser Linie stehen **Stationen** in festem Meterabstand (100 m). Jede
Station trägt:

* ihre Position und ihre Bogenposition *s*
* die Region, in der sie liegt
* die lokale Rohgeländehöhe (5×5-Mittel) als Referenz
* den zugeteilten **Küstentyp** samt dessen Wellenform, Reichweite, Zielhöhe

**Der Sweep** ist die gedankliche Bewegung der Wellenform entlang dieser
Stationsreihe. Ausgewertet wird er nicht als Geometrie, sondern als Feld: für
einen beliebigen Punkt *p* liefert ein KDTree über dicht abgetastete
Linienpunkte den Abstand *d* und die Bogenposition *s*; daraus werden die
Profilwerte interpoliert.

> **Warum nicht die Profile geometrisch sweepen?** In einer engen Bucht ist der
> Krümmungsradius kleiner als die Reichweite — die Querprofile schneiden sich
> dann auf der Innenseite (das klassische Offsetkurven-Problem). Die
> Feldauswertung über *d* hat dieses Problem nicht, weil sie nie zwei Profile
> geometrisch nebeneinanderlegt, sondern ihre Beiträge gewichtet mischt
> (Abschnitt 5).

**Steht:** Linienextraktion, Stationen, KDTree, Bogenparametrisierung
(`core/vektor_kueste.py`).
**Fehlt:** geschlossene Konturen laufen noch nicht zyklisch um — an der
willkürlichen Startstelle einer Insel entsteht dadurch eine Naht.

---

## 2. Küstenabschnitte: Segmente statt Dauermischung

Jede Station bekommt einen Typ. Zusammenhängende Stationen gleichen Typs
bilden ein **Segment**.

**Im Inneren eines Segments ist der Typ rein.** Nur an den Segmentgrenzen
liegt eine **Übergangszone fester Breite**, in der die beiden Nachbartypen
überblenden.

Das ist bewusst *nicht* „zwischen je zwei Stationen interpolieren" — dann wäre
jeder Punkt eine Mischung und kein Typ je rein zu sehen. Die Vorgabe war:
schnell übergehen, aber definierte Küsten behalten.

Drei Zahlen müssen zueinander passen:

| Größe | Vorschlag | Bedingung |
|---|---|---|
| Stationsabstand | 100 m | kleiner als der kleinste unterscheidbare Abschnitt |
| Übergangsbreite | 200–300 m | |
| Mindest-Segmentlänge | 3–4× Übergangsbreite | sonst wird der Typ nie rein sichtbar |

Läufe unter der Mindestlänge werden in den Nachbarn eingeschmolzen. **Damit
erledigt sich die kleine Insel von selbst:** ihre gesamte Kontur ist kürzer
als die Mindestlänge, sie bekommt genau einen Typ.

Die Überblendkurve ist smoothstep-artig, nicht linear — linear gäbe an beiden
Enden der Übergangszone einen Knick in der Ableitung, sichtbar als Kante längs
der Küste.

**Wieviele Typen je Region?** Nicht fest. Die Küstenlänge je Region schwankt
gemessen um Faktor 50 (Nevadin 1.0 km, Thalassia 54.6 km). Die
Typenzahl folgt daher der Küstenlänge: `max(1, Küstenlänge ÷ Ziellänge je Typ)`.

**Fehlt:** die Segmentbildung ganz. Heute wird zwischen Stationen durchgehend
interpoliert.

---

## 3. Formklassen: Insel, Halbinsel, Festland

Die Zuordnung kommt aus einer **Formanalyse**, nicht aus einer Handliste.

Verfahren: die Landfläche wird schrittweise nach innen versetzt
(`shapely.buffer(-d)` in groben Stufen, 25–50 m). Dabei passieren drei Arten
von Ereignissen, und jede sagt etwas über die Form:

| Ereignis | Bedeutung |
|---|---|
| Teilfläche **verschwindet** bei Tiefe *k* | dieses Stück ist 2·*k* dick |
| Teilfläche **spaltet sich** bei Tiefe *k* | dort liegt ein Hals der Breite 2·*k* |
| Teilfläche **bleibt** | kontinentaler Kern |

Daraus je Küstenabschnitt drei Zahlen: **Hinterlandtiefe**, **Halsbreite**,
**Zugehörigkeit** (Insel / Halbinsel / Festland).

Gemessen an der echten Karte: 58 getrennte Landmassen, 55 davon dünner als
400 m, zusammen aber nur 0.8 % der Fläche. Die Kleinteiligkeit ist also häufig,
aber flächenmäßig nebensächlich — sie darf Rechenzeit kosten, aber nicht viel.

**Wozu das gebraucht wird:** der Katalog ist nach **Tiefenklasse** sortiert.
Ein Abschnitt mit 50 m Hinterland kann die Fjordwand mit 700 m Reichweite gar
nicht ziehen. Damit wird die Maßstabsfrage eine Katalogspalte statt ein
Sonderfall im Code.

**Fehlt:** komplett. Die Ringe sind einmal als Machbarkeit gemessen
(2.7 s für 30 Ringe), aber nicht angebunden.

---

## 4. Kleine Inseln: nicht stauchen, sondern zur Mitte schließen

Das naheliegende Falsche: die 400-m-Wellenform auf eine 100-m-Insel stauchen.
Dann bekommt die Insel eine Miniaturlandschaft mit Düne und Hügel — sie sieht
aus wie ein Modell ihrer selbst.

Das ebenso Falsche: die Wellenform abschneiden. Dann wird die Insel ein
**Kegelstumpf**, weil nur der steile Anfangsteil der Kurve benutzt wird.

**Richtig ist ein erzwungenes Schließen zur Mitte.** Dafür braucht jeder Punkt
neben *d* eine zweite Zahl: **d_max**, die größte Küstenentfernung *seiner
Landmasse*. Daraus

```
t = d / d_max        t = 0 an der Küste,  t = 1 auf der Medialachse
```

und ein Schließgewicht σ(t), das erst spät einsetzt (etwa ab t = 0.6) und bei
t = 1 den Wert 1 erreicht:

```
h = h_welle(d) · (1 − σ(t))  +  h_kamm · σ(t)
```

`h_kamm` ist der Wert, den die Wellenform bei *d_max* hätte. Das Ergebnis
nähert sich der Medialachse **mit Steigung null** — eine glatte Kuppe statt
einer Spitze, ohne dass die Wellenform gestaucht wurde.

**Warum das auf dem Festland nichts tut:** dort ist d_max = 2750 m gemessen,
die Reichweite höchstens 700 m. Im ganzen Küstenband ist t < 0.25, σ(t) = 0.
Das Schließen greift nur dort, wo es gebraucht wird.

**d_max je Landmasse** statt punktgenau: für kleine Inseln ist das exakt (eine
Insel, ein Wert), für das Festland ist es grob — aber dort greift es ohnehin
nicht.

**Zusätzlich muss die Wellenform selbst mit Steigung null enden** (bei u = 1).
Eine Kurve, die am Ende noch steigt, erzeugt sonst auch mit Schließen einen
Grat. Das ist eine Bedingung an den Katalog, keine an den Code — beim Laden
zu prüfen.

**Fehlt:** komplett.

---

## 5. Zwei Küsten, die sich treffen

Das betrifft zwei Fälle, die dieselbe Antwort haben:

* die **zwei Seiten einer Halbinsel** oder einer Insel
* **benachbarte Abschnitte** an einem Hals

**Nicht: der nächste gewinnt.** Das ist eine Voronoi-Aufteilung, und auf der
Medialachse kippt die Zuordnung von einer Seite zur anderen. Gemessen an einem
Querschnitt durch eine Landzunge: **292 m Höhensprung** genau auf der Achse.

**Sondern: gewichtet mischen.** Über die *k* nächsten Küstenabschnitte
(zwei reichen fast immer, drei an Ecken):

```
h = Σᵢ wᵢ · zielᵢ · welleᵢ(dᵢ / reichweiteᵢ)  /  Σᵢ wᵢ
wᵢ = (1 − dᵢ / reichweiteᵢ)²
```

Derselbe Querschnitt damit: **3.9 m** statt 292 m, der Steigungssprung fällt
von 57817 auf 13.9.

Das ist ein **gewichteter Mittelwert**, also beschränkt zwischen dem kleinsten
und größten Beitrag. Vektoren können sich konstruktiv **nicht aufsummieren** —
das war eine ausdrückliche Sorge und ist hiermit strukturell ausgeschlossen.

**Steht:** die Interpolation der Profilwerte entlang *s* (Abschnitt 1).
**Fehlt:** die Mischung über mehrere Küsten. Heute gewinnt der nächste.

---

## 6. Die Vektoren selbst

Eine **Wellenform** ist:

| Teil | |
|---|---|
| `h_norm(u)` | Höhenkurve, u = d/Reichweite ∈ [0,1], Wert in [0, ~1] |
| `t_norm(u)` | Rauschtransparenz auf derselben u-Achse |
| `reichweite_m` | Skalar |
| `zielhoehe_m` | Skalar |

**Normiert, mit festen Enden.** (0,0) ist die Wasserlinie, (1,1) die Zielhöhe,
Steigung dort null. Dazwischen frei, auch über 1.0 hinaus (die Düne vor der
Mulde). Die festen Enden werden **erzwungen und beim Laden geprüft**, nicht
gespeichert — stünden sie versehentlich anders in der Datei, hätte man die
Spitze aus Abschnitt 4 zurück, ohne zu wissen warum.

**Warum normiert:** nur so lässt sich eine 180-m-Strandwelle mit einer
700-m-Klippenwelle morphen. Sonst überblendete man Formen verschiedener Länge.

**Gespeichert werden Kontrollpunkte** (8–16), nicht die dichte Abtastung. Die
Datei hält die *Absicht*, die Arbeitsdarstellung wird beim Laden daraus
gerechnet. Format: JSON im Repo, mit Formatversion.

**Vier Formfamilien** ergaben sich aus 19 vermessenen Vorbildküsten:

| Familie | Vorbilder | Kennzeichen |
|---|---|---|
| **Treppe** | Moher, Rügen, Kap Kaliakra | Sprung in < 250 m, dann Plateau |
| **Rampe** | Lofoten, Geiranger, Stromboli | gleichmäßig über die volle Strecke |
| **Flach** | Stockholmer Schären | 2–36 m, kein Anstieg, nur Rauschen |
| **Buckel** | Milos, Capri, Santorini | steigt an und fällt wieder |

**Fehlt:** die Wellenform als Datenstruktur. Heute ist das Profil fest
`ziel · (1 − e^(−d/skala))` — eine einzige monoton sättigende Kurve, die keine
der vier Familien außer „Rampe" ausdrücken kann.

---

## 7. Die zwei Heightmaps

**Eine Höhenfunktion, zwei Abtaster.** Es gibt genau eine Funktion
`hoehe(x, y, …)`, die beliebige Fließkommakoordinaten nimmt.

| | wird abgetastet auf | ergibt |
|---|---|---|
| **A** | — | Basisgelände, ohne Küstenformung |
| **B** | Pixelmitten | Rasterkarte für Hydrologie, Biome, Siedlung, Export |
| **Mesh** | freie Vertexpositionen | die sichtbare Geometrie |

Der Unterschied zwischen B und Mesh ist **ausschließlich die Abtastdichte**,
kein Modellunterschied. Nachgewiesen: mit demselben Parameter liefern beide
Abtaster **bitgleiche** Ergebnisse.

**Der eine erlaubte Unterschied** ist die Mindest-Anstiegsstrecke: das Raster
bekommt 2·mpp (feiner kann ein Gitter eine Klippe nicht zeigen), das Mesh
einen kleinen festen Wert. Genau dieser Parameter erzeugt die Rastertreppe —
er ist jetzt benannt und einstellbar statt implizit.

Gemessen: an freien Punkten stehen p99 29–79 m (max 195 m) Höhendetail, die
eine bilineare Interpolation von B nicht hergibt.

**Die Küstenlinie legt A fest, nicht die Welle.** Die Welle formt nur
*dahinter*. Sonst verschöbe jede Formänderung den Wasseranteil je Region und
entwertete die Regionseichung. Zugesichert und geprüft: **0 Pixel** wechseln
durch die Küstenformung die Seite.

**Steht:** vollständig (`core/vektor_kueste.py`,
`tests/smoke_test_vektor_kueste.py`, 6/6 grün).
**Offen:** A enthält heute noch den alten `_kuesten_umformen()`-Pass, die
Vektoren formen also eine bereits geformte Küste ein zweites Mal.

---

## 8. Das Mesh, befreit vom Gitter

Die Höhe vom Raster zu lösen genügt nicht — die **Linie** muss es auch. Sonst
sitzen die Vertices weiter auf Gitterpunkten und Übersampling macht die Treppe
nur kleiner.

**Verfahren:** das Gitter wird zellweise entlang der Nullkontur
**geschnitten**. Wo die Linie eine Zellkante kreuzt, entsteht ein neuer Vertex
an der interpolierten Stelle; die Zelle wird so zerlegt, dass diese Punkte
Dreiecksecken sind und die Küstenlinie eine echte Dreieckskante wird.

**Warum das ohne Constrained Delaunay auskommt:** auf einer gemeinsamen Kante
rechnen beide Nachbarzellen denselben Schnittpunkt aus denselben zwei
Eckhöhen. Die Naht ist **per Konstruktion** dicht.

Der Umlauf `c00 e0 c10 e1 c11 e2 c01 e3` liefert Land- und Seeteilfläche
direkt; nur der **Sattel** (diagonale Ecken gleicher Seite) ist mehrdeutig und
wird über den Mittelwert der vier Ecken entschieden.

Gemessen bei 384 px: **4240 Konturvertices, 99.7 % nicht auf einer Pixelecke**
(Median-Versatz 0.164 px), Höhen exakt 0, keine Kante an mehr als zwei
Dreiecken, Fläche exakt 383², **+2.9 % Dreiecke**.

Zum Vergleich: Quadtree 0.000004 px, also exakt auf der Ecke.

**Steht:** vollständig (`gui/widgets/kuesten_schnitt.py`,
`tests/smoke_test_kuesten_schnitt.py`, 6/6 grün).

---

## 9. Wie Rauschgelände und Küste zusammengehen

**Die Küste ersetzt das Rauschen nicht, sie überblendet es** — und wieviel
durchkommt, sagt die Wellenform selbst über ihren zweiten Kanal `t_norm(u)`:

```
h = mische( rauschen(p),  küstenbeitrag(p),  t_norm(u) )
```

Eine Klippenwand ist glatt (Transparenz 0), das Hinterland voll strukturiert
(Transparenz 1). Heute ist das eine feste quadratische Kurve; als Kanal wäre
es gestaltbar — und es ist der Unterschied zwischen „aufgeklebt" und
„gewachsen".

**Zur Seeseite** braucht es keine glatte Fortsetzung. Die Küstenfunktion läuft
einfach aus (`d < 0` ⇒ kein Beitrag), und die Meerestiefe kommt aus einer
eigenen, nicht-iterativen Formel über den Seegrad
(`_seetiefe_aus_archetyp()`). **Was verhindert werden muss, ist ein
Wiederanstieg** hinter der Küste — eine Stufe, die vom Wasser weg erst fällt
und dann wieder steigt. Das ist eine Monotoniebedingung auf der Seeseite, kein
Glättungsproblem.

---

## 10. Varianz — damit nicht jede Insel gleich aussieht

Später, aber der Weg steht:

**Nicht ein zufälliger Ausschnitt aus den Messdaten.** 2 km Moher auf 200 km
Küste wäre dutzendfache Wiederholung, und Wiederholung sieht man.

**Sondern drei Sorten Variation:**

1. **Typwahl** je Insel/Abschnitt, aus dem Seed
2. **Skalare** innerhalb der gemessenen Spannen (Moher: Zielhöhe 141–185 m)
3. **Änderungsrate längs** aus der Messung: 20 m/km im Median über 19 km
   Küste. Das ist die Zahl, die sagt, wie stark benachbarte Stationen
   voneinander abweichen dürfen, damit es echt wirkt.

**Kann die Form „erlernt" werden?** Für diesen Zweck braucht es kein Lernen im
engeren Sinn. Ein autoregressives Modell mit der gemessenen Änderungsrate und
Autokorrelationslänge reproduziert den *Charakter* der Längsvariation, ohne
die Messreihe zu kopieren — billig und ausreichend. Wer mehr will, speichert
je Typ eine **normierte Längs-Signatur** (Zielhöhe über die Abschnittslänge,
wieder 0..1) und streckt sie auf die vorhandene Länge. Damit sieht 10 km nie
wie fünfmal dieselben 2 km aus.

---

## Was steht, was fehlt

| Abschnitt | Zustand |
|---|---|
| 1 Konturen, Stationen, Sweep | steht, außer zyklischem Umlauf |
| 2 Segmente und Übergänge | **fehlt** |
| 3 Formanalyse (Insel/Halbinsel/Festland) | **fehlt**, Machbarkeit gemessen |
| 4 Schließen zur Inselmitte | **fehlt** |
| 5 Mischung mehrerer Küsten | **fehlt**, Wirkung gemessen |
| 6 Wellenform als Datenstruktur | **fehlt**, Formfamilien bestimmt |
| 7 Zwei Heightmaps | steht, geprüft |
| 8 Mesh-Schnitt | steht, geprüft |
| 9 Rauschmischung | steht als feste Kurve, Kanal fehlt |
| 10 Varianz | Entwurf |
