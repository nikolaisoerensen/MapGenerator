# MapGenerator — Konzept-Diskussion: Geology als 3D-Gesteinsstapel (Stand 2026-07-22)

## Status: offene Design-Diskussion, keine Entscheidung, keine Umsetzung

Dieses Dokument ist eine Fortsetzung von
`docs/session_review_2026-07-22_geology.md` (5-Fragen-Review, abgeschlossen).
Dort wurde die Idee des Nutzers — Geology als Schichtstapel entlang der
Höhenachse zu modellieren, statt als 2D-RGB-Mischverhältnis + additives
`height_delta` — grob bewertet und als eigene, spätere Umsetzungsrunde
empfohlen. Diese Runde vertieft das Konzept: ein Architektur-Vorschlag
plus ein Katalog offener Fragen, die der Nutzer vor einer Umsetzung
entscheiden muss. **Kein Code wurde in dieser Runde geändert.**

---

## Architektur-Vorschlag (zur Diskussion, nicht final)

### 1. Grundrepräsentation: Höhenfelder statt Voxel-Raster
Kein dichtes 3D-Array (Layer × H × W) nötig. Stattdessen pro Schicht `i`
(0 = älteste/unterste, N = jüngste/oberste) ein 2D-Höhenfeld
`boundary[i](x,y)` = reale Z-Höhe der Schichtobergrenze, undeformiert:

```
boundary[i](x,y) = base_datum + Σ_{j<=i} thickness[j](x,y)
```

`thickness[j]` kann konstant oder leicht lateral variierend sein (offene
Frage A2). Jede Schicht trägt Gesteinstyp, Härte, Farbe — analog zur
Legende im vom Nutzer gezeigten Referenzbild (Holozän → Kristallin).

### 2. Tektonik als EIN gemeinsames Verschiebungsfeld
Statt fünf lose gekoppelter Einzeleffekte (wie heute: Ridge-Warping,
Bevel-Warping, Foliation, Folding, Igneous-Flowing rechnen je eigene,
unabhängige Formeln) wird ein einziges Z-Verschiebungsfeld
`Δz(x,y)` konstruiert, das additiv auf ALLE Schichtgrenzen gleichzeitig
wirkt (starre Verschiebung des ganzen Stapels an jedem Punkt) — plus ein
diskreter Störungs-Versatz-Anteil:

```
Δz(x,y) = tilt(x,y) + fold_broad(x,y) + fold_fine(x,y) + fault_throw(x,y)
```

**Mapping der 5 bestehenden Slider auf das neue Modell:**

| Bisheriger Slider | Rolle im neuen Modell |
|---|---|
| `ridge_warping` | Amplitude der hochfrequenten Komponente `fold_fine` (Detail-Rauheit der Verformung) |
| `bevel_warping` | Breite/Weichheit der Übergangszone an `fault_throw`-Kanten (wie "abgeschliffen" ein Störungsversatz aussieht) |
| `metamorph_folding` | Amplitude der niederfrequenten Komponente `fold_broad` (echte geologische Faltung) |
| `metamorph_foliation` | Wird rein visuell/texturell (feines Streifenmuster in Härte/Farbe metamorpher Zonen), **kein** Höhen-Einfluss mehr — sauberere Trennung als heute, wo Foliation/Folding beide Höhe UND Gesteinsverschiebung mischen |
| `igneous_flowing` | Randrauschen-Stärke der Intrusionskörper-Kontur (siehe Punkt 4) |

Das löst nebenbei den in der letzten Runde gefundenen Key-Mismatch-Bug
strukturell mit, weil die fünf Alt-Parameter nicht mehr getrennt
`.get(key, 0.0)`-gelesen werden, sondern in eine gemeinsame Feld-
Konstruktion einfließen.

### 3. Störungslinien (Faults)
Kleine Anzahl (z.B. 3-8, siehe Frage B5) prozedural erzeugter Linien/
Polylinien mit Vorzeichen-Distanzfeld `d_fault(x,y)`. Pro Fault ein
`throw` (vertikaler Versatz in Metern), optional entlang der Linie
auslaufend (Frage B7). Jeder Pixel wird per "nächstgelegene Fault-Seite"
einem Block zugeordnet (Voronoi-artig), der Block trägt den kumulierten
Versatz seiner Faults; ein schmaler geglätteter Übergang an der
Bruchlinie selbst ist genau das, was `bevel_warping` heute konzeptionell
sein will. Vollständig vektorisierbar (Punkt-zu-Segment-Distanz für
wenige Linien × alle Pixel).

### 4. Igneous Intrusionen statt Flächen-Fließmuster
Statt eines globalen Sinus-Musters: eine Handvoll lokal begrenzter
"Intrusionskörper" (Blob-Position, -Größe, Rand-Rauschen), die die
darunterliegende Schicht lokal durchschlagen — analog zu den dunkelgrünen
Basalt-Flecken im Referenzbild, die quer zu den Sedimentbändern liegen.
`igneous_flowing` wird zur Randrauschen-Stärke dieser Blobs.

### 5. Ausbiss-Berechnung (Outcrop)
Für jedes `(x,y)`: die reale (bereits vom Terrain-Generator berechnete)
Geländehöhe `z = terrain_height(x,y)` gegen die deformierten
Schichtgrenzen `boundary_deformed[i](x,y) = boundary[i](x,y) + Δz(x,y)`
vergleichen → welches Intervall `[boundary[i-1], boundary[i]]` enthält
`z`? Bei N~10 Schichten ist das N Vergleichsmasken über das ganze Array —
vollständig vektorisiert, kein Pixel-Loop, kein echtes 3D-Sampling nötig.
Optional: schmale Übergangszone an der Schichtgrenze weich blenden für
nicht-scharfkantige Ausbisslinien.

`height_delta` für die kombinierte Heightmap bleibt (wie heute)
additiv — ist aber jetzt exakt dasselbe `Δz`-Feld, das auch den Ausbiss
bestimmt. Das behebt die heutige Inkonsistenz, dass Höhenwirkung und
Gesteinsverschiebung aus unabhängigen Formeln stammen und macht Effekte
diagnostisch sichtbar (ein Slider verändert EIN Feld, das man direkt
anzeigen kann).

### 6. Sediment-Überlagerung
Nach der Ausbiss-Berechnung: junge Sedimente (Alluvium) in Senken/Tälern
überlagern, erkannt über einen Relief-Proxy (z.B.
`terrain_height - gaussian_blur(terrain_height, groß)`, negative Werte =
Senke) oder — falls zeitlich/pipeline-technisch möglich — echte Water-
Flow-Accumulation-Daten (Frage D12 klärt, ob das zum Berechnungszeitpunkt
überhaupt vorliegt).

### 7. Performance
Alle Schritte sind elementweise NumPy-Array-Operationen über wenige
Schichten/Faults/Intrusionen — strikt billiger als die heutigen 6
Python-Pixel-Loops, und von Anfang an GPU-Shader-tauglich (jeder Schritt
ist eine einfache elementweise Operation, gut portierbar für den in der
letzten Runde gewünschten echten GPU-Pfad).

### 8. LOD-Integration: "Wachsen über LOD" + LOD-Zahl-Unabhängigkeit
Vorschlag: jede Δz-Komponente bekommt eine **feste Ziel-Wellenlänge in
km** (nicht in Pixeln, nicht an eine LOD-Nummer gekoppelt). Bei jeder
Auflösung wird geprüft, wie viele Pixel pro Wellenlänge zur Verfügung
stehen (`px_per_cycle = (width_px / map_distance_km) * wavelength_km`).
Unter ~2 px/Zyklus (Nyquist-Grenze) trägt die Komponente ~nichts bei,
darüber blendet sie über eine kurze Rampe (z.B. 2→8 px/Zyklus) weich ein.

Das löst beide Teile der Nutzer-Anforderung gleichzeitig:
- **"Wachsen über LOD"** passiert automatisch: grobe LODs zeigen nur, was
  bei ihrer Pixeldichte überhaupt auflösbar ist (Tilt/breite Faltung),
  feine LODs schalten schrittweise Ridge-Detail/Fault-Bevel/Intrusions-
  Rand-Rauschen dazu — kein manuelles `lod_level/5.0`-Gating wie heute
  nötig.
- **LOD-Zahl-Unabhängigkeit**: Da das Ein-/Ausblenden einer Komponente
  ausschließlich von der aktuellen Auflösung/Weltgröße abhängt (nicht
  davon, die wievielte LOD-Stufe von wie vielen insgesamt gerade aktiv
  ist), ergibt ein direkter Sprung von LOD1 auf die Ziel-Auflösung
  exakt dasselbe Feld wie das schrittweise Durchlaufen aller
  Zwischen-LODs — die Felder sind reine, deterministische Funktionen von
  (Weltkoordinate in km, aktuelle Auflösung), nicht von der
  Berechnungshistorie. Das ersetzt auch das heutige
  `scipy.ndimage.zoom`-Fortschreiben von `previous_height_delta` zwischen
  LOD-Stufen (das IST pfadabhängig) durch direktes Neu-Auswerten der
  Felder an den neuen Sample-Punkten.

### 9. GUI-Konsequenzen (grob, siehe Fragen G17-G19)
- Neue Diagnose-Anzeigemodi: Δz-Komponenten einzeln (Tilt/Fold-breit/
  Fold-fein/Fault-Throw/Fault-Bevel/Intrusions-Rand), damit jeder Slider
  isoliert sichtbar wird — direkte Antwort auf die Kernbeschwerde des
  Nutzers.
- Möglicher neuer Diagnose-View: vertikaler Schichtstapel-Querschnitt an
  wählbarer XY-Position (zeigt Schichten + Terrainhöhe als Linie).
- Slider-Neugestaltung nötig — offen, wie stark (Frage G17).

---

## Offene Fragen an den Nutzer (20)

### A) Schichtstapel-Modell
1. Feste kuratierte Schichtenliste (wie die Legende im Referenzbild:
   Holozän/Pleistozän/Basalt/Jungtertiär/Alttertiär/Kreide/Jura/Keuper/
   Muschelkalk/Buntsandstein/Zechstein/Rotliegend/Paläozoikum/Kristallin),
   oder parametrische Anzahl (Slider "Anzahl Schichten")?
2. Sollen Schichtdicken über die Karte konstant sein oder lateral
   variieren (z.B. dünner am Rand, dicker in "alten Becken")? Falls
   variabel — wodurch gesteuert?
3. Bleiben die 3 heutigen Grundkategorien (Sediment/Igneous/Metamorphic)
   pro Schicht erhalten, oder bekommt jede Schicht einen eigenen freien
   Typ/Farbe/Härte (wie die 13 Legendeneinträge im Referenzbild)?

### B) Tektonik / Verformung
4. Tilt global einheitlich (eine Richtung/Stärke für die ganze Karte)
   oder regional mit mehreren Zentren?
5. Wie viele Störungslinien typischerweise, und eher zufällig verteilt
   oder in Mustern (parallel wie im Grabenbruch-Referenzbild, radial,
   netzartig)?
6. Störungen als gerade Linien oder gekrümmte/verzweigte Polylinien?
7. Versatz je Störung konstant entlang der Linie, oder an den Enden
   auslaufend?
8. Soll "Ridge Warping" als eigener benannter Effekt bestehen bleiben,
   oder wie vorgeschlagen zur Detail-Komponente von Folding werden?

### C) Igneous / Metamorphic
9. Vulkanische Intrusionen als diskrete Blob-Körper (wie im
   Referenzbild), oder weiterhin ein flächendeckendes Fließmuster?
10. Metamorphose nur als Funktion von Störungsnähe, oder auch durch
    Intrusionsnähe (Kontaktmetamorphose)?
11. Foliation rein visuell/texturell ohne Höhenwirkung (Vorschlag oben),
    oder soll sie weiterhin auch Höhe beeinflussen?

### D) Sediment / Erosion
12. Reicht ein einfacher Relief-/Krümmungs-Proxy für die Sediment-
    Zuweisung in Senken, oder soll das an echte Water-Flow-Accumulation-
    Daten gekoppelt werden (Reihenfolge Water/Geology in der Pipeline
    müsste dafür geprüft werden)?

### E) LOD / Skalierung
13. Sollen ALLE Verformungskomponenten über feste km-Wellenlängen
    parametrisiert werden (Vorschlag), oder soll bewusst etwas weiterhin
    proportional zur Kartengröße statt zu echten km skalieren?
14. Reicht das Nyquist-basierte Fade-in als "Wachsen über LOD", oder soll
    zusätzlich bei groben LODs bewusst eine reduzierte Schichtenzahl (nur
    2-3 statt aller) angezeigt werden?

### F) Performance / Architektur
15. Von Anfang an GPU-Shader-tauglich vektorisiert umsetzen, auch bevor
    der eigentliche GPU-Pfad gebaut wird?
16. `height_delta` weiterhin separates additives Feld (kompatibel mit
    `get_terrain_data_combined()`), oder engere Terrain-Kopplung als
    denkbare noch spätere Option?

### G) GUI / Slider
17. Dürfen die 5 heutigen Slider-Namen (`ridge_warping` etc.) komplett
    durch neue, physikalisch benannte Regler ersetzt werden, oder sollen
    Namen/Ranges möglichst erhalten bleiben (auch wenn intern anders
    verrechnet)?
18. Neuer Schichtstapel-Querschnitt-Diagnose-View gewünscht, oder reichen
    2D-Draufsicht-Diagnosemodi pro Δz-Komponente?
19. Härte pro Einzelschicht einstellbar (mehr Slider), oder bei wenigen
    Kategorie-Reglern bleiben (wie heute: 3 Gesteins-Härten)?

### H) Migration / Scope
20. Komplettersatz des heutigen Modells (Breaking Change), oder erst als
    paralleler "v2"-Modus neben dem bestehenden Rechner, umschaltbar bis
    er sich bewährt hat?

---

## Nächster Schritt
Nutzer beantwortet die obigen Fragen (ganz oder teilweise). Erst danach:
konkreter Umsetzungsplan (Datenmodell, Dateien, Migrationsschritte,
GUI-Änderungen) für eine dedizierte Implementierungsrunde.
