# MapGenerator — Geology-Review-Runde (Stand 2026-07-22)

## Zweck dieses Dokuments
Zweiter Durchgang der Generator-Review-Reihe (Terrain → **Geology** →
Weather → Water → Biome → Settlement). Reine Analyse-Runde, **keine
Code-Änderungen**. Grundlage für spätere, separat zu beauftragende
Umsetzungs-Runden. Siehe `docs/session_handover_2026-07-22_terrain_review.md`
für die Terrain-Runde und das allgemeine Vorgehen (5 Standardfragen je
Generator).

Zusätzlich zu den 5 Standardfragen wurde in dieser Runde eine
Architektur-Idee des Nutzers bewertet: ein "3D-Gesteinsstapel"-Modell
(Schichten entlang der Höhenachse → Tektonik verformt den Stapel →
Ausbisslinie = Schnitt mit der realen Geländehöhe → Sedimente in Senken →
Multiplikation mit Hillshade) als möglicher Ersatz für das jetzige Modell
(2D-RGB-Gesteins-Mischverhältnis + additives `height_delta`).

Codebasis: Haupt-Checkout (`C:\Lokale Dateien\Projects\Python\MapGenerator`,
Branch `main`), analog zur Terrain-Runde. Recherche-Basis: voller Read von
`core/geology_generator.py` (1517 Zeilen), `gui/tabs/geology_tab.py`,
`gui/config/value_default.py` (GEOLOGY-Sektion), `gui/OldManagers/
data_lod_manager.py` (Geology-Methoden), `gui/OldManagers/
calculator_graph.py` und `generation_orchestrator.py`, vollständiger
Shader-Verzeichnis-Scan, Vergleich mit dem Terrain-Skalierungs-Pattern aus
`core/terrain_generator.py`.

---

## Frage 1 — Rechenschritte / Shader

7 Calculator-Nodes in `GeologySystemGenerator.calculate_geology()`
(`core/geology_generator.py:817-887`):
`classify_elevation` → `slope_hardening` → `blend_zones` →
`tectonic_deformation` → `faceted_boundaries` → `mass_conservation` →
`hardness`.

**Kein GPU-Pfad, keine Shader.** `shaders/` hat Unterordner für
Water/Weather/Biome/Settlement/3d_display, aber **keinen**
`shaders/geology/`. Der einzige geology-bezogene Shader-Treffer ist eine
reine Farb-Auswahl-Funktion im 3D-Viewport-Renderer
(`shaders/3d_display/terrain.frag`, `getGeologyColor()`), die nur eine
bereits fertig berechnete CPU-Textur einfärbt — keine eigene Simulation.

`get_generation_info()` (`geology_generator.py:1368`) und der
Modul-Docstring (Zeile 5) behaupten ein "3-stufiges GPU/CPU/Simple-
Fallback-System" — das ist **falsch/veraltet**. Real existiert nur ein
einziges breites `try/except` um `calculate_geology()` (836-887), das bei
**jedem** Fehler auf `_create_fallback_geology_data()` (1281-1336)
springt. `generation_orchestrator.py:1011-1014` bestätigt das selbst im
Kommentar: Geology hat "bisher KEINE Shader-Anbindung implementiert".
`GeologySystemGenerator.__init__()` nimmt nicht mal einen
`shader_manager`-Parameter entgegen.

6 der 7 Stufen sind zusätzlich **nicht vektorisiert** — verschachtelte
reine Python-`for y/for x`-Loops (`classify_elevation`, `slope_hardening`,
`blend_geological_zones`, `normalize_rock_masses`,
`_force_mass_conservation`, `_create_fallback_geology_data`), teils mit
1-3 `OpenSimplex`-Aufrufen pro Pixel. Nur `tectonic_deformation`,
`faceted_boundaries` und `hardness` sind echtes vektorisiertes NumPy.

**Empfehlung für später:** (a) Docstring/`get_generation_info()` an
Realität anpassen, (b) **echten GPU-Compute-Pfad bauen** — sinnvollerweise
erst NACH einer Vektorisierung der 6 Loop-Stufen auf NumPy, weil ein
GPU-Shader ohnehin eine vektorisierte Formulierung jeder Stufe braucht;
die Vektorisierung ist damit ein Zwischenschritt, keine separate
Baustelle.

---

## Frage 2 — Variable vs. fixierte Größen

**8 Slider** (alle in `value_default.py:122-170`, UI in
`geology_tab.py:106-159`):

| Slider | Range | Default |
|---|---|---|
| `sedimentary_hardness` | 1-100 | 30 |
| `igneous_hardness` | 1-100 | 80 |
| `metamorphic_hardness` | 1-100 | 65 |
| `ridge_warping` | 0.0-2.0 | 0.5 |
| `bevel_warping` | 0.0-2.0 | 0.3 |
| `metamorph_foliation` | 0.0-1.0 | 0.4 |
| `metamorph_folding` | 0.0-1.0 | 0.6 |
| `igneous_flowing` | 0.0-1.0 | 0.7 |

**~30 fixierte Konstanten** im Code, u.a.: alle Rausch-Frequenzen (3/2/4
für die drei Zonen, 6 für Ridge, 8 für Bevel, 20/15/12/8 für
Foliation/Folding/Flow, 8+30 für den Facetten-Domain-Warp, 2.5 für die
Hardness-Tier-Maske), alle Schwellwerte (`>0.2/0.6/0.8`, `>0.3`, `>0.4`,
`>0.05`, `>0.5`), die Height-Delta-Kappungen (**15%** Ridge, **3%**
Metamorph, **3%** Igneous Flow — das sind die "15%/3%/3%"-Werte aus der
Terrain-Übergabe), `HARDNESS_TIER_FACTORS = (0.7, 1.0, 1.3)`, der
Elevation-Härte-Faktor (`0.85 + 0.15*norm_height`), alle Rock-Shift-
Faktoren (0.4/0.5/0.6/0.8 je Effekt) und die Ziel-Verteilungen
(85/7.5/7.5% Facetten, 80/10/10% Zonen).

**Konkreter Bug gefunden — Parameter-Key-Mismatch:** Die live gelesenen
Keys sind `metamorph_foliation`/`metamorph_folding`
(`geology_generator.py:1097-1098`, passt zur GUI). Aber
`_load_default_parameters()` (759-791) und der Parameter-Hash/Validator
(`1257-1279`, `1456-1495`) verwenden `metamorphic_foliation`/
`metamorphic_folding` (mit "-ic") — eine andere Schreibweise, die niemals
gelesen wird. Folge: Jeder Aufrufer, der die Parameter nicht exakt über
die GUI liefert (Tests, Standalone-Nutzung, die Legacy-Wrapper
`generate_rock_distribution()`/`generate_complete_geology()`), bekommt
`metamorph_foliation`/`_folding` = 0.0 (stiller Default-Fallback im
`.get(..., 0.0)`), **nicht** die konfigurierten Defaults 0.4/0.6.
Zusätzlich verfolgt der Cache-Invalidierungs-Hash die falsche
Schreibweise, d.h. Foliation/Folding-Änderungen lösen ggf. keine
Neuberechnung aus.

Zusätzlich: `validate_geology_parameters()`/`get_geology_parameter_info()`
(1456-1517) haben stale Ranges `[0.0, 1.0]`/Default `0.3`/`0.2` für
Ridge/Bevel-Warping, obwohl die echten Slider `[0.0, 2.0]` sind — wird
aktuell nirgends im Live-Pfad aufgerufen, ist aber eine Falle für künftige
Nutzung.

---

## Frage 3 — Skalierungsverhalten bei Map-Size-Änderung (z.B. 128→512)

**Keine km-Kopplung:** `geology_generator.py` liest `WORLD_SIZE_KM`/
`map_distance_km` **nirgends** (0 Treffer). Alle Rausch-Koordinaten sind
auf `[0,1]` normiert (`norm_x = arange(width)/width`), die
Frequenz-Multiplikatoren wirken relativ zur Kartenbreite in Pixeln, nicht
zur realen km-Ausdehnung. Das unterscheidet sich vom Terrain-Fix aus der
letzten Runde (4f): dort skaliert `spacing = world_size_m /
heightmap.shape[0]` echte Meter pro Pixel, unabhängig von Auflösung UND
Weltgröße. Geology hat dieses Pattern nicht übernommen — wurde in 4f auch
bewusst nicht mit-propagiert.

Konkrete Folge: Verdoppelt man nur die Auflösung (128→512) bei gleicher
`map_distance_km`, bleibt die Anzahl der Gebirgsketten/Gesteinszonen
optisch gleich (gut: kein "mehr Rauschen" bei mehr Pixeln). Ändert man
aber `map_distance_km` (z.B. 10km→100km) bei gleicher Auflösung, bleibt
die Anzahl der Ridge/Zonen-Features trotzdem gleich — physikalisch
sollten bei 10x größerer Kartenausdehnung z.B. auch mehr/größere
Gebirgszüge sichtbar sein oder dieselben Züge 10x so breit in km. Das
passiert aktuell nicht.

**Nyquist-Randfall bei niedrigem LOD:** Die feine Domain-Warp-Frequenz in
`apply_faceted_boundaries()` (`*30`, Zeile 554/556) hat bei LOD1 (32x32)
nur ~1.07 Pixel/Zyklus (deutlich unter der 2px/Zyklus-Grenze → Aliasing),
bei LOD2 (64x64) ~2.1px/Zyklus (Grenzfall), erst ab LOD4 (256x256,
~8.5px/Zyklus) sauber abgetastet. Kein katastrophaler Bug (löst sich mit
steigender Auflösung von selbst), aber inkonsistent — anders als Terrains
`_max_safe_octaves()`, das für alle LOD-Stufen eine garantiert
aliasing-freie Oktaven-Zahl erzwingt, hat Geology kein Äquivalent.

**Fixer Pixel-Radius bei Bevel-Warping:** `gaussian_filter(...,
sigma=2.0)` (Zeile 252) ist ein **Pixel**-Radius. Bei doppelter Auflösung
(gleiche `map_distance_km`) deckt derselbe `sigma=2.0` nur noch die halbe
reale (km-)Fläche ab — die "abgeschliffene Kante" wird bei höherer
Auflösung physisch schmaler statt gleich zu bleiben.

**Performance skaliert schlecht:** Die 6 Python-Loop-Stufen (s. Frage 1)
sind O(H×W) in interpretiertem Python. 128→512 Pixel Kantenlänge = 16x
mehr Iterationen, linear mehr `OpenSimplex`-Aufrufe. Das ist der praktisch
größte Skalierungs-Effekt (Rechenzeit), größer als jedes
Aliasing-Problem.

**Was bereits richtig skaliert:** Die Height-Delta-Kappungen (15%/3%/3%)
werden relativ zur tatsächlichen `height_range` des aktuellen Heightmaps
berechnet (`geology_generator.py:1077`), nicht zu einem festen Meter-Wert
— die Effekt-Stärke bleibt damit unabhängig von Auflösung und Weltgröße
konsistent. Auch das LOD-zu-LOD-Resampling von `previous_height_delta` via
`scipy.ndimage.zoom` ist reiner Pixel-Ratio-Resize und dafür
unproblematisch.

**Weg zur Auflösungs-/Weltgrößen-Unabhängigkeit (Empfehlung für später):**
Analog zum Terrain-4f-Pattern `map_distance_km` aus
`DataLODManager.get_map_distance_km()` lesen und
(a) alle Rausch-Frequenzen von reiner UV-Normierung auf eine reale
km-Skala umstellen (`frequenz_zyklen_pro_karte = ziel_wellenlaenge_km >
map_distance_km` statt fixer Multiplikator),
(b) `gaussian_filter`-`sigma` von Pixel-Einheiten auf
`sigma_px = sigma_km * (width / map_distance_km)` umrechnen,
(c) den `*30`-Domain-Warp mit einem `_max_safe_facet_frequency(width)`-
Clamp analog zu `_max_safe_octaves()` gegen Aliasing absichern.

---

## Frage 4 — Bewertung

**Stärken:**
- Rock-Hardness ist kein Blindgänger, sondern real downstream verdrahtet:
  `water.erosion_sedimentation` hängt explizit von `geology.hardness` ab
  (`calculator_graph.py:85-87`) — härteres Gestein bremst Erosion.
- Mass-Conservation (R+G+B=255 je Pixel) ist ein durchdachtes Constraint,
  das die Gesteins-Mischung physikalisch plausibel hält.
- Die Height-Delta-Kappen relativ zur Height-Range sind resolution-/
  weltgrößen-robust (s.o.).
- Die Delta-only-Anzeige-Infrastruktur (`get_geology_height_delta()`) aus
  der letzten Session steht bereit, auch wenn noch nicht in der UI
  verdrahtet.

**Schwächen:**
- Irreführende GPU/Fallback-Doku (Frage 1).
- Der Key-Mismatch-Bug (Frage 2) — funktioniert nur "zufällig" über den
  GUI-Pfad korrekt.
- Keine km-Kopplung (Frage 3) — größtes strukturelles Defizit, weil es
  Geology aus dem in der Terrain-Runde etablierten Skalierungs-Konsens
  herausfallen lässt.
- Performance (6 Python-Loops) wird bei höheren LODs/Auflösungen zum
  Flaschenhals.
- **UX-Kernproblem, das der Nutzer selbst benennt:** Es gibt aktuell keine
  Möglichkeit, einzelne Effekte (Ridge- vs. Bevel-Warping vs. Foliation
  vs. Folding vs. Flow) visuell voneinander zu unterscheiden.
  `geology_tab.py` hat nur 3 Anzeige-Modi (Height=kombiniert, Rock Types,
  Hardness) — kein Delta-Only-Modus, keine Isolierung einzelner
  Tektonik-Effekte. Ein Regler-Wert ändern und den Effekt NICHT sehen zu
  können, macht die Slider praktisch unbedienbar für informierte
  Entscheidungen.
- Konzeptionell ist das Modell ein reines 2D-"Mischverhältnis pro Pixel" +
  additives Delta — es gibt keine Tiefen-/Schicht-Dimension. Das erklärt,
  warum es (im Gegensatz zur echten geologischen Karte, die der Nutzer als
  Referenzbild gepostet hat) keine scharfen, durch Störungen versetzten
  Ausbiss-Bänder erzeugen kann, sondern nur weich verrauschte Zonen.

---

## Frage 5 — Slider-Verbesserungen

- **Wichtigster Hebel laut Nutzer-Feedback:** Nicht unbedingt neue Slider,
  sondern **Sichtbarkeit der bestehenden Effekte** — ein Diagnose-/Debug-
  Anzeigemodus pro Effekt (Ridge-only, Bevel-only, Foliation/Folding-only,
  Flow-only, sowie das aktuell schon vorbereitete reine `height_delta`)
  würde es erlauben, jeden Slider isoliert zu verstehen, bevor man ihn
  kombiniert einsetzt.
- Key-Mismatch (`metamorph_*` vs. `metamorphic_*`) sollte bereinigt
  werden, bevor an den Reglern selbst etwas geändert wird, sonst bleibt
  die Default-Wirkung für Nicht-GUI-Pfade weiter verdeckt kaputt.
- `ridge_warping`/`bevel_warping` (Range 0-2) vs. die restlichen 3
  Tektonik-Slider (Range 0-1) sind inkonsistent skaliert, ohne dass ein
  Nutzer aus der UI ersehen kann, warum — entweder vereinheitlichen oder
  die Tooltips um eine kurze Begründung ergänzen.
- Falls künftig km-Kopplung kommt (Frage 3): Tooltips sollten dann die
  reale Wellenlänge/Ausdehnung des Effekts in km nennen, nicht nur
  "mehr/weniger" — das macht Slider-Werte für den Nutzer konkret
  interpretierbar statt abstrakt.
- Rein spekulativ, abhängig von einer möglichen 3D-Stack-Umsetzung: dann
  würden sich "Ridge/Bevel Warping" u.U. in Fault-bezogene Slider
  (Anzahl/Dichte/Rauheit der Störungslinien) und eine Layer-Definition
  (Anzahl Schichten, Dicke) verwandeln — aber das ist eine Konsequenz
  einer Architektur-Entscheidung, kein eigenständiger Slider-Fix.

---

## Bewertung der 3D-Gesteinsstapel-Idee

**Kernidee des Nutzers:** Schichtstapel entlang Höhenachse definieren →
Tektonik (Verkippung/Faltung/Störungsversatz) auf den Stapel anwenden →
für jedes (x,y) die reale Geländehöhe z mit dem verformten Stapel
schneiden (= welches Gestein liegt an der Oberfläche) → Sedimente in
Senken/Flusstälern ergänzen → Ergebnis mit Hillshade multiplizieren.

**Fachliche Einordnung:** Das ist im Kern das Standardverfahren, mit dem
echte geologische Karten entstehen (Schichtenfolge + Störungsversatz +
Erosionsschnitt = Ausbisslinie) — genau das Muster, das auf dem vom
Nutzer gezeigten Referenzbild (Oberrheingraben-Karte) sichtbar ist:
parallele, durch Störungslinien versetzte Gesteinsbänder, die von der
Topographie "angeschnitten" werden. Das aktuelle 2D-
Mischverhältnis-Modell kann dieses Muster strukturell nicht erzeugen,
weil es keine Tiefenachse kennt.

**Machbarkeit ohne echtes 3D-Voxel-Raster:** Der Ansatz muss nicht als
dichtes 3D-Array (Layer × H × W) implementiert werden — das wäre
speicher-/rechenintensiv, aber unnötig. Da Schichten in einem Stapel
monoton in der Tiefe angeordnet sind, genügt es, pro Schicht eine 2D-
"Ober-/Untergrenzen-Höhe" als Funktion von (x,y) zu führen (z.B.
`layer_top[i](x,y) = base_elevation[i] + tilt(x,y) + fold(x,y) +
fault_offset(x,y)`), und den Ausbiss-Schritt als reinen Array-Vergleich
`z(x,y)` gegen diese Grenzen zu berechnen (`argmax`/`searchsorted`-artig)
— vollständig vektorisierbar in NumPy, kein Pixel-Loop nötig. Das macht
den Ansatz sowohl performant als auch mit dem Skalierungs-Fix aus Frage 3
kompatibel, wenn `tilt`/`fold`/`fault_offset` von Anfang an in
km-Einheiten statt UV-Raum definiert werden.

**Wo bestehende Bausteine weiterverwendbar sind:** `ridge_warping`/
`bevel_warping` könnten als Rauheit/Unregelmäßigkeit der Störungslinien-
Geometrie weiterleben statt als eigenständige Höhen-Deltas; Foliation/
Folding passen konzeptionell gut als Verformung der Schichtgrenzen selbst
(Faltung im eigentlichen Sinne); die Hardness-Werte pro Gesteinstyp
würden zu Hardness pro Schicht (mit Alter/Ära gekoppelt, wie vom Nutzer
vorgeschlagen) statt zu 3 gemischten RGB-Kanälen.

**Offene Punkte, die vor einer Umsetzung geklärt werden müssten:**
Anzahl/Dicke-Modell der Schichten (fix vs. variabel über die Karte), wie
Störungslinien geometrisch erzeugt werden (Voronoi-Kanten? Polylinien mit
Rauschen?), wie die Sediment-Ergänzung in Senken/Tälern konkret erkannt
wird (Wasser-Flow-Accumulation, falls vorhanden, oder lokale
Relief-Flachheit als Näherung), und dass die Hillshade-Multiplikation für
die 2D-Kartenansicht aktuell **gar nicht existiert**
(`map_display_2d.py:set_shadow_overlay()` ist nur ein Platzhalter-Hook
ohne Implementierung, Zeile 1040-1047) — der letzte Schritt des
Nutzer-Vorschlags bräuchte also zusätzlich eine neue
2D-Hillshade-Implementierung, die aktuell nirgends existiert (nur
3D-Viewport hat Shading).

**Empfehlung:** Als eigene, spätere Umsetzungsrunde angehen, idealerweise
NACHDEM die in Frage 4/5 empfohlenen Diagnose-Anzeigen für die aktuellen
Einzeleffekte existieren — das liefert die visuelle Grundlage, um zu
entscheiden, welche der jetzigen 5 Tektonik-Effekte 1:1 in das
Schicht-Modell übernommen werden sollten und welche im neuen Modell
überflüssig werden.

---

## Was in dieser Runde NICHT gemacht wurde (bewusst)
- Keine Code-Änderungen (reine Analyse-Runde, per Nutzer-Entscheidung).
- Kein Fix des Key-Mismatch-Bugs, keine km-Kopplung, keine Vektorisierung,
  kein GPU-Pfad, keine Diagnose-Anzeigemodi, kein 3D-Stack-Prototyp — alle
  oben als "Empfehlung für später" markiert, aber unimplementiert.

## Empfehlung für den Einstieg in die nächste Runde
Der Nutzer geht als Nächstes vermutlich in eine Umsetzungs-Runde für
Geology (Auswahl aus den obigen Empfehlungen) oder direkt weiter zu
Weather. Falls Umsetzung: sinnvolle Reihenfolge wäre (1) Key-Mismatch-Bug
fixen, (2) 6 Python-Loop-Stufen vektorisieren, (3) km-Kopplung nach
Terrain-4f-Pattern, (4) Diagnose-/Debug-Anzeigemodi in `geology_tab.py`
(pro-Effekt-Isolierung), (5) GPU-Pfad, (6) als separate, größere Runde:
3D-Gesteinsstapel-Prototyp.
