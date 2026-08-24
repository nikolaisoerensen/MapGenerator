# Anzeige im 3D und Binnenseen — Plan

Stand 2026-08-24. Drei Nutzerwuensche aus einer Nachricht, hier getrennt
und geordnet. **Diese Datei ist die Ordnung** — was hier nicht steht, ist
nicht beschlossen.

---

## Was schon da ist (geprueft 2026-08-24)

Damit niemand baut, was es gibt:

| | Stand |
|---|---|
| Strand-Biom | **Existiert.** `beach` in `core/biome_generator.py:1547`, Bedingung `h <= sea_level+5` UND Abstand zum Ozean `<= bank_width`. Das ist fast genau der Fallback, den der Nutzer beschrieben hat ("unter 10 m Hoehe, max 50 m zum Meer"). |
| Fluss-Biome | **Existieren.** `lake` (1), `grand_river` (2), `river` (3), `creek` (4) aus `water_biomes_map`. |
| Overlay-System im 3D | **Existiert.** `self.overlay_data[tab_type]`, Layer werden als Textur aufs Mesh gelegt (`_colorize_layer`, `_render_overlay`). |
| Koordinatenanzeige | **Nur in 2D**, und nur die Koordinaten: `coord_label` in `map_display_2d.py:535`, gefuellt von `_on_mouse_move`. Kein Layerwert. |
| Maus -> Weltposition im 3D | **Fehlt.** `gui/widgets/karten_auswahl.py` kann nur VORWAERTS projizieren (Weltposition -> Bildschirm), um Orte anzuklicken. Der umgekehrte Weg — Mausposition -> Gelaendepunkt — ist nicht gebaut. |

**NACHGEMESSEN 2026-08-24 - die Vermutung stimmte nur zur Haelfte:**

  * **Straende: ja, reines Anzeigeproblem.** 311 Strandpixel liegen in
    `biome_map_super`. In `biome_map` sind sie 0 - dort fehlen ALLE sechs
    Wahrscheinlichkeits-Biome (beach, cliff, lake_edge, river_bank,
    snow_level, alpine_level). Das ist Bauart: erst das Supersampling
    setzt Wahrscheinlichkeiten in Pixel um. Wer Straende zeigen will,
    muss `biome_map_super` nehmen.
  * **Grosse Fluesse: NEIN, das war ein Erzeugungsproblem.** `river` und
    `grand_river` kamen ueberhaupt nicht vor - eine Perzentilschwelle mit
    absoluten Faktoren multipliziert, siehe SITZUNGSLOG. Behoben; es gibt
    jetzt 299 River- und 77 Grand-River-Laeufe.

---

## Block A — Was ist unter dem Cursor

Nutzerwunsch: *"entweder das man die kuestentypen markieren kann und es an
der seite steht oder wenn das zu komplex ist, dann einfach nur bei den
koordinaten die angezeigt werden steht ueber was man gerade drueber haelt.
also kuestentyp im reiter fuer kuestentyp, biom fuer biome etc."*

Der Nutzer hat die einfachere Variante selbst benannt — die
Koordinatenzeile erweitern. Genau die wird gebaut.

### [ ] A.1 Layerwert in der 2D-Koordinatenzeile
Der schnelle Teil: `_on_mouse_move` kennt die Pixelposition, die
Layerdaten liegen vor. Statt nur `(x, y)` auch den Wert des gerade
angezeigten Layers — als Zahl bei skalaren Layern, als NAME bei
kategorischen (Biom, Kuestentyp, Gestein).

### [ ] A.2 Maus -> Gelaendepunkt im 3D
Der eigentliche Aufwand. Braucht einen Strahl vom Auge durch das
Mauspixel und einen Schnitt mit dem Hoehenfeld (schrittweises Marschieren,
dann Verfeinern). Rein rechnerisch, also headless pruefbar — anders als
Color-Picking.

### [ ] A.3 Dieselbe Zeile im 3D
Sobald A.1 und A.2 stehen, ist das nur noch Verdrahtung.

---

## Block B — Fluesse und Straende sichtbar machen

Nutzerwunsch: *"dass man im 3D modus bei dem Flussnetzwerk keine fluesse
sehn kann. ich will in den jeweiligen reitern die fluesse auf dem boden
sehen. die grossen fluesse koennen als overlay bei biome drin sein."*

### [ ] B.1 Flussnetz als Overlay im Fluss-Reiter
`river_mask` und `river_order` liegen als Terrain-Ausgaben vor
(`terrain.redistribution`). Sie muessen nur als Overlay-Layer registriert
und eingefaerbt werden — das System dafuer steht.

### [x] B.2 Grosse Fluesse im Biome-Reiter - ERZEUGUNG BEHOBEN 2026-08-24
Die Biome `grand_river` und `river` gibt es bereits. **Zuerst pruefen, ob
sie im Bild ankommen** — wenn `water_biomes_map` sie liefert und die
Biomkarte sie faerbt, ist nichts zu tun ausser der Sichtpruefung.

### [x] B.3 Straende - GEKLAERT 2026-08-24 (sie existieren, in `biome_map_super`)
Ebenfalls **zuerst pruefen**: `beach` existiert mit fast genau der vom
Nutzer beschriebenen Bedingung. Falls es im Bild fehlt, liegt es an
`bank_width`/`sea_level` oder daran, dass ein anderes Biom es
ueberschreibt (Prioritaet 6).

Der Nutzer schlug vor, die Information aus dem KUESTENTYP zu ziehen. Das
waere genauer (ein Archetyp mit `strand_anteil` 0.65 hat mehr Strand als
einer mit 0.05), ist aber nur noetig, wenn die vorhandene Loesung nicht
reicht.

---

## Block C — Binnenseen

Nutzerwunsch: *"die binnenseen unter 0 m sind ja nicht geglaettet vom
untergrund. diese koennten theoretisch auch mit dem
meeresvertiefungs-code gesenkt werden wenn sie dadurch die form nicht
verlieren, oder aber eine glatte flaeche bekommen damit sie homogen
aussehn. dann haette ich gerne, dass die biome in solchen seen dann als
Lake oder Sea erkannt werden (je nachdem ob es kontakt zum meer gibt)."*

### [ ] C.1 Seeboden glaetten
Gemessen (512 px, Seed 20260804): 11 Binnenseen ueber 4 Pixel, groesster
252 Pixel, verteilt auf Mittelmeer, Atlantikkueste, Griechische Inseln,
Huegelland. **Keiner im Fjordland oder in der Taiga.**

Zwei Wege, wie der Nutzer sagt:
  * Mit `_seetiefe_aus_archetyp()` senken — der Code existiert, arbeitet
    aber auf dem HAUPTMEER (`_hauptmeer_maske`). Binnenseen sind bewusst
    ausgenommen.
  * Eine glatte Flaeche auf Ueberlaufhoehe. Einfacher und homogener.

### [ ] C.2 Lake gegen Sea unterscheiden
Kontakt zum Meer = `Sea`, kein Kontakt = `Lake`. Die
Zusammenhangsanalyse dafuer gibt es schon (`ndimage.label`, in
`vektor_kueste._inseln_bauen()` fuer Landstuecke).

**Beachten:** das Biom `lake` haengt heute an `water_biomes_map == 4`,
also am WASSERsystem, nicht am Terrain. Wer C.2 baut, muss entscheiden,
ob die Terrain-Binnenseen dort eingespeist werden oder ob die Biomkarte
eine zweite Quelle bekommt. **Zwei Quellen fuer dieselbe Aussage sind in
diesem Projekt schon mehrfach schiefgegangen** (SPEZIFIKATION §4.5).

---

## Reihenfolge

1. **B.2 und B.3 PRUEFEN** — moeglicherweise ist nichts zu bauen. Billig,
   und es verhindert, dass Vorhandenes ein zweites Mal entsteht.
2. **B.1** — Flussnetz als Overlay, das System steht.
3. **A.1** — Layerwert in 2D, schnell und sofort nuetzlich.
4. **A.2/A.3** — Maus-Raycast im 3D, der eigentliche Aufwand.
5. **C.1/C.2** — Binnenseen, mit der Quellenfrage aus C.2 vorab geklaert.
