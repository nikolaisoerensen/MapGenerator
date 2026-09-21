# Übergabe — Sitzung 2026-08-24

Geschrieben beim Accountwechsel. **Diese Datei ist der Einstieg für die
nächste Sitzung.** Danach `docs/SITZUNGSLOG.md` (neueste Einträge oben).

---

## ZUERST: nichts ist committet

```
git status --short | wc -l     →  88
```

**88 Dateien geändert oder neu, alles uncommitted, auf `main`.** Darunter
die gesamte Arbeit dieser Sitzung *und* die der vorherigen (Vektorküste,
Remesh, Wegsuche, Spielkarten). Der letzte Commit ist vom 12.08.

Vor allem anderen: **das sichern.** Ein verlorener Arbeitsbaum kostet hier
Wochen. Der Nutzer arbeitet bewusst direkt auf `main` ohne Zwischencommits
(siehe Memory `feedback_edit_main_checkout_not_worktree`) — das ist seine
Entscheidung, aber 88 Dateien sind viel Risiko auf einmal.

---

## Was in dieser Sitzung passiert ist

Fünf Stränge, in dieser Reihenfolge. Die Details stehen alle im
SITZUNGSLOG; hier nur, was man wissen muss, um weiterzuarbeiten.

### 1. Das 3D-Mesh war zerschnitten

Ursache: falsche Dreiecks-Wicklung in `gui/widgets/kuesten_schnitt.py`
(`_faecher()`), gegen `glFrontFace(GL_CW)`. Behoben. Der Test prüfte nur,
dass die Wicklung *einheitlich* ist — nicht, dass sie zum Gitter passt;
`wicklung_wie_gitter` prüft das jetzt.

### 2. Leistung

Fünf Maßnahmen umgesetzt und bit-identisch verifiziert (numba-A*,
bandweiser Erosionsfilter, Poisson-Punkte, Teilschritte-Protokoll,
Fortschrittsanzeige im Terrain-Generator). Fünf weitere Vorschläge
**widerlegt und im Code dokumentiert**, damit sie niemand erneut
versucht. Details: `docs/PERFORMANCE_2026-08-23.md`.

**Eine Falle, die Zeit gekostet hat:** der parallele Erosionsfilter war
5x schneller und machte danach *alles übrige numpy dauerhaft 2.4x
langsamer* (Allokator-Fragmentierung). Die sequentielle Fassung ist
3x schneller ohne Nachwirkung. `smoke_test_erosionsfilter_baender.py`
hat dafür eine eigene Gruppe `keine_nachwirkung`.

### 3. Küstenmodell auf Meter umgestellt

Die 27 Archetyp-Profile sind jetzt **in Metern** aus echten Vorbildern
vermessen (`tools/archetyp_vorbilder.py` nennt zu jedem den Ort).
`MESS_PROFIL_M_JE_ARCHETYP` in `core/vektor_kueste.py`.

Der alte Test maß *normierte Formen* — eine Küste mit richtiger Form und
völlig falscher Höhe war grün. Genau so blieb das Weissmeer bei 187 m
unbemerkt (soll 5). `smoke_test_kuestenprofiltreue.py` misst jetzt Meter.

**Achtung bei DEM-Messungen:** die Pixel sind bei hoher Breite
rechteckig (31.1 m N-S gegen 14.5 m O-W bei 62°N). `zellgroesse_m()` in
`tools/kuestenlaengsschnitt.py` gibt beide zurück. Alle 27 Profile
mussten deswegen neu gemessen werden.

### 4. Flüsse folgen jetzt dem Wasser

Flussgröße hängt am Niederschlag statt an der Fläche; Hauptstrom-Quoten
je Region (Skerrheim 100 %, Morobora und Atlantik je 66 %); Talformen V/U
je Region über `talform`. Die Ordnung dazu steht in
`docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md` — **was dort nicht steht, ist nicht
beschlossen.**

### 5. Drei größere Fehler gefunden — alle nach demselben Muster

Das ist der wichtigste Teil dieser Übergabe.

| Fehler | Was falsch war | Ausmaß |
|---|---|---|
| **Archetyp-Verteilung** | weißes Rauschen in der Saat-Zuweisung | **8 von 27 Küstentypen kamen gar nicht vor** |
| **Flächeneichung** | `flaeche_soll` bei 6 von 9 Regionen ungesetzt | Spanne beste/schlechteste Region 2.10x |
| **Flussstufen** | Perzentilschwelle mit absoluten Faktoren multipliziert | **`river` und `grand_river` kamen gar nicht vor** |

Alle drei sind behoben und durch Tests abgesichert. **Alle drei liefen
fehlerfrei durch und lieferten plausible Ergebnisse** — kein Absturz,
keine Warnung, alle bestehenden Tests grün.

---

## DIE LEHRE, die in die nächste Sitzung gehört

Alle drei Fehler waren **Ketten aus lauter einzeln richtigen Teilen**.

Beim Küstenbug wurden nacheinander geprüft und für korrekt befunden:
Profil, Mischung, Reichweite, Höhendeckel, Inselschluss, Stärke. Jedes
Glied war grün. Der Fehler saß in der **Zuordnung dazwischen**, und keine
Einzelgliedprüfung konnte ihn sehen.

Gefunden wurde er erst, als eine Messung **zwei Enden der Kette
gegeneinander hielt**: Anteil an Saatstationen gegen Anteil an fertiger
Küstenlänge. Das fand ihn in einem Schritt.

**Wenn also mehrere Prüfungen grün sind und das Ergebnis trotzdem falsch
ist: aufhören, Einzelglieder zu prüfen. Zwei Enden gegeneinander
messen.**

Dieselbe Lehre steht in CLAUDE.md schon zweimal (GPU-Fallback nach
Dateiumzug, adaptives Mesh). Sie hat sich hier ein drittes Mal
wiederholt.

### Ein Messfehler, der mehrere Runden gekostet hat

Mehrere Messungen bauten sich ihre `VektorKueste` selbst:

```python
vd.VEKTOR_KUESTE_AKTIV = False
H0, f = weltfeld(512, SEED)      # ← läuft den ALTEN Rasterpfad!
vk = VektorKueste(H0, ...)
```

Mit dem Schalter auf `False` läuft `weltfeld()` durch
`_kuesten_umformen()` — ein anderes Gelände. Die Messungen sagten 144 m,
die Karte zeigte 14 m.

**Immer `felder["vektor_kueste"]` aus dem echten Lauf nehmen.**

---

## Stand der Tests

Grün:

```
smoke_test_kuestenprofiltreue.py       2/2   Median 3.1 m, flache Küsten 14/14
smoke_test_archetyp_verteilung.py      3/3   alle 27 Archetypen kommen vor
smoke_test_flussstufen.py              3/3   alle vier Wasserstufen
smoke_test_regionen_fairness.py        2/2   Spanne 1.25x, alle Ziele getroffen
smoke_test_fluss_overlay.py            5/5
smoke_test_erosionsfilter_baender.py   inkl. keine_nachwirkung
smoke_test_wegsuche_schnell.py
```

Rot, **und war es vor dieser Sitzung schon**:

```
smoke_test_regionen_welt.py            Naht 1.540 (Grenze 1.25)
                                       Regionshänge daneben, Ordnung verrutscht
```

Zur Naht: sie lag vor dieser Sitzung bei exakt 1.25, also **ohne jede
Reserve**. Die Metrik mischt Küstenformen mit Regionsnähten und kann
deshalb nicht sagen, welches von beidem zu steil ist. **Bevor jemand
deswegen am Gelände dreht, sollte er klären, ob die Naht überhaupt misst,
was sie messen soll.** Der Test bleibt trotzdem der empfindlichste
Wächter für Geländeänderungen (CLAUDE.md) — er läuft rund zwei Minuten.

---

## Was als Nächstes zu tun ist

In dieser Reihenfolge. Punkt 2 ist eine Entscheidung des Nutzers, kein
Bauauftrag.

### 1. B.1 — Flussnetz als Overlay im Fluss-Reiter
Das Overlay-System steht (`smoke_test_fluss_overlay.py` sichert die
Verdrahtung), und seit dem Flussstufen-Fix gibt es endlich große Flüsse,
die sich zu zeigen lohnen. Plan: `docs/archiv/2026-08-24_ANZEIGE_UND_SEEN.md`.

### 2. ENTSCHEIDUNG: Anzeige auf `biome_map_super` umstellen?
Es gibt zwei Biomkarten:

* `biome_map` — die grobe. Enthält die Wasserstufen, aber **keines** der
  sechs Wahrscheinlichkeits-Biome (beach, cliff, lake_edge, river_bank,
  snow_level, alpine_level). Alle sechs sind dort 0.
* `biome_map_super` — die feine. Erst das Supersampling setzt
  Wahrscheinlichkeiten in Pixel um; dort liegen 311 Strandpixel.

Die Umstellung brächte Strände, Klippen, Ufersäume und die Alpinzone
sichtbar ins Bild. **Nicht ungefragt umstellen** — es ändert, was der
Nutzer überall angezeigt bekommt.

### 3. A.1 / A.2 / A.3 — was ist unter dem Cursor
A.1 (Layerwert in der 2D-Koordinatenzeile) ist schnell. A.2 (Maus →
Geländepunkt im 3D per Raycast) ist der eigentliche Aufwand, aber rein
rechnerisch und damit headless prüfbar.

### 4. C.1 / C.2 — Binnenseen
Glätten und Lake/Sea unterscheiden. **Vorher die Quellenfrage klären:**
das Biom `lake` hängt heute an `water_biomes_map`, nicht am Terrain. Zwei
Quellen für dieselbe Aussage sind in diesem Projekt schon mehrfach
schiefgegangen (SPEZIFIKATION §4.5).

### 5. Zurückgestellt, auf Wunsch des Nutzers
* Morobora-Seenlandschaft („so wie in Lappland") — erster Schritt wäre
  `water.lake_detection`.
* Block 1.3 (Verdunstung), 5.1/5.2 (Fjordarme, Fjord → Geiranger-Küstentyp).
* Saisonale Flüsse.

---

## Der stehende Blocker: nichts ist visuell bestätigt

**Seit dem 12.08. hat niemand die Küstenformen im laufenden Programm
gesehen.** Diese Sitzung hat danach noch erheblich am Gelände geändert:
alle 27 Archetypen kommen jetzt vor (vorher 19), Skerrheim ist größer und
steiler, die Wasserklassifikation hat drei statt einer Flussstufe.

Headless ist alles grün. Gesehen hat es niemand.

`docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md` listet, was nur am laufenden Programm zu prüfen
ist. Das sollte früh in der nächsten Sitzung passieren — je mehr sich
darüber stapelt, desto schwerer wird zuzuordnen, was einen Fehler
verursacht hat.

---

## Arbeitsweise, die der Nutzer sich gewünscht hat

Aus dieser und früheren Sitzungen, weil sie sonst verlorengeht:

* **„sag mir immer was als nächstes zu tun ist, so dass ich weiß was
  danach gemacht wird."** Jede Antwort endet mit den nächsten Schritten.
* **„versuche es für mich etwas zu ordnen und die ordnung im programm für
  mich zu halten."** Beschlüsse gehören in die Doku, nicht nur in den
  Chat. Der Nutzer hat zu Recht gerügt, dass das eine Zeit lang nicht
  passiert ist.
* **Jede Antwort endet mit „was du jetzt anders siehst"** — und sagt
  ehrlich, wenn nichts sichtbar ist.
* **Vor großen, riskanten Umbauten nachfragen.** Zweimal bestätigt; die
  kleinere sichere Option war beide Male richtig.
* **Der OpenTopography-Schlüssel wird NIE in den Chat gepastet.** Er wird
  ausschließlich aus `OPENTOPOGRAPHY_API_KEY` gelesen und nie ausgegeben.
  `tools/_dem_cache/` steht in `.gitignore`.

---

## Werkzeuge, die diese Sitzung hinterlässt

| Datei | Wofür |
|---|---|
| `tools/flaeche_eichen.py` | `flaeche_soll` je Region auf die Zielwerte einregeln. Logarithmischer Regelkreis, drei Runden. |
| `tools/archetyp_vorbilder.py` | Welcher echte Ort welchem Archetyp zugrunde liegt, mit Begründung je Korrektur. |
| `tools/region_gegen_vorbild.py` | Erzeugte Region gegen echtes DEM halten (maßstabsfreie Formkennzahlen). |
| `tools/kuestenlaengsschnitt.py` | Profile aus echten Küsten messen. **`zellgroesse_m()` beachten.** |
| `managers/teilschritte.py` | Teilschritte in Log und Fortschrittsbalken. `Teilschritte(None)` ist ein No-op. |
