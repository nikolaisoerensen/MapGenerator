# Übergabe — Stand 2026-08-12

Dieses Dokument bringt einen neuen Rechner (oder eine neue Sitzung) auf den
Stand. Es ist bewusst kurz; die Einzelheiten stehen in den verlinkten Dateien.

---

## 0. ZUERST: der Stand ist NICHT gesichert

**Der letzte Commit ist vom 2026-08-04. Alles, was seither entstanden ist,
liegt ausschließlich als Arbeitskopie auf diesem Rechner.** Das sind rund
**14 000 Zeilen** — 29 geänderte Dateien und etwa 30 Dateien, die git noch nie
gesehen hat.

Darunter ist **die wichtigste Datei des Projekts**:

    core/terrain_weltkarte.py      ← Weltkarte, 9 Regionen, Seegliederung,
                                     Küsten-Archetypen — NIE COMMITTED

Ebenso `core/terrain_weltfluesse.py`, `gui/widgets/adaptive_terrain_mesh.py`,
zwei komplette Reiter (`river_tab.py`, `settlement_regional_tab.py`), 17
Testdateien und **sämtliche Dokumentation in `docs/`**.

> **Ein `git clone` auf dem anderen Rechner holt von all dem nichts.**
> Vor dem Umzug muss committed und gepusht werden — das ist Schritt 1, alles
> Weitere hängt daran. Das Remote steht bereits:
> `github.com/nikolaisoerensen/MapGenerator.git`

Empfohlenes Vorgehen (nicht ausgeführt — Commits macht der Nutzer):

```bash
git status                 # zuerst ansehen, was mitkommt
git add -A
git commit -m "Zwischenstand 2026-08-12: Weltkarte, Küsten, Siedlungen, adaptives 3D-Mesh, Doku"
git push
```

`.gitignore` wurde am 2026-08-12 so ergänzt, dass die 48 MB Laborbilder
(`lab_output/`) und die Laufzeit-Logs (`logs/`) **nicht** mitgehen.

---

## 1. Umgebung auf dem Zielrechner

| | |
|---|---|
| Python | 3.13.5 |
| Umgebung | `.venv` im Projektstamm, **nicht pro Worktree** |
| Start | `.venv/Scripts/python.exe main.py` |
| Tests | `.venv/Scripts/python.exe tests/smoke_test_<name>.py` |

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### Die eine Falle bei den Abhängigkeiten

`requirements.txt` pinnt **`numpy==2.4.6`, obwohl 2.5.1 existiert. Das ist
Absicht.** `numba` verlangt numpy ≤ 2.4. Mit numpy 2.5 schlägt `import numba`
fehl — und `opensimplex` fängt das ab und rechnet still ohne JIT weiter. Keine
Fehlermeldung, nur ein Programm, das **213-mal länger** rauscht (gemessen:
0.0018 s → 0.3831 s für `noise2array` 256×256).

Nach dem Aufsetzen einmal prüfen:

```bash
.venv/Scripts/python.exe -c "import numba, numpy; print(numpy.__version__)"
```

Schlägt der Import fehl, ist numpy zu neu und der schnelle Pfad ist weg.

---

## 2. Was man gelesen haben muss

In dieser Reihenfolge:

1. **`CLAUDE.md`** — Arbeitsregeln. Enthält vier teuer bezahlte Lektionen
   (Worktrees, numpy/numba, Pfade aus `__file__`, headless testbare Shader).
2. **`docs/SPEZIFIKATION.md`** §1–§6 — Ziel, Komponentenziele, **Invarianten**.
   Vor jeder Änderung lesen, nach jeder Änderung die Prüflisten abgehen.
   (§7 und alles danach sind datierte Sitzungsprotokolle, kein Pflichtstoff.)
3. **`docs/OFFENE_PUNKTE.md`** — **die einzige Aufgabenliste.** 95 Punkte,
   67 erledigt, 9 teilweise, 19 offen. Ganz oben steht „Was zuerst".
4. **`docs/TESTBERICHT.md`** — frisch gemessen am 2026-08-12, 34/40 Tests grün,
   jeder Fehlschlag einzeln erklärt.

`docs/TODO.md` gibt es nicht mehr — vollständig in `OFFENE_PUNKTE.md`
Abschnitt 12 aufgegangen.

---

## 3. Der wichtigste offene Punkt

**OFFENE_PUNKTE 6.18 — jeder Wechsel der 3D-Ansicht dauert rund 15 Sekunden.**
Das behindert die tägliche Arbeit am stärksten und ist als Nächstes dran.

Was dazu bekannt ist, und was daran ungewöhnlich ist: Terrain-**Heightmap**
ruckelt **nicht**, **Slope** ruckelt; Geologie **Rock Outcrop** kaum,
**Hardness** und **Water Depth** deutlich. Es sind also genau die Ansichten
langsam, die eine andere Karte als die Heightmap zeigen — obwohl sich am
Gelände nichts ändert, nur die Textur.

**Ein Erklärungsversuch ist bereits widerlegt:** vermutet wurde ein unnötiger
Neuaufbau des 3D-Netzes bei jedem Wechsel. Dagegen wurde eine
Änderungserkennung eingebaut — der Nutzer meldete danach **unverändert 15 s**.
Die Ursache liegt woanders.

**Messpunkte stehen bereits im Code** (`print`-Zeilen in
`map_display_3d.py::update_heightmap()` und
`base_tab.py::_push_data_to_current_display()`). Sie melden getrennt Vergleich,
Heightmap-Beschaffung und Netzaufbau.

> **Der nächste Schritt ist reines Ablesen, keine Analyse:** Programm starten,
> Pipeline durchlaufen lassen, auf Slope wechseln, Konsole lesen.
>
> **Diese `print`-Zeilen müssen nach der Diagnose wieder raus** — sie sind
> Werkzeug, kein Dauerzustand.

Noch ungeprüfte Verdächtige: `get_terrain_data_combined()` kopiert und addiert
bei **jedem** Aufruf bis zu fünf volle 1024×1024-Karten; und der 2D-Renderer
zeichnet bei jedem Wechsel neu.

---

## 4. Was diese Sitzung geändert hat

| Was | Wo | Zustand |
|---|---|---|
| Adaptives 3D-Netz (Quadtree statt gleichförmigem Gitter) | `gui/widgets/adaptive_terrain_mesh.py` (neu) | **19.7 %** der Dreiecke bei 1024 px, im Programm bestätigt |
| Vorbedingung des Netzes war falsch → lief nie | `adaptive_terrain_mesh.py` | behoben, siehe unten |
| Klippen sprangen senkrecht auf einem Pixel | `core/terrain_weltkarte.py` | Median-Sprung 63 m → 20 m |
| Küstenpass hat die Regions-Hangeichung verschoben | — | **neu gefunden, offen** (3.9) |
| GPU/CPU-Unterschied bei Landmarks | — | **neu aufgenommen, ungeprüft** (7.8) |
| Doku vollständig auf einen Stand gebracht | `docs/` | TODO gelöscht, Testbericht neu, Archiv angelegt |

### Zwei Lektionen, die über diesen Fall hinausgehen

**Grüne Tests können eine tote Funktion verdecken.** Das adaptive Netz lief
wochenlang gar nicht: die Vorbedingung verlangte Kantenlänge `2ⁿ+1` (die
übliche RTIN-Konvention), dieses Projekt benutzt aber Größen, die **selbst**
Zweierpotenzen sind (1024, nicht 1025). Damit war die Bedingung für **jede
reale Kartengröße** falsch und der Rückfall auf den alten Pfad griff still.
Die zehn Tests waren grün — **weil sie alle mit 129/257/513 gebaut waren, also
mit einer Größenklasse, die im Programm nicht vorkommt.** Gefunden wurde es
nur, weil eine Logzeile den Rückfall laut meldete.

*Daraus folgt:* Tests mit den **echten** Eingabegrößen bauen, und jeder stille
Rückfall auf einen Ersatzpfad braucht eine laute Logzeile.

**Eine Geländeänderung verstimmt zuerst die Regionseichung.** Die
Küsten-Archetypen (3.8) wurden gegen vier Terrain-Tests geprüft —
`smoke_test_regionen_welt.py` war nicht darunter, und genau der schlägt jetzt
fehl. Gemessen: Samarcia 10.6° mit Küstenpass gegen 6.6° ohne, bei Ziel 6.5°.

*Daraus folgt:* Dieser Test gehört in die Prüfliste **jeder** Änderung an
`weltfeld()`.

---

## 5. Zustand der Dokumentation

Alles in `docs/` wurde am 2026-08-12 gesichtet.

| Datei | Zustand |
|---|---|
| `OFFENE_PUNKTE.md` | **gültig, einzige Aufgabenliste** |
| `TESTBERICHT.md` | **gültig, frisch gemessen** |
| `SPEZIFIKATION.md` | gültig (§1–§6 Pflicht, §7+ Protokolle) |
| `BIOME_MATRIX.md` | gültig als **Sollvorgabe** — Kopf sagte fälschlich „nicht umgesetzt" |
| `KLIMA_UND_SEE.md` | gültig, besonders §0 — Kopf sagte fälschlich „zur Abstimmung" |
| `KULTUREN_UND_ORTE.md`, `SIEDLUNGEN_ENTWURF.md` | gültig, umgesetzt |
| `INTEGRATIONSPLAN.md` | **Weltmaße veraltet** (21.3 km statt 15 km), Warnkasten gesetzt |
| `generation_pipeline_dependencies.md` | **Knotenzahl veraltet** (38 statt 29), Datenflüsse gültig, Warnkasten gesetzt |
| `backlog.md` | Archiv; 3 offene Punkte nach 7.9–7.11 gerettet |
| `archiv/` | historisch, gilt nicht mehr — eigenes README |

Ordner aufgeräumt: 7 Logdateien, 2 Absturzabbilder und eine Profildatei
gelöscht (~350 KB), `logs/` geleert (1.8 MB), fünf historische Dokumente und
das elf Monate alte `descriptor.py` (281 KB, kannte keine der heutigen
Kerndateien) nach `docs/archiv/` verschoben.
