# Die neun Regionen und ihre Zielwerte

**Was das hier ist.** Alle Sollzahlen an einer Stelle: je Region (Teil A) und
je Komponente (Teil B). Wer wissen will, ob eine Messung gut oder schlecht
ist, findet hier den Wert, gegen den sie zu halten ist.

**Was hier NICHT steht.** Wie die Werte zustande kommen — das steht in den
Themendateien [11_GELAENDE.md](11_GELAENDE.md), [12_WASSER.md](12_WASSER.md)
und [13_KLIMA_UND_BIOME.md](13_KLIMA_UND_BIOME.md). Und: die einundzwanzig
**realen Vorbildlandschaften** mit Luftbildern, Talformen und Klimadaten
liegen unter `docs/regionen/` (eigenes `README.md`, je Ort eine `REGION.md`).
Das ist ein anderer Katalog als die neun Spielregionen hier — er dient dem
Abgleich mit der Wirklichkeit, nicht der Erzeugung.

> **Bekannte Unstimmigkeit:** `docs/regionen/README.md` spricht von „Zwanzig
> realen Landschaften", es liegen aber einundzwanzig Ordner dort. Nicht
> aufgelöst.

*Herkunft: `docs/archiv/2026-07-29_SPEZIFIKATION.md` (Stand 2026-07-29,
Teil A berichtigt 2026-09-17 unter Ticket #43), §2 „Zielkatalog: Neun
Regionen" und §3 „Komponenten-Ziele". Umschriften wie „fuer" sind beim
Übernehmen in echte Umlaute zurückgeführt; Zahlen, Tabellen und Aussagen sind
unverändert.*

---

## A. Die neun Regionen

**Berichtigt 2026-09-17 (Ticket #43).** An dieser Stelle stand bis hierhin
eine Liste von 20 frei erdachten Einzel-Landschaften (Sahara, Alpen, Amazonas,
...) mit durchweg leeren Zielwerten — als Katalog aus der Zeit angelegt, bevor
die Karte in Regionen eingeteilt wurde, und seither nie befüllt. Das
Programm rechnet seit der Weltkarte nicht mehr mit frei wählbaren
Einzel-Landschaften, sondern mit **neun festen Regionen**, die zusammen einen
Kontinent bilden (Voronoi-Aufteilung, siehe `core/terrain_weltkarte.py`).
Jede Region ist an einem realen Bezugsort geeicht — das Oberziel „reale
Landschaften nachbilden" ([01_ZIEL.md](01_ZIEL.md)) gilt unverändert, nur an neun festen statt an
20 frei wählbaren Stellen.

Maßgebliche Quelle für sämtliche Kennwerte in diesem Abschnitt ist
`core/daten/regionen_welt.toml` (Stand 2026-09-17, seit Ticket #29 in einer
eigenen Datei; zuvor standen dieselben Werte, unverändert, als
Modulkonstanten in `core/terrain_weltkarte.py`). Wer die Karte ändern will,
ändert diese Datei; wer nur diese Spezifikation ändert, ändert am
Programm nichts.

| Region | Volk | Bezugsort | Charakter |
|---|---|---|---|
| Clonagh | Kelten | Cork | sanfte Wellen, breite Sohlen, dichtes Bachnetz |
| Skerrheim | Wikinger | Bergen | ein Hauptfjord, Hochfläche, steile Wände |
| Morobora | Slawen | Wologda | flaches Hochland, weite Mulden, träge Mäander |
| Estrande | Franken | La Rochelle | Küstenebene mit Ästuar, Kliff im Norden |
| Nevadin | Alemannen | Chur | Trogtäler, scharfe Grate, große Massive |
| Nebelrode | Sachsen | Bamberg | dichte dendritische Zertalung |
| Samarcia | Andalusier | Madrid | Trockentäler, weite Flächen, wenig Netz |
| Macchia | Italiener | Rom | Küstengebirge direkt am Meer, kurze steile Läufe |
| Thalassia | Byzantiner | Iraklio | Archipel, viel Wasser, kleine steile Inseln |

### A.1 Gelände-Kennwerte

Höhen sind auf die Kontinentgröße umgerechnet, nicht von den Vorbildern
abgeschrieben — übernommen ist das VERHÄLTNIS von Relief zu Breite, nicht
der Absolutwert realer Gebirge (Begründung ausführlich im Kopfkommentar
von `core/daten/regionen_welt.toml`).

| Region | hoehe_m | relief_m | formgroesse_m | rauheit | potenz | talform |
|---|---|---|---|---|---|---|
| Clonagh | 165.3 | 79.5 | 1600 | 0.52 | 1.0 | 1.3 |
| Skerrheim | -52.0 | 484.9 | 1400 | 0.45 | 0.55 | 2.6 |
| Morobora | 293.6 | 118.1 | 3000 | 0.42 | 0.9 | 2.0 |
| Estrande | -80.9 | 147.1 | 2400 | 0.50 | 1.3 | 1.5 |
| Nevadin | 1000.0 | 1050.0 | 3800 | 0.68 | 1.5 | 0.8 |
| Nebelrode | 350.0 | 134.7 | 1400 | 0.62 | 1.0 | 1.3 |
| Samarcia | 230.0 | 109.4 | 2600 | 0.48 | 1.4 | 1.1 |
| Macchia | 0.6 | 312.9 | 1800 | 0.60 | 1.2 | 0.9 |
| Thalassia | -67.8 | 403.6 | 1100 | 0.58 | 1.1 | 0.9 |

`hoehe_m` ist die MITTLERE Hoehe der Region (negativ = überwiegend Wasser),
nicht der tiefste Punkt. `talform` ist der Exponent der Querschnittskurve
beim Täleingraben (`profil = (1 - exp(-abstand/breite)) ** talform`): klein
(0.8–1.1) V-Tal, mittel (1.3–1.5) dazwischen, groß (2.0–2.6) U-Tal — das
ersetzt die frühere kategoriale Spalte "Talform: V / U / Schlucht / flach"
aus dem alten A.1 (unten, A.4, bleibt nur noch für Größen offen, die
diese Datei nicht enthält).

> Nachgebessert 2026-08-06: die erste Eichung von Skerrheims `hoehe_m` (2
> Seeds) stand auf 196.6 m; über 5 Seeds gemessen fehlten dabei 10
> Prozentpunkte Wasseranteil. Der oben stehende Wert (-52.0 m) ist bereits
> die korrigierte, aktuelle Fassung.

**Was hier fehlt:** eine Untergrund-/Gesteinsspalte wie im alten §2 gibt es
für die neun Regionen nicht. Gemessene Pro-Region-Konstanten dieser Art
(`MESSWERTE_JE_REGION` u.ae.) liegen laut Commit `212dda0` (Ticket #29)
weiterhin literal in `core/vektor_kueste.py` — bewusst außerhalb des
Zuschnitts von #29, und auch von diesem Ticket nicht bewegt.

### A.2 Wasser, Küste, Klima

| Region | wasser_soll (%) | flaeche_soll | kuestenform | hang_trockenheit |
|---|---|---|---|---|
| Clonagh | 0.0 | 1.11 | 1.45 | 0.15 |
| Skerrheim | 20.0 | 1.09 | 1.90 | 0.10 |
| Morobora | 0.0 | 1.00 | 0.45 | 0.20 |
| Estrande | 45.0 | 1.44 | 1.00 | 0.12 |
| Nevadin | 0.0 | 1.21 | 1.00 | 0.30 |
| Nebelrode | 0.0 | 1.09 | 0.45 | 0.20 |
| Samarcia | 0.0 | 0.92 | 1.10 | 0.45 |
| Macchia | 40.0 | 1.39 | 1.00 | 0.35 |
| Thalassia | 65.0 | 1.22 | 1.00 | 0.30 |

`flaeche_soll` ist ein EINGABE-Faktor für die Grundfläche, nicht der
Zielwert selbst — die eigentlichen Flächenziele (Nullsummenspiel, eine
Region gewinnt nur, was eine andere abgibt) stehen in
`tests/smoke_test_regionen_fairness.py`. `hang_trockenheit` ist eine
RELATIVE Modulation: ein voller Südhang wird um diesen Anteil trockener als
die Ebene, ein voller Nordhang um denselben Betrag feuchter.

| Region | temp_mittel_m0 (°C, Meereshöhe) | temp_spanne (K) | niederschlag_mm (mm/Jahr) | Wind |
|---|---|---|---|---|
| Clonagh | 10.9 | 9.5 | 1200 | sh. B.5 |
| Skerrheim | 8.6 | 13.0 | 2250 | sh. B.5 |
| Morobora | 3.8 | 29.0 | 600 | sh. B.5 |
| Estrande | 13.6 | 14.0 | 780 | sh. B.5 |
| Nevadin | 12.8 | 18.5 | 850 | sh. B.5 |
| Nebelrode | 11.2 | 18.5 | 640 | sh. B.5 |
| Samarcia | 20.0 | 19.0 | 430 | sh. B.5 |
| Macchia | 16.9 | 17.5 | 800 | sh. B.5 |
| Thalassia | 19.7 | 14.0 | 480 | sh. B.5 |

`temp_mittel_m0`/`temp_spanne` sind KEINE gemessenen Kartenwerte, sondern
gegen die Regionsmischung vorkompensierte Eingabewerte — ein als Region
geführtes Pixel trägt im Mittel fremdes Gewicht der Nachbarregionen und
wird dadurch verdünnt. Geeicht am 2026-08-07 (nach Einführung von
KLIMA_SCHAERFE, 3 Seeds, 7 Runden), so dass die GEMESSENEN Werte auf der
Karte die Vorbildorte treffen; `tests/smoke_test_regionen_welt.py` prüft
die gemessenen Werte gegen diese hier. `niederschlag_mm` ist dagegen direkt
der Zielwert: das Wettersystem normiert das fertige Niederschlagsfeld
unmittelbar darauf (`weather_generator._je_region_auf_mittel`). Für Wind
gilt bereits die vollständige, auf die neun Regionen umgestellte
Ziel/Stand-Tabelle in B.5 — die dortigen Zielwerte sind bit-identisch mit
`wind_mittel_ms` in `core/daten/regionen_welt.toml`.

### A.3 Bekannte Ziel/Ist-Abweichungen mit Datum

| Region | Größe | Ziel | Ist | Datum | Erklärung |
|---|---|---|---|---|---|
| Morobora | Hangneigung (Regionsmittel) | 3.0 (Sollhang, reine Region) | 3.4 (Regionsmittel, ~33 % Fremdgewicht der Nachbarn) | 2026-08-06 | Bei Mischungs-Reinheit > 0.95 wird der Sollhang exakt getroffen. `relief_m` bleibt bewusst auf die REINE Region geeicht statt auf die Mischung — sonst würde das Relief auf rund 30 m gedrückt, eine Region ohne Charakter, nur damit eine Kennzahl stimmt, die etwas anderes misst. |
| alle neun | Windgeschwindigkeit | s. B.5 | s. B.5 | 2026-08-04 (Seed) | Bereits vollständig migriert, eigene Tabelle mit allen neun Regionen. |
| Gesamt | Wasserbilanz | < 1 % | +10.7 % | s. B.6 | Ursache noch ungeklärt. |

### A.4 Noch offen — je Region zu messen

Nicht Teil von `core/daten/regionen_welt.toml`, weil es keine Reglergrößen
sind, sondern gemessene Ergebnisse der fertigen Karte. **Noch offen —
gemeinsam mit dem Nutzer zu füllen**, sobald ein Messverfahren dafür steht.

| Größe | Einheit | Anmerkung |
|---|---|---|
| Anzahl Flüsse | pro 15×15 km | wieviele erreichen den Kartenrand |
| Flussbreite | m, größter Lauf | folgt aus dem Durchfluss |
| Biome | Flächenanteile in % | 3–5 dominante, je Region |

---

## B. Zielwerte je Komponente

Jede Komponente hat ein eigenes Zielbild, eine Messgröße und einen Stand.
Ohne Messgröße gibt es kein Ziel, sondern nur eine Meinung.

### B.1 Terrain

**Ziel:** stetiges Gefälle wie in realen Höhenprofilen. Keine Nadelgrate.
Talböden nicht künstlich flach, sondern von der Erosion geformt.

| Messgröße | Ziel | Stand 2026-07-29 |
|---|---|---|
| Nadeln je Querschnitt (Sprung > 3 % Relief) | ≈ rohes Gelände | **6325 gegen 3962 roh** — die Erosion erzeugt sie |
| mittlere Steigung | passend zu Relief/Ausdehnung | 118 % gegen 8.3 % im Referenzprofil |
| Höhenspanne | genau `BASE_ELEVATION_M` … `AMPLITUDE` | erfüllt |

**Erkenntnis:** `redistribute_power` ist **nicht** die Ursache der Nadeln und
flachen Böden. Querschnitte über 3.5 / 2.0 / 1.4 / 1.0 zeigen identische Form,
nur vertikal gestreckt. Vorher behauptet, per Bild widerlegt.

### B.2 Erosion

**Ziel:** dichte verästelte Entwässerung, breite helle Talsohlen, Kämme
erhalten. Vorbild: die Luftbilder verflochtener Gebirgsflüsse.

| Messgröße | Ziel | Stand |
|---|---|---|
| `beta` (Hangneigung über Einzugsgebiet) | −0.4 … −0.7 | −0.638 ✓ |
| Netzgröße (größte Kanalkomponente) | möglichst groß | 2055 px |
| Zusammenflüsse | viele | 770 |
| Randabfluss (Kanalzellen am Rand) | > 0 | 369 ✓ |
| Relieferhalt | Ausgang ≈ Eingang | 400→400, 1400→1395 ✓ |
| Massenbilanz | < 1 % | 0.18 % ✓ |

**Offener Konflikt:** Böschungserosion beseitigt die Nadeln vollständig (6325 →
**0**), verdreifacht das Netz und verbessert `beta` auf −0.713 — sieht dem
Nutzer aber nicht zu (»als ob Säure drübergeschüttet wurde«), auch beim
kleinsten Wert. Ein Ersatz muss **nur den neuen Abtrag** räumlich verteilen
(Rinnenbreite), nicht das Gelände.

### B.3 Weather — Temperatur

**Ziel:** die abgestimmte Klimatologie treffen, über alle Breiten und Monate.

| Messgröße | Ziel | Stand |
|---|---|---|
| Abweichung vom Sollwert 20–50° | < 2 K | 0.0 … 2.0 K ✓ |
| Abweichung Tropen / Pole | < 2 K | +1.4 / −3.3 |
| Jahresschwankung 40° | 23 K | 22.6 K ✓ |
| Höhengradient | 6 K/km | erfüllt |

### B.4 Weather — Niederschlag

**Ziel:** Doppelstruktur über die Breite — feuchter Äquator, **trockener
Subtropengürtel bei 20–33°**, wieder feuchtere Westwindzone.

| Messgröße | Ziel (Verhältnis zum Äquator) | Stand |
|---|---|---|
| 20° | 0.18 | 0.35 |
| 30° | 0.20 | 0.31 |
| 40° | 0.32 | 0.36 |
| Absolutwert Äquator | 2200 mm | 63 mm (Faktor fehlt) |

### B.5 Weather — Wind

**Festgelegt 2026-08-11.** Anders als 3.3/3.4 (aus der Zeit vor der
Weltkarte, latitudenbasiert) gilt dieser Abschnitt für das AKTUELLE
9-Regionen-System ([13_KLIMA_UND_BIOME.md](13_KLIMA_UND_BIOME.md)) — Zielwerte je Region, nicht je
Breite, aus realen Referenzorten (Gedächtniswerte, ±1 m/s, vor endgültiger
Abnahme gegenzuprüfen wie die Klimatabelle selbst).

**Ziel:** Jahresmittel je Region treffen, mit Luv/Lee-Kontrast am Gebirge und
Böigkeit, OHNE jahreszeitliche Drehung (Richtung bleibt fest — passt zum
bereits etablierten Luv/Lee-Muster bei 1.3, nur die STÄRKE schwankt übers
Jahr, analog zur Temperaturkurve aus 1.1. "Festlegung statt Simulationskreis",
[13_KLIMA_UND_BIOME.md](13_KLIMA_UND_BIOME.md)).

| Region | Referenzort | Ziel | Stand (256px, Seed 20260804) | Charakter |
|---|---|---:|---:|---|
| Clonagh (Kelten) | Cork | 4.5 m/s | 4.32 ✓ | exponiert atlantisch, windig |
| Skerrheim (Wikinger) | Bergen | 3.0 m/s | 3.01 ✓ | fjordgeschützt, aber Böen vom Meer |
| Morobora (Slawen) | Wologda | 3.2 m/s | 3.22 ✓ | kontinental, ruhiger |
| Estrande (Franken) | La Rochelle | 4.5 m/s | 4.29 ✓ | atlantisch exponiert |
| Nevadin (Alemannen) | Chur | 2.2 m/s | 2.67 ✓ | Tal geschützt, aber Föhn-Spitzen |
| Nebelrode (Sachsen) | Bamberg | 3.0 m/s | 3.03 ✓ | gemäßigt kontinental |
| Samarcia (Andalusier) | Madrid | 3.0 m/s | 3.24 ✓ | Hochebene, mäßig |
| Macchia (Italiener) | Rom | 3.5 m/s | 3.56 ✓ | küstennah |
| Thalassia (Byzantiner) | Iraklio | 4.5 m/s | 4.37 ✓ | Ägäis, Meltemi-Böen im Sommer |

Erreicht über `_wind_regional_faktor()` (core/weather_generator.py): das
REGIONALE MITTEL der simulierten Windgeschwindigkeit wird direkt auf
`wind_ziel_map` normiert, exakt das "direkt auf den Zielwert normieren"-
Prinzip aus 1.3 (Niederschlag). Alle neun Regionen innerhalb 0.5 m/s ihres
Ziels, Rangfolge stimmt (Nevadin ist die windärmste Region). Gesichert in
`tests/smoke_test_weather_wind_regions.py`.

**Luv/Lee-Kontrast am Gebirge: Ziel 1.5–2×, NICHT verlässlich erreicht.**
Ein multiplikativer Term (`_wind_luv_lee_faktor`, Hangneigung in
Windrichtung, analog zum Niederschlags-Luv-Term) ist eingebaut. Gemessen im
Nevadin: die vorhandene 3-Schicht-Simulation hat selbst schon eine
terraingetriebene Windstruktur (eigene Ablenkungs-/Speedup-Terme), die mit
diesem einfachen Ansatz ANTIKORRELIERT (-0.53 Korrelationskoeffizient
Faktor↔Basisgeschwindigkeit) statt neutral zu sein - der Zusatzterm wird
dadurch weitgehend neutralisiert. Eine verlässliche Lösung bräuchte
entweder eine Abstimmung mit der internen Terrain-Kopplung der Simulation
oder einen stärkeren, die Basis dominierenden Faktor - beides nicht Teil
dieser Änderung. Deshalb NICHT in der Zusicherung geprüft.

**Böigkeit:** ein Böenfaktor statt eines Turbulenzfelds, Spitze ≈ 1.4–1.6×
des Mittels - als Designentscheidung festgehalten, noch NICHT als eigenes
Feld gebaut (kein Konsument dafür bisher).

**Richtung:** vorherrschend West, fest — keine jahreszeitliche Drehung.

### B.6 Water — Gewässer

**Ziel:** 2–3 Bäche vereinigen sich zu einem Fluss, dieser trifft weiter unten
auf einen weiteren. Breite wächst mit dem Durchfluss. Mäander in der Talsohle.
Gelegentlich ein See. Flüsse verlassen die Karte.

| Messgröße | Ziel | Stand |
|---|---|---|
| Seenfläche | > 0 | **0.0 %** |
| Mäander (Sinuosität der Hauptläufe) | > 1.2 | nicht gemessen |
| Breitenvariation | Faktor > 3 vom Bach zum Fluss | nicht gemessen |
| Wasserbilanz | < 1 % | **+10.7 %** ungeklärt |

### B.7 Biome

**Ziel:** zusammenhängende Zonen, keine gestreuten Einzelpixel. Zwei Stufen —
Klimazone aus Breite und Höhe (großflächig), Biom daraus aus Feuchte, Neigung
und Lage im Tal.

| Messgröße | Ziel | Stand |
|---|---|---|
| Zusammenhang (mittlere Flächengröße je Biom) | groß | nicht gemessen |
| Höhengürtel wandern mit der Breite | Baumgrenze 3600 m am Äquator, 900 m bei 60° | umgesetzt |
| Sumpf breitengradabhängig | in hohen Breiten häufiger | umgesetzt |

---


## Wer diese Werte bewacht

| Test | Was er prüft | Dauer |
|---|---|---|
| `tests/smoke_test_regionen_welt.py` | die neun Regionen gegen Sollhang und Wasseranteil, dazu Naht, Pixelunabhängigkeit und GPU/CPU-Parität | ~2 min |
| `tests/smoke_test_regionen_fairness.py` | die eigentlichen Flächenziele (Nullsummenspiel) | |
| `tests/smoke_test_weather_wind_regions.py` | die Windziele aus B.5 | |

`smoke_test_regionen_welt.py` gehört in die Prüfliste **jeder** Änderung an
`weltfeld()`. Er ist der empfindlichste Wächter für die Geländeform — die
Küstenarchetypen wurden 2026-08-12 gegen vier Terrain-Tests geprüft und für
regressionsfrei erklärt; dieser Test war nicht darunter und schlägt seither
fehl.

**Stand 2026-09-23** (384 px, Mittel aus 5 Seeds): vier bekannte Befunde —
Macchia (Hang 18.5 statt 14.5), Thalassia (Hang 14.8 statt 11.5, Wasser 48.8
statt 65), Regionsgrenzen steiler als die Regionen selbst (1.600 gegen
0.891), und die Landschaft hängt noch an der Pixelzahl (r = +0.94).
