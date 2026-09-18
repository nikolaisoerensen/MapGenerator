# Sitzungslog

Fortlaufendes Protokoll der Arbeitssitzungen. **Neueste Sitzung oben.**
Zweck: bei einem Wechsel auf einen anderen Account ohne Rueckfragen
weiterarbeiten koennen.

Aufbau je Eintrag: was gemacht wurde, was gemessen wurde, was NICHT
funktioniert hat, was offen bleibt. Vermutungen sind als solche
gekennzeichnet; alles andere ist geprueft.

---

# 2026-09-18 — Ticket #30: CPU-Erosion bei 1024 px gemessen, Faktor-385-Praemisse war schon behoben

## Ausgangslage

Ticket #30 verlangte zwei Messungen, bevor irgendetwas repariert wird: wie
lange der CPU-Erosionspfad bei 1024 px gegen die GPU braucht, und ob er im
Betrieb ueberhaupt genommen wird. Der im Ticket-Text zitierte "Faktor 385"
war zu diesem Zeitpunkt schon ueberholt - der Github-Kommentar vom 17.09.2026
auf Issue #30 (aus der vorangegangenen Nachtsitzung) hatte das bereits
festgehalten: die Paritaet ist seit dem 27.08.2026 hergestellt (Eintrag
"Erosion wieder eingeschaltet", unten), am 16.09.2026 mit echter GPU erneut
bestaetigt. Erneut laufen lassen
(`tests/smoke_test_erosion_gpu_parity.py`, echte GPU, kein Fallback)
bestaetigt das ein drittes Mal: PASS, Einzelschritt-Abweichung 0.

Die zwei tatsaechlich noch offenen Kriterien aus dem Ticket waren damit nur
noch: Laufzeit messen, und klaeren, ob der CPU-Pfad im Betrieb je gebraucht
wird.

## Gemessen

1024x1024, Terrain wie im bestehenden Paritaetstest gebaut (Gauss-verwischtes
Rauschen, hier Relief 974 m), Produktions-Vorgaben (`max_steps=8000`,
`convergence_threshold=1e-6`), zwei Wiederholungen wegen der bekannten
Maschinenschwankung (Faktor 2-3, siehe `docs/TESTBERICHT.md`):

| Lauf | GPU (vollstaendig bis Konvergenz) | CPU (Rate hochgerechnet) |
|---|---|---|
| 1 | 7,70 s, 750 Schritte, konvergiert, Bilanz -0,19 % | 0,64 s/Schritt (gemessen ueber 25 Schritte) x 750 = 497 s (8,3 min) |
| 2 | 16,43 s, 750 Schritte, konvergiert, Bilanz -0,19 % | 0,81 s/Schritt (gemessen ueber 100 Schritte) x 750 = 607 s (10,1 min) |

Die CPU-Rate wurde NICHT durch einen echten Lauf bis zur Konvergenz bei
1024 px gemessen - laut dem Kommentar bei
`HydraulicFieldSimulator.MAX_CPU_RESOLUTION` wuerde das "Stunden" dauern und
haette das Nachtbudget gesprengt. Stattdessen: kurze, ungeklemmte Laeufe (25
bzw. 100 Schritte, `StepProbe`-Technik wie im Paritaetstest) fuer die
Sekunden/Schritt-Rate, hochgerechnet auf die Schrittzahl, die der echte
GPU-Lauf bis zur Konvergenz brauchte. Hochrechnung auf die volle Obergrenze
`max_steps=8000` (worst case, keine Konvergenz erreicht): 88-108 Minuten.

**Ergebnis: Faktor 37-65x, CPU-Vollauf 8-10 Minuten fuer einen realen
konvergenten Lauf gegen 8-16 Sekunden auf der GPU.** Das trifft die
Groessenordnung aus dem Ticket-Beispiel ("6 Minuten gegen 25 Sekunden") -
eindeutig der Fall "um Groessenordnungen langsamer".

Messskript nicht eingecheckt (Wegwerfskript dieser Sitzung). Es hebt
`HydraulicFieldSimulator.MAX_CPU_RESOLUTION` NUR zur Laufzeit im eigenen
Skript an (256 auf 4096, in einem try/finally wieder zurueckgesetzt), um
ueberhaupt einen CPU-Lauf bei 1024 px anstossen zu koennen - keine Aenderung
an `core/erosion_generator.py`.

## Wird der CPU-Pfad im Betrieb genommen?

Nachverfolgt in `core/erosion_generator.py`:

- **Normalfall (GPU vorhanden):** `ErosionSystemGenerator` bekommt den
  `shader_manager` immer vom `GenerationOrchestrator`
  (`managers/generation_orchestrator.py`, Zeile 1346). Ist `has_gpu_path()`
  True, laeuft ausschliesslich die GPU. Die CPU-Schleife wird nicht
  erreicht.
- **Keine GPU ueberhaupt** (`has_gpu_path()` False, z.B. kein OpenGL 4.3):
  `simulate()` verweigert bei Kartengroessen ueber `MAX_CPU_RESOLUTION`
  (256 px) SOFORT mit einer lauten `ValueError` (Zeile 606-612). Bei der
  Produktionsgroesse 1024 px wird die langsame CPU-Schleife also gar nicht
  erreicht - die vorhandene Sperre wirkt wie vorgesehen.
- **Ein neuer, schmaler Fund:** Ist eine GPU zwar vorhanden UND registriert
  (`has_gpu_path()` True), schlaegt aber EIN Abschnitt mitten im Lauf fehl
  oder reisst den 30-s-Timeout von `GPUWorker.submit()` (`_simulate_gpu()`
  gibt dann `None` zurueck, mit `logger.warning(...)`), faellt `simulate()`
  danach OHNE erneute Groessenpruefung in die CPU-Schleife (Zeile 619-624).
  Bei 1024 px wuerde das den oben gemessenen 8-10-Minuten- bis
  88-108-Minuten-Lauf ausloesen - geloggt, aber ohne dieselbe Sperre, die
  fuer den "GPU fehlt ganz"-Fall bereits existiert. Ein seltener Pfad
  (GPU-Treiberfehler, Abschnitts-Timeout), kein Alltagsfall, und keine
  Paritaetsfrage (die Physik stimmt exakt ueberein) - aber der einzige
  verbleibende Rueckfall, der nicht durch dieselbe Sperre wie die anderen
  beiden Faelle abgefangen ist. Vermerkt in `docs/OFFENE_PUNKTE.md` 10.7,
  NICHT in dieser Sitzung behoben (siehe unten, warum).

## Entscheidung nach Ticket-Vorgabe

Der Fall "um Groessenordnungen langsamer und im Normalbetrieb nicht genommen"
trifft zu - mit der oben genannten Einschraenkung fuer den seltenen
Abschnitts-Fehlschlag-Fall. Eine Aenderung an `core/erosion_generator.py`
(dieselbe `MAX_CPU_RESOLUTION`-Pruefung auch im Fallback-Zweig von
`simulate()` einziehen) waere die naheliegende Behebung, ist aber
ausdruecklich NICHT Teil dieser Sitzung: die Nachtvorgabe fuer Ticket #30
untersagt jede Aenderung an der Erosionsberechnung selbst ohne Ruecksprache
mit dem Nutzer, unabhaengig davon, wie klein die Aenderung aussieht.
`tests/smoke_test_erosion_gpu_parity.py` bleibt unveraendert - er ist
bereits gruen und deckt bereits genau den Paritaetsfall, um den es in der
urspruenglichen Ticket-Formulierung ging.

## Ergebnis fuer Ticket #30

- [x] Laufzeit CPU gegen GPU bei 1024 px gemessen (siehe Tabelle oben).
- [x] Belegt: CPU-Pfad wird im Normalbetrieb (GPU vorhanden ODER GPU fehlt
      ganz) nicht genommen; einzige Ausnahme ist ein seltener
      GPU-Abschnitts-Fehlschlag mitten im Lauf.
- [x] Paritaet besteht bereits (das war die eigentliche Ausgangsfrage, siehe
      Ticket-Kommentar vom 17.09.2026), `smoke_test_erosion_gpu_parity.py`
      erneut gruen (echte GPU, kein Fallback).
- [ ] Der CPU-Pfad ist nicht zusaetzlich "laut stillgelegt" fuer den
      schmalen Fallback-Fall - er verweigert bereits laut bei fehlender GPU
      oberhalb 256 px, aber nicht in diesem einen Zweig. Als
      `docs/OFFENE_PUNKTE.md` 10.7 vermerkt, keine Aenderung an der
      Erosionsberechnung vorgenommen.

## Lehre

Dieselbe Lehre wie beim Eintrag vom 16.09.2026, aus einer anderen Richtung:
ein Ticket, dessen Praemisse laengst korrigiert wurde, verlangt nicht
automatisch keine Arbeit mehr - die Restpunkte, die im Github-Kommentar vom
17.09. schon als "kleiner als gedacht" markiert waren, waren trotzdem echte,
messbare Arbeit, und die Messung selbst hat einen neuen, kleinen echten Fund
gebracht (10.7). Eine ueberholte Praemisse ist ein Grund, den Rahmen zu
verkleinern - kein Grund, gar nicht erst nachzusehen.

---

# 2026-09-16 — Testbericht behauptete einen laengst behobenen Befund (Fund beim Abschluss-Testlauf)

## Befund

Beim einmaligen Voll-Testlauf am Ende der Nacht (`tools/testlauf.py`, 73
Dateien) war `smoke_test_erosion_gpu_parity.py` gruen - obwohl
`docs/TESTBERICHT.md` Abschnitt 3 ihn zu diesem Zeitpunkt noch als
"echter Paritätsbruch, Faktor 385" auffuehrte, und ich selbst diese Zeile
wenige Stunden vorher in Ticket #63 unveraendert stehen liess.

Einzeln nachgemessen (`tests/smoke_test_erosion_gpu_parity.py` direkt
gestartet, echte GPU, kein Fallback): ein Schritt GPU gegen CPU Abweichung
0, Langlauf beide Seiten 0,1 m Export - PASS auf ganzer Linie.

**Ursache der falschen Doku:** der Fund war bereits am 27.08.2026 behoben
(`docs/SITZUNGSLOG.md`, Eintrag "Erosion wieder eingeschaltet" desselben
Tages) - ein zu grobes GPU-Meldeintervall (`PROGRESS_REPORT_INTERVAL = 500`
statt `CONVERGENCE_CHECK_INTERVAL = 25`), kein echter Zahlenfehler. Die
Behebung geschah aber offenbar NACH dem Testlauf, aus dem
`docs/TESTBERICHT.md` an jenem Tag geschrieben wurde - die Datei wurde
seither nie neu erzeugt und blieb drei Wochen falsch.

## Behoben

`docs/TESTBERICHT.md` (Abschnitt 3, Tabelle und Fliesstext), `docs/AUFRAEUMPLAN.md`
(Abschnitt 4.7 und die Ziel-Tabelle) und der Nachtrag in
`docs/SPEZIFIKATION.md` §7 korrigiert. `docs/NACHTBETRIEB.md` und
`docs/SOLLBESCHREIBUNG.md` enthalten dieselbe veraltete Behauptung
(Faktor 385 / "ungeklaerter Faktor 385"), sind aber hart gesperrt und daher
nachts nicht anfassbar - offener Punkt fuer den Nutzer oder eine Tagsitzung.

## Lehre

Dieselbe Lehre wie bei Ticket #28/#63, nur am anderen Ende: nicht nur eine
falsche Erklaerung fuer einen roten Test ueberlebt eine Behebung, sondern
auch eine korrekte Erklaerung fuer einen inzwischen gruenen Test kann liegen
bleiben, wenn der Bericht nach der Behebung nie neu geschrieben wird. Ohne
den Voll-Testlauf am Nachtende waere das nicht aufgefallen - ein Grund mehr,
ihn nicht zu ueberspringen.

---

# 2026-09-16 — Uebersichts-Reiter: geloescht statt gebaut (Ticket #6)

## Entscheidung

`gui/tabs/overview_tab.py` enthielt zwei getrennte Dinge: einen
funktionierenden Teil (Weltstatistik, Qualitaetspruefung, Export, Regler-
Zusammenfassung) und eine seit jeher tote "Composite View"-Funktion
(vier Buttons fuer Gesamtwelt-/Klima-/Zivilisations-/Geologie-Ansicht).
Die toten Methoden riefen ausschliesslich `self.map_display.XXX()` auf -
ein Attribut, das in dieser Klasse **nirgends** zugewiesen wird. Jeder
Klick waere lautlos ins Leere gelaufen (`AttributeError`, gefangen durch
den umgebenden `hasattr(self, 'map_display')`-Wächter, der immer
zutrifft und die Methode sofort verlaesst).

Ticket #6 fragte "loeschen oder bauen" - **entschieden: loeschen, aber
NUR die tote Composite-View-Funktion, nicht den ganzen Reiter.** Der
Rest des Reiters (Statistik, QS, Export, Parameter-Zusammenfassung) ist
echte, benutzte Funktionalitaet und war vom Befund nicht betroffen. Eine
woertliche Lesart des Tickets ("Der Reiter ist wirkungslos") haette den
kompletten Reiter nahegelegt - das haette funktionierenden Code
zerstoert, den niemand kaputt gemeldet hat. `docs/SPEC_OVERLAYS.md`
(2026-09-14) nennt genau diese Composite-View-Stelle explizit als "Out
of Scope" fuer die dortige Overlay-Architektur und verlangt "einen
eigenen Punkt" dafuer - das ist der Ursprung dieses Tickets.

## Was entfernt wurde

`CompositeViewControlsWidget`-Klasse komplett, ihre Einbindung in
`setup_overview_ui()`, der Refresh-Aufruf in `on_data_updated()`, das
`setEnabled()` in `check_world_completeness()`, `update_composite_view()`
und alle fuenf `render_*_view()`-Methoden, sowie der zugehoerige
Composite-Export-Block in `export_png_collection()`. Modul-Docstring
entsprechend korrigiert (Composite-View-Behauptung entfernt, Begruendung
mit Verweis hierhin ergaenzt).

## Geprueft

* `import gui.tabs.overview_tab` laeuft weiter fehlerfrei.
* Grep auf `composite|Composite|map_display\b|QComboBox|QCheckBox` in der
  Datei zeigt keine Ueberbleibsel des geloeschten Features mehr - nur
  unbetroffene Treffer (neuer Docstring-Text, `format_combo` im
  unveraenderten Export-Widget).
* `tests/smoke_test_display_methoden_existieren.py` erfasst diesen
  Befund NICHT (sein `hasattr()`-Regex greift nur, wenn der
  Objekt-Ausdruck "display"/"map_display"/... im Text enthaelt - hier war
  nur der aeussere `hasattr(self, 'map_display')` betroffen, dessen
  Objektausdruck `"self"` lautet). Ein zukuenftiger Erreichbarkeitstest
  (SPEC_OVERLAYS.md, Testentscheidung 3) waere der richtige Ort dafuer.

---

# 2026-08-27 — Live-Vorschau im Flussreiter (Schritt 3 von 5)

Damit steht die Wochenvorgabe vom 2026-08-26 vollstaendig:
**Regionen -> Kontinent -> Flussnetzwerk -> Terrain**, jeder Reiter mit
eigener Live-Vorschau, in dieser Reihenfolge in der Reiterleiste.

## Was gebaut wurde

`gui/tabs/river_tab.py`: Haekchen *Live-Vorschau (128 px)*, ein
Einzelschuss-Taktgeber mit 250 ms Verzoegerung, `_vorschau_rechnen()`.

**Das Grundgelaende wird zwischengespeichert.** Die fuenf Regler dieses
Reiters (Talabstand, Talbreite, Taltiefe, Talform, Flusskosten) aendern
`weltfeld()` NICHT - nur das Netz und die Taeler darin. Ohne den
Zwischenspeicher kostete jeder Reglerzug den vollen Aufbau.

## Gemessen (128 px, dieselbe Maschine)

| | Zeit |
|---|---:|
| erste Vorschau, mit Grundgelaende | 2.92 s |
| jede weitere, Grundgelaende behalten | 0.67 - 0.75 s |

Vorher gemessen, WARUM 128 px und nicht 256: `flussnetz()` beherrscht die
Rechnung mit 0.50 s bei 128 px. Die am selben Tag hinzugekommene
Linienbreite nach Wassermenge kostet davon 0.05 s - ausdruecklich
nachgemessen, weil sie in derselben Schleife sitzt und damit der erste
Verdaechtige gewesen waere.

Die Vorschau ist beim Oeffnen **aus**. Ein Reiter, der beim Anklicken drei
Sekunden rechnet, macht das Programm beim Start zaeh.

## Ein Fehler beim Bauen, mit einer Lehre

    RuntimeError: super-class __init__() of type RiverTab was never called

`QTimer(self)` stand VOR `super().__init__(...)`. Ein QObject als
Elternteil muss zu dem Zeitpunkt fertig gebaut sein. Der Import blieb
sauber - der Absturz kam erst beim Bauen des Reiters, also genau dort, wo
er am 2026-08-26 schon einmal die ganze Reiterleiste verschoben hatte.
Behoben, mit Begruendung im Code.

## Absicherung

`tests/smoke_test_fluss_vorschau.py` (neu, 8 Pruefungen, alle gruen):

* sie ist beim Oeffnen aus und hat nichts vorgerechnet,
* der Zwischenspeicher haelt (Grundgelaende bitgleich),
* ein Reglerzug bleibt unter 1.6 s (gemessen 0.67 s),
* **ein Reglerzug aendert das GELAENDE** (Talbreite 0.2 gegen 2.0:
  3.8 m im Mittel) - geprueft wird das Ergebnis, nicht der Parameter,
  dieselbe Lehre wie bei `smoke_test_regionsregler_wirken.py`,
* die Vorschau geht ueber den gewoehnlichen `_show_data(..., "heightmap")`
  und damit ohne Sonderweg durch 2D **und** 3D,
* Ausschalten verwirft den Zwischenspeicher.

## Doku

`docs/PRUEFLISTE_LIVE.md` neu aufgebaut: **Teil A** ist der neue Ablauf
(A.0 Reiterfolge, A.1 Regionen, A.2 Kontinent, A.3 Fluesse, A.4 die vier
neuen Karten je in 2D UND 3D, A.5 Stufenschalter, A.6 Nevadin-Spitzen,
A.7 die offene Frage zu den Kuestentypen in der Regionsvorschau), **Teil
B** ist die unbestaetigte Liste vom 2026-08-24.

## Voller Testlauf: 60 von 70, zwei Befunde bearbeitet

**1. Vier Karten liefen in 3D ohne Farbtafel** (neu, selbst verursacht).
`smoke_test_layer_2d_3d_parity.py` fand, dass `river_water`,
`river_order`, `hinterland_height` und `voronoi_map` zwar in
`layer_ranges` standen, aber nicht in `_LAYER_RANGE_KEY_MAP`
(`gui/widgets/map_display_3d.py`). Die 3D-Ansicht liest die erste Tabelle
NICHT direkt - sie geht ueber die zweite.

Die Folge waere kein leeres Bild gewesen: `_colorize_layer()` findet
keinen Schluessel, faellt auf Auto-Skalierung ohne Farbtafel zurueck und
zeichnet trotzdem etwas Plausibles - nur mit anderen Farben und anderer
Skala als 2D. **Zum vierten Mal derselbe stille Rueckfall.** Behoben.

Bemerkenswert: gefunden hat es nicht der Blick aufs Bild, sondern der
Test, der genau diese Doppelbuchhaltung bewacht.

**2. Vendee-Straende -16.7 P war ein Fehler IM TEST.** Gemessen wurden
zwei Enden der Kette bei zwei Aufloesungen:

| Vendee | Saatanteil | Laengenanteil |
|---|---:|---:|
| 384 px | -5.0 | **-20.1** |
| 768 px | -4.9 | **-0.2** |

Saatende gleich, Laengenende weg - die Zuordnung stimmt, der Verlust
entsteht im Einschmelzen kurzer Segmente. Bretagne-Klippen spiegelt es
exakt (+11.4 bei 384 px, -5.8 bei 768 px). `MIN_SEGMENT_M` (750 m) ist
absolut; Vendee hat mit 0.18 km die kuerzeste Reichweite der drei.

Der Test prueft jetzt beide Enden mit zwei Grenzen: Saatanteil scharf
(10 P, schlechtester gemessener Wert 5.2 ueber 24 Archetypen),
Laengenanteil weich (22 P). Ein systematisch ausfallender Archetyp liegt
25 bis 50 Punkte daneben und faellt durch beide.

Zwei Vermutungen wurden dabei ausgeschlossen, bevor gemessen wurde: die
Profilstauchung von heute (`kandidaten_gewicht` haengt nur vom
Konturabstand ab) und ein zu kleines Budget (Vendee ist der
Resteinsammler seiner Region). Notiert als OFFENE_PUNKTE 6b.

**Dabei etwas verloren:** `docs/TESTBERICHT.md` wurde ersetzt statt
ergaenzt. Die uncommittete ausfuehrliche Fassung vom 24.08. mit der
Herleitung der Laengenquote ist weg; die committete Vorfassung steht in
`git show HEAD:docs/TESTBERICHT.md`, ein Zeiger darauf im Kopf der Datei.

## Erosion wieder eingeschaltet (Nutzervorgabe "Erosion los")

`EROSION_AKTIV = True`. Sie stand seit dem 30.07. auf False, mit der
Begruendung: *"sie ERZEUGT die Becken, die sie aufloesen soll"*. Die war
richtig - fuer das Gelaende von damals. Sie traegt nicht mehr, seit das
Flussnetz ein entwaessertes Gelaende vorlegt.

**Gemessen** (256 px, Seed 20260804, die fuenf Kennzahlen aus
`smoke_test_erosion_quality.py`, auf drei Gelaenden statt einem):

| Gelaende | Top5 | Netz | Krater | Ebenen | Schritte |
|---|---:|---:|---:|---:|---:|
| synthetisch (so misst der Test) | 0.397 | 105 | 12 | 4.0 % | 875 |
| echt, ohne Flussnetz | 0.899 | 546 | 1 | 23.3 % | 125 |
| echt, MIT Flussnetz | 0.909 | 273 | **3** | 23.7 % | 100 |

Drei Krater von 65536 Pixeln. Die drei roten Befunde von
`erosion_quality` sind Eigenschaften seines SYNTHETISCHEN Testgelaendes
(`make_terrain()`), nicht der Erosion - dieselbe Lehre wie beim adaptiven
Mesh: ein Test mit ausgedachten Eingaben prueft eine ausgedachte Lage.

**Was vorher behoben werden musste - und was der Testbericht falsch
darstellte.** `smoke_test_erosion_gpu_parity` meldete "Export 38.5 gegen
0.1 m" und war als **Paritaetsbruch um Faktor 385** gefuehrt. Er war
keiner:

    EIN Schritt GPU gegen CPU:  Abweichung 0.00e+00
    Langlauf:  GPU 500 Schritte, CPU 25 Schritte

Die Physik stimmt exakt. Die GPU hat zwanzigmal laenger weitererodiert,
weil sie die Konvergenz nur alle `PROGRESS_REPORT_INTERVAL` (500) Schritte
prueft, die CPU aber alle `CONVERGENCE_CHECK_INTERVAL` (25). Der Abschnitt
des GPU-Laufs war das MELDE-Intervall des Ladebalkens.

Behoben: der Abschnitt ist jetzt das Konvergenz-Intervall, der Fortschritt
wird weiter alle 500 Schritte gemeldet. Danach beide Seiten 25 Schritte,
beide 0.1 m Export, Test gruen.

**Ohne das waere die Produktionskarte auf der GPU zwanzigmal zu stark
erodiert worden** - und zwar lautlos, weil das Ergebnis plausibel aussieht.

**Nach dem Einschalten geprueft:** `water_pipeline_full` und
`biome_preseed` gruen, `regionen_welt` mit exakt denselben vier Befunden
wie vorher, `pipeline_outputs` unveraendert bei fuenf. Keine Regression.

## Die neun Regionen heissen jetzt anders

Nutzerentscheidung 2026-08-26/27, in mehreren Runden. Der Klang soll
sagen, wo man ist:

| alt | neu | Klang |
|---|---|---|
| Huegelland | **Clonagh** | irisch |
| Fjordland | **Skerrheim** | nordisch |
| Taiga | **Morobora** | nordslawisch |
| Atlantikkueste | **Estrande** | franzoesisch (*estran* = Gezeitenzone) |
| Alpenland | **Nevadin** | raetoromanisch (*neve*, Kadenz von Engadin) |
| Mittelgebirge | **Nebelrode** | Harz (`-rode` wie Wernigerode) |
| Steppe | **Samarcia** | zentralasiatisch |
| Mittelmeer | **Macchia** | mediterran |
| Griechische Inseln | **Thalassia** | griechisch |

989 Vorkommen in 57 Dateien, gross-/kleinschreibungsgenau und
wortweise ersetzt.

**Zwei Fallen dabei, beide gepruefet:**

1. Die Vorkommen in `core/biome_generator.py` sind REGIONSSCHLUESSEL, keine
   Biomnamen - die Biome sind klein geschrieben (`nadelwald`, `macchia`).
   Ein Ersetzen mit Beachtung der Schreibweise trifft sie nicht.
2. In `docs/regionen/*/REGION.md` beschreiben "Taiga" und "Mittelgebirge"
   REALE Gegenden (Westsibirien, Kanadischer Schild, Bamberg). Dort war
   die Umbenennung falsch und wurde in vier Dateien zurueckgenommen.

**Zu beachten:** die Region *Macchia* enthaelt kuenftig ein Biom `macchia`.
Der Code trennt beide ueber die Schreibweise; beim Lesen ist es
verwechselbar.

**Danach gruen:** regionsfeld, region_tab, regionsregler_wirken,
kuestengebiete, stufen_schalter, reiter_vertrag, archetyp_verteilung,
regionen_fairness, seegliederung, weather_wind_regions. `regionen_welt`
und `weather_temperature_direktnormierung` melden Zahl fuer Zahl
dieselben Befunde wie vorher, nur mit den neuen Namen.

## Was offen bleibt


* Fuenf Regionsnamen warten auf deine Wahl.
* Ob die Regionsvorschau alle drei Kuestentypen erzwingen soll (A.7).
* Kuestenformregeln, Erosion wieder anschalten, Zwischenspeicher je Stufe,
  Umzug von `map_size`/`map_distance_km` auf den Kontinentreiter - alles
  bewusst nicht in diesem Durchgang.

---

# 2026-08-26 — Kontinentreiter (Schritt 2 von 5)

`gui/tabs/kontinent_tab.py`, registriert als **zweiter** Reiter zwischen
Regionen und Terrain - die Reiterfolge ist der Arbeitsablauf.

## Was er besitzt, und was bewusst nicht

Genau EINEN Parameter: `kontinentform`. `map_size`, `map_distance_km` und
`map_seed` gehoeren dem Terrain-Reiter. Sie hier ZUSAETZLICH anzulegen
waere ein Doppelschluessel, den `smoke_test_parameter_eindeutig.py` zu
Recht ablehnt (5/5 gruen geblieben). Ein Umzug waere ein VERSCHIEBEN, kein
Kopieren - eigener Schritt.

Den Seed liest die Vorschau vom Terrain-Reiter, statt einen zweiten zu
halten: er gehoert zur Karte, nicht zu dieser Ansicht.

## Der Regler beginnt AUSGESCHALTET

`form=None` erzeugt die eingemessene Gestalt, gegen die alle
Regionsflaechen, Wasseranteile und `smoke_test_regionen_welt` laufen - und
die liegt NICHT auf der Reglerkurve. Ein Reiter, der beim blossen Oeffnen
eine Form erzwingt, haette sie alle still verschoben. Erst ein Haken
schaltet den Regler scharf; ohne ihn liefert
`get_current_parameters()` nichts.

## Zwei Ansichten mit verschiedenen Kosten, mit Absicht

    2D   Kontinentform plus die neun Regionen    0.34 - 0.49 s
    3D   ein VOLLES weltfeld() bei 192 px        rund 1.1 s

Die 2D-Ansicht ist die Arbeitsansicht und bleibt beim Ziehen fluessig; die
3D beantwortet die andere Frage ("was wird daraus?") und laeuft nur auf
Anforderung.

## Gegengeprueft

Die Masken des Reiters sind **bitgleich** zum Direktaufruf von
`kontinentform()`, und sie unterscheiden sich zwischen den
Reglerstellungen. Beides gemessen - eine Bildmontage bei 256 px liess die
vier Stellungen zunaechst aehnlich aussehen, was ohne die Zahlen leicht als
"Regler wirkt nicht" durchgegangen waere.

`smoke_test_reiter_vertrag.py`: Reiterleiste 12, Viewport 12, Parameter 12,
Statistik 12, keine doppelte Beschriftung.

---

# 2026-08-26 — Die Regler erreichen jetzt die Karte (Schritt 1 von 5)

## Der Befund, der den Plan geaendert hat

Vor dem Bauen wurde nachgesehen, WER die Einstellungen des Regionsreiters
eigentlich liest. Antwort: **niemand.**

    grep "ueberschreibungen" ausserhalb von region_tab.py   -> 0 Treffer
    grep "kontinentform_regler" ausserhalb von weltkarte.py -> 0 Treffer

Der Regionsreiter war fertig, seine Vorschau reagierte auf jeden Regler,
und die Einstellungen kamen NIRGENDS an. Man haette eine Region einstellen,
"Generieren" druecken und dieselbe Karte bekommen koennen - ohne Absturz,
ohne Meldung, ohne roten Test. Der Reiter waere ein Spielzeug gewesen.

## Die Kette, vier Glieder

    RegionTab.get_current_parameters()
      -> ParameterManager (neu angemeldet als "region")
      -> TerrainTab.get_current_parameters() nimmt sie mit
      -> BaseTerrainGenerator -> weltfeld(regionen_ueberschreibung=...)
      -> parameterfeld() ersetzt einzelne Katalogwerte

Warum ueber den Terrain-Reiter: die Generierung holt sich
`get_tab_parameters(self.generator_type)`, also NUR den eigenen Satz. Ein
Reiter, der sich bloss anmeldet, wird bei der Erzeugung nie gefragt.

Als GESCHACHTELTES Dict unter einem Schluessel, nicht als flache
`region_<Name>_<Regler>`-Schluessel: die Regionsnamen enthalten
Leerzeichen, und eine Namensverstuemmelung waere eine zweite Kodierung, die
irgendwann von der ersten abweicht.

## Gemessen (192 px, Seed 20260804, Nevadin relief_m 1050 -> 400)

| | ohne | mit |
|---|---:|---:|
| `relief_m`-Feld im Regionskern | 728 m | **331 m** |
| Hoehenspanne im Kern | 701 m | **452 m** |
| Aenderung ausserhalb, Median | — | **0.7 m** |

Der Katalogwert 400 kommt im Feld als 331 an - der Rest ist die
Voronoi-Mischung mit den Nachbarn, und die soll so sein.

**Eine Zwischenmessung war irrefuehrend und wurde korrigiert:** "ausserhalb
im Mittel 30 m Aenderung" klang nach Ueberlauf, war aber ein Metrikartefakt
- der Mittelwert wurde vom Grenzband dominiert, in dem die Mischung zu
Recht mitzieht. Der Median liegt bei 0.7 m.

## Der neue Test prueft die WIRKUNG, nicht den Parameter

`tests/smoke_test_regionsregler_wirken.py`. Ein Test auf "der Schluessel
steht im Dict" waere an genau diesem Fehler vorbeigelaufen - der Schluessel
stand ja nirgends, und niemand hatte ihn erwartet. Geprueft wird deshalb
das `relief_m`-Feld und die Hoehenspanne der Region, plus: ohne
Ueberschreibung muss die Karte BITGLEICH zur Vorgabe sein.

## Testlage

| Test | Stand |
|---|---|
| `smoke_test_regionsregler_wirken.py` | **neu, gruen** |
| `smoke_test_reiter_vertrag.py` | gruen |
| `smoke_test_region_tab.py` | gruen |
| `smoke_test_kontinentform.py` | gruen |
| `smoke_test_regionsfeld.py` | gruen |
| `smoke_test_regionen_welt.py` | 4 Befunde (unveraendert) |

---

# 2026-08-26 — Kontinentform als Regler (Aufraeumplan 4.10)

Nutzerentwurf: *"links rund, mitte laenglich, rechts mit vielen
auslaeufern."*

`kontinentform(..., form=0..1)` und `_kontinent_gestalt()`. Der Kontinent
entsteht aus Zentralscheibe, Lappen, weicher Vereinigung und abgezogenen
Buchten; der Regler interpoliert genau diese Groessen. Auch durch
`weltfeld(..., kontinentform_regler=...)` erreichbar.

**"Laenglich" ist keine Interpolation, sondern eine zusaetzliche Zutat:**
die Lappenmitten werden entlang einer Achse gestaucht, mit Maximum in der
Reglermitte (4t(1-t)). Wer das beim Bauen uebersieht, bekommt dort einfach
etwas Halbrundes.

## Der erste Entwurf hatte die Enden vertauscht

Er interpolierte Kern, Lappen und Vereinigungshaerte - liess die
abgezogenen BUCHTEN aber fest bei 2-3 Stueck mit Radius 0.30-0.55.
**Gemessen war die Stellung "rund" damit die UNRUNDESTE von allen:**

    Rundheit (isoperimetrisch) bei form = 0.0:   0.53   statt 0.92

Die festen Buchten schnitten tief in eine Landmasse, die durch die nahen
Lappen ohnehin kompakter geworden war, und die Flaecheneichung blies sie
zusaetzlich auf. Im Bild sah das nach "irgendwie unruhig" aus - erst die
Rundheitsmessung zeigte, dass der Regler an seinem linken Anschlag das
Gegenteil von rund lieferte. Buchtenzahl und -groesse gehoeren mit in die
Gestalt.

## Gemessen (256 px, drei Seeds)

| form | Land | Rundheit | Seitenverhaeltnis | Teile |
|---:|---:|---:|---:|---:|
| 0.00 | 37.9 % | **0.916** | 0.94 | 1 |
| 0.25 | 37.9 % | 0.904 | 1.14 | 1 |
| 0.50 | 37.9 % | 0.805 | **1.26** | 1 |
| 0.75 | 38.2 % | 0.838 | 1.21 | 1 |
| 1.00 | 38.3 % | **0.380** | 0.96 | 1 |

**Die Landflaeche bleibt konstant** (37.9-38.3 %) - die Intervallhalbierung
auf `KONTINENT_KM^2` haelt, der Regler aendert die Gestalt und nicht die
Groesse. Und der Kontinent bleibt bei jeder Stellung EIN Stueck.

**`form=None` reproduziert die bisherige Gestalt bitgleich.** Das ist keine
Bequemlichkeit: an ihr haengen alle Regionsflaechen, Wasseranteile und
`smoke_test_regionen_welt` (unveraendert 4 Befunde).

Neu: `tests/smoke_test_kontinentform.py` - er prueft ausdruecklich, dass
die Enden tun, was sie versprechen, statt es nur zu behaupten.

## Noch nicht gebaut

Der Kontinent-REITER (Kartengroesse, Formregler, Vorschau) fehlt. Der
Mechanismus steht und kostet 0.13 s je Neuberechnung.

---

# 2026-08-26 — Der Regionsreiter brach die Shell (behoben)

**Nutzerbefund am laufenden Programm:** *"geht nicht und oben ist der reiter
doppelt und jeder naechste reiter ist verschoben (also regionen(2) ist
terrain und terrain ist flussnetzwerk etc."*

## Was falsch war

Die Shell zerlegt jeden Reiter in DREI Teile:

    index = self.main_tab_bar.addTab(tab_name)          # (1)
    self.viewport_stack.addWidget(tab.viewport_widget)  # (2)
    self.parameter_stack.addWidget(tab.parameter_widget)
    self.statistics_stack.addWidget(tab.statistics_widget)

Mein Reiter war EIN Widget mit Splitter und hatte diese drei Attribute
nicht. Zeile (2) warf `AttributeError` - **nachdem (1) schon gelaufen war**.
In der Leiste blieb eine Beschriftung ohne Inhalt stehen, also war jeder
folgende Reiter um eins verschoben; und die aeussere Fehlerbehandlung legte
zusaetzlich einen Fehlerreiter gleichen Namens an, daher der doppelte
Eintrag.

Behoben: `parameter_widget` (Dropdown, Regler, Haekchen), `viewport_widget`
(2D/3D-Umschalter und Vorschau), `statistics_widget` (Kennzahlen).

## Warum kein Test das sah - und was daraus folgt

`smoke_test_region_tab.py` lief mit **zwoelf gruenen Pruefungen** durch. Es
prueft den Reiter FUER SICH; der Vertrag lebt aber in der Shell. Genau
dieselbe Fehlerklasse wie das 3D-Anzeigeregister am selben Tag: nicht eine
fehlende Methode, sondern eine fehlende Verbindung zwischen zwei Teilen,
die einzeln in Ordnung sind.

Neu: `tests/smoke_test_reiter_vertrag.py`. Er prueft drei Stufen, und die
dritte ist die entscheidende:

1. Die Shell greift wirklich diese drei Attribute ab (per `inspect` am
   Quelltext - sonst veraltet der Test still, wenn jemand die Shell umbaut).
2. Jede baubare Reiterklasse liefert sie, und zwar als QWidget.
3. **Die Shell wird tatsaechlich aufgebaut und ABGEZAEHLT:** Reiterleiste
   gegen Viewport-, Parameter- und Statistikstapel, keine doppelte
   Beschriftung, `tab_order` passend. Nur diese Stufe haette den Befund
   direkt gefunden - die ersten beiden fragen "hat der Reiter das
   Attribut", die dritte fragt "kommt am Ende gleich viel heraus".

Stand: Reiterleiste 11, Viewport 11, Parameter 11, Statistik 11,
`Regionen | Terrain | Flussnetzwerk | ...` ohne Doppelung.

---

# 2026-08-26 — Der Regionsreiter steht (Aufraeumplan 4.10)

`gui/tabs/region_tab.py`, registriert in `map_editor.py` als **erster
Reiter** - vor Terrain. Die Reihenfolge der Reiter IST der Arbeitsablauf des
Nutzers, und sie ordnet ihn zugleich nach den Kosten: der Regionsreiter
rechnet in 0.1-1.2 s, die Vollkarte in rund 40 s.

## Was er hat

* Dropdown ueber alle neun Regionen; die Regler springen auf deren
  Katalogwerte.
* Fuenf Gelaenderegler (`hoehe_m`, `relief_m`, `formgroesse_m`, `rauheit`,
  `potenz`) - **Ueberschreibung nur fuer die aktuelle Karte**, je Region
  gemerkt, mit Knopf zum Zuruecksetzen.
* Vier Erosionsfilterregler, nach GEMESSENER Wirkung ausgewaehlt.
  `DETAIL` fehlt bewusst (0.18 m Wirkung ueber den ganzen Reglerweg, im Bild
  nicht unterscheidbar).
* Haekchen fuer die Kuestentypen - beim Ziehen aus (unter 0.3 s), beim
  Loslassen an.
* **2D UND 3D**, in derselben Aenderung. Beide zeigen dasselbe Feld; die
  3D-Ansicht bekommt es ueber `MapDisplay3DWidget.update_heightmap()`.
* Ein Puffer von 90 ms sammelt schnelle Reglerbewegungen ein.

## Zwei Fehler beim Bauen, beide vom selben Typ wie fruehere

**1. Das Oeffnen des Reiters erzeugte sofort eine Ueberschreibung.** Hoehe
und Relief hatten Schrittweite 10 m, die Katalogwerte liegen aber nicht auf
dieser Stufe (Clonagh 165.3 m, Skerrheim 484.9 m). Die Regler rasteten
auf 170 bzw. 480 - und der Reiter meldete eine Aenderung, die niemand
gemacht hatte. **Exakt derselbe Fehler wie bei `MAP_DISTANCE_KM` am selben
Tag**, nur an einer Stelle, die `smoke_test_parameter_eindeutig.py` nicht
sieht: die Regler dieses Reiters kommen nicht aus einer Reglertabelle.
Behoben mit Schritt 0.1 plus einer Toleranz von einer halben Schrittweite.
`tests/smoke_test_region_tab.py` bewacht es jetzt.

**2. Die Farbgebung normierte auf das Regionsmaximum.** Dadurch war das
Clonagh (bis 211 m) zur Haelfte weiss "verschneit" und sah aus wie das
Nevadin (1580 m). Eine Vorschau, die dem VERGLEICH dient, darf zwei
Regionen mit siebenfach verschiedener Hoehe nicht gleich einfaerben. Jetzt
absolute Meterbaender: gruen bis 300 m, braun bis 1200 m, darueber Fels und
Schnee.

Ausserdem: das Vorschaubild stand als 256x160-Briefmarke in der Mitte der
Flaeche. Es wird jetzt eingepasst, mit `FastTransformation` - die Vorschau
soll ihre Pixel ZEIGEN, man stellt daran die Feinheit des Gelaendes ein.

## Testlage

| Test | Stand |
|---|---|
| `smoke_test_region_tab.py` | **neu, gruen** (12 Pruefungen) |
| `smoke_test_regionsfeld.py` | gruen |
| `smoke_test_anzeige_register_3d.py` | gruen |

Gemessen im Test: langsamste Region 0.29 s, Median 0.14 s - live.

## Sichtpruefung steht aus

Der Reiter wurde headless gebaut und abgegriffen. Wie er im laufenden
Programm aussieht, hat noch niemand gesehen.

---

# 2026-08-26 — Vektorkueste in der Regionsansicht

`regionsfeld(..., kueste=True)` legt jetzt die Vektorkueste mit den drei
Archetypen der Region auf. `region_map` ist dabei ein KONSTANTES Feld mit
dem Index dieser Region - dadurch waehlt `VektorKueste` genau ihre drei
Archetypen und mischt nichts von Nachbarn dazu.

**Quadratisch gerechnet, danach zugeschnitten.** `VektorKueste` leitet ihren
Massstab aus `shape[0]` ab und benutzt `self.size` an 15 Stellen - sie setzt
ein quadratisches Feld voraus. Eine Region IST quadratisch (7.1 x 7.1 km);
ein rechteckiger Ausschnitt wird deshalb quadratisch gerechnet und
zugeschnitten, statt 15 Stellen umzubauen. Der Fall wird laut protokolliert.

Ohne Wasserlinie (Nevadin) wird die Kuestenformung uebersprungen - mit
Logzeile, nicht still.

## Kosten (256x160 px, Seed 20260804)

| | ohne Kueste | mit Kueste |
|---|---:|---:|
| Morobora, Estrande | 0.10 s | 0.31 s |
| Clonagh | 0.29 s | 0.37 s |
| Macchia | 0.23 s | 0.83 s |
| Thalassia | 0.23 s | 1.19 s |

Damit ist die Kueste als ZUSCHALTER richtig aufgehoben: beim Ziehen eines
Reglers ohne (unter 0.3 s), beim Loslassen mit.

## OFFEN: zwei Regionen zeigen einen Archetyp gar nicht

Gemessen, Anteil an der Kuestenlaenge in der Vorschau:

| Region | Segmente | Anteile |
|---|---:|---|
| Clonagh | 3 | Moher **0 %**, West-Cork 75 %, Luce Bay 25 % |
| Morobora | 3 | Kola 33 %, Weissmeer **0 %**, Labrador 67 % |
| Macchia | 9 | 38 / 26 / 37 % |
| Thalassia | 35 | 29 / 24 / 47 % |

**Ursache: zu wenige unabhaengige Ziehungen.** Die Vorschauküste ist EINE
Linie quer durch die Karte (rund 8 km), und die Saatzuweisung ist bewusst
kohaerent (`SAAT_KOHAERENZ_STATIONEN = 3.5`), damit benachbarte Stationen
denselben Typ bekommen und lange Laeufe entstehen. Auf einer kurzen Kueste
bleiben davon nur drei Segmente uebrig - und dann kann einer der drei Typen
leer ausgehen.

Der Nutzer will aber ausdruecklich *"dazwischen cliffs of moher"* sehen. Fuer
eine VORSCHAU waere es richtig, die drei Typen zu gleichen Teilen laengs der
Kueste zu erzwingen - aber das weicht vom Verhalten der Karte ab und ist
deshalb eine Nutzerentscheidung, keine stille Aenderung.

---

# 2026-08-26 — Land-See-Aufbau der Regionsansicht (Aufraeumplan 4.11)

Nutzertabelle: Clonagh 3/4 Land und 1/4 Meer mit geschwungener
Kuestenlinie, Thalassia 1/2 zu 1/2 mit Klemmung der linken Haelfte,
Skerrheim ein Fjord in der Mitte, Nevadin gar keine Kueste.

## Umgesetzt als HOEHENVERSATZ, nicht als neues Relief

`REGIONS_AUFBAU` und `_aufbau_anwenden()` in `core/terrain_weltkarte.py`.
Der Aufbau verschiebt nur die HOEHENLAGE der beiden Seiten - Relief,
Formgroesse, Rauheit und Potenzkurve der Region bleiben unangetastet. Die
Vorschau zeigt also denselben Charakter wie vorher, nur mit einer Kueste
darin. Die Kuestenlinie ist eine aus drei Sinusgliedern zusammengesetzte
Kurve; eine gerade Trennung saehe nach Schnittkante aus.

**Es ist ein VORSCHAU-Aufbau, keine Karteneigenschaft.** Das Clonagh hat
`wasser_soll = 0`, bekommt hier aber ein Viertel Meer - weil man einen
Kuestentyp nur an einer Kueste beurteilen kann. Auf der Karte entscheidet
weiterhin die Kontinentform.

## Der Versatz zog zuerst Land HERUNTER

Der erste Entwurf setzte beide Versaetze unbedingt:

    versatz_land = LANDSOCKEL - percentile(H, 5)

Fuer eine Region, die laengst ueber Wasser liegt, ist das NEGATIV. Gemessen
landete das Clonagh (121..194 m) bei **-98..81 m**. Die Landanteile
stimmten dabei alle - der Fehler steckte allein in den Hoehen, und ein Test
nur auf den Landanteil haette ihn durchgelassen. Mit `max(0, ...)` bzw.
`min(0, ...)` wirkt jeder Versatz nur in eine Richtung; jede Region bleibt
auf ihrer eigenen Hoehe, und der Aufbau greift nur dort ein, wo er muss.

## Ergebnis (256x160 px, Seed 20260804)

| Region | Aufbau | Land | Hoehe |
|---|---|---:|---|
| Clonagh | kueste | 81 % | -98..197 m |
| Skerrheim | fjord | 69 % | -330..331 m |
| Morobora | kueste | 82 % | -123..336 m |
| Estrande | kueste | 79 % | -147..143 m |
| Nevadin | ohne_kueste | 100 % | 587..1580 m |
| Nebelrode | kueste | 82 % | -161..421 m |
| Samarcia | kueste | 82 % | -112..285 m |
| Macchia | kueste | 79 % | -275..288 m |
| Thalassia | inseln | 57 % | -261..350 m |

Alle neun in 1.34 s. `smoke_test_regionsfeld.py` prueft jetzt sechs Dinge,
darunter ausdruecklich **beides**: den Landanteil UND dass der Aufbau
vorhandenes Land nicht senkt.

---

# 2026-08-26 — Regionsansicht: der Rechenkern steht (Aufraeumplan 4.10)

## Was gebaut wurde

`regionsfeld(name, breite, hoehe, seed, ueberschreibung, km, erosion)` in
`core/terrain_weltkarte.py`. EINE Region auf Regionsmassstab (7.1 km), ohne
Kontinent, ohne Voronoi-Mischung, ohne Kueste, ohne Fluesse - nur
Oktavenstapel mit einem Parametersatz, Potenzkurve und Erosionsfilter.

**Gemessen: 0.18 s je Region bei 256x160 px, alle neun in 1.59 s.** Damit
ist die Live-Vorschau am Regler moeglich.

Zwei Vorarbeiten waren noetig, beide klein:

* `oktavenstapel()` kann rechteckig. Es baute (size, size) aus EINER Achse;
  `noise2array()` nimmt seit jeher zwei. Der GPU-Pfad kann nur quadratisch
  und meldet das jetzt LAUT, statt still auf CPU zu wechseln.
* `filter_heightmap()` kann rechteckig. Die geforderte Quadratform war keine
  Bedingung des Filters, sondern eine Annahme der Umsetzung. Beide Achsen
  werden mit DERSELBEN Laenge normiert - sonst waeren die Rinnen in der
  laengeren Richtung gestreckt.

## Die Oktavenformel stand zweimal im Code

Mein erster Entwurf schrieb das zweiseitige Tor in `regionsfeld()` ein
ZWEITES Mal hin. Das ist genau die Falle aus SPEZIFIKATION 4.5: aendert
jemand `weltfeld()` und vergisst die Vorschau, stellt der Nutzer seine
Regionen an einem Gelaende ein, das es auf der Karte nicht gibt - ohne
Fehlermeldung.

**An der Wurzel behoben statt getestet:** neue Funktion `oktavengewicht(k,
formgroesse, rauheit)`, die Skalare UND Felder nimmt. Beide Wege rufen
dieselbe Funktion; ein Test dagegen wuerde nur numpy pruefen.

## Der Test hatte dreimal unrecht, nicht der Code

`tests/smoke_test_regionsfeld.py` ist neu. Er hat sich dreimal selbst
korrigiert, und jedes Mal lag der Fehler bei ihm:

1. **Er verglich Hangwinkel bei verschiedenen Pixelgroessen** (Vorschau
   27.7 m/px gegen Pipeline 55.5 m/px) und meldete bis zu 15 Grad. Hang ist
   aufloesungsabhaengig - derselbe Effekt, den `regionen_welt` als
   "Landschaft haengt an der Pixelzahl" fuehrt.
2. **Er verlangte ueberhaupt Gleichheit mit der Pipeline.** Auch bei
   gleichem Massstab bleiben 13 Grad Unterschied im Nevadin - **und das
   ist der Zweck der Pipeline**: dort wird jede Region ueber die
   Voronoi-Gewichte mit ihren Nachbarn verschmolzen, ein Alpenlandpixel
   traegt anteilig auch Nebelrode (Relief 135 statt 1050). Gleichheit
   zu fordern hiesse, die Mischung fuer einen Fehler zu halten. Der Test
   rechnet jetzt die dokumentierte Formel unabhaengig nach.
3. **Er prueft den Mittelwert statt des Medians.** Die Potenzkurve dreht
   ausdruecklich um den MEDIAN. Und die Grenze musste relativ zum Relief
   werden: die Abweichung liegt bei 0.9 bis 7.2 % des Reliefs, und die
   Klemmung erklaert das NICHT (nur 0.6-5 % der Pixel liegen am Rand) - es
   ist Stichprobenrauschen, weil ein 128er-Feld nur wenige unabhaengige
   Formen enthaelt.

## Was noch fehlt, und es ist keine Kleinigkeit

**Bei den wasserreichen Regionen zeigt die Vorschau fast nur Meer.**
Estrande (`hoehe_m` = -81 m) und Skerrheim (-52 m) liegen als REINER
Parametersatz unter dem Meeresspiegel; auf der Karte hebt sie die
Kontinentmaske heraus.

Das ist genau der Punkt, den der Nutzer in seiner Aufbautabelle schon
beschrieben hat (AUFRAEUMPLAN 4.11): die Regionsansicht soll eine
Land-See-Aufteilung KONSTRUIEREN - Clonagh 3/4 Land und 1/4 Meer mit
einer geschwungenen Kuestenlinie dazwischen, Thalassia 1/2 zu 1/2
mit Klemmung der linken Haelfte, Skerrheim ein Fjord in der Mitte,
Nevadin gar keine Kueste. **Das ist der naechste Bauabschnitt**, nicht ein
Nachtrag.

## Testlage

| Test | Stand |
|---|---|
| `smoke_test_regionsfeld.py` | **neu, gruen** (5 Pruefungen) |
| `smoke_test_regionen_welt.py` | 4 Befunde (unveraendert) |
| `smoke_test_erosionsfilter_baender.py` | gruen |

---

# 2026-08-26 — Kuestenprofile auf den Weltmassstab gebracht (Aufraeumplan 4.12)

Nutzervorgabe: *"lass uns erstmal verkleinern, ausser bei den flachen typen
pro region. vielleicht passt es ja dann."*

## In BEIDEN Achsen, nicht nur in der Hoehe

Am 2026-08-24 hatte derselbe Nutzer verlangt, die Profile sollten *"das
gleiche hoehen zu tiefen verhaeltnis wie in echt"* behalten. Eine reine
Hoehenstauchung haette das verletzt. Der Faktor wirkt deshalb auf beide
Achsen:

    h'(x) = f * h(x / f)

Form und Neigung bleiben unveraendert, es entsteht ein Modell im Massstab.
Beide Nutzervorgaben sind damit zugleich erfuellt.

## Die flachen Typen bleiben unangetastet

Kriterium ist die BEREITS VORHANDENE Definition aus
`smoke_test_kuestenprofiltreue.py`: h(150 m) <= 25 m. **Keine zweite
Definition danebengestellt** - und sie passt auch sachlich, denn zu hoch
ragen die STEILEN Profile, die flachen waren nie das Problem.

| Region | Faktor | Beispiel |
|---|---:|---|
| Clonagh | 0.35 | Moher-Klippen 148 -> 52 m |
| Skerrheim | 0.43 | Fjordwand 559 -> 336 m |
| Samarcia | 0.59 | Costa-Brava 93 -> 78 m |
| Macchia | 0.62 | Cinque-Terre 252 -> 178 m |
| Nebelrode | 0.71 | Ruegen-Kreide 95 -> 74 m |
| Morobora, Estrande, Griech. Inseln, Nevadin | 1.00 | unveraendert |

Morobora: alle drei Archetypen sind nach dem Kriterium flach, also kein
Faktor - ihr Verhaeltnis lag ohnehin nur bei 1.52. Nevadin: braeuchte
rechnerisch 2.81, also eine STRECKUNG des 900-m-Profils auf 2529 m, weit
ueber p2. Der Faktor ist bei 1.0 gedeckelt; praktisch gegenstandslos, weil
das Nevadin auf 62 von 64 Karten keine Kueste hat.

## Der Test mass danach die falsche Vorlage

`smoke_test_kuestenprofiltreue.py` verglich weiter gegen
`MESS_PROFIL_M_JE_ARCHETYP`, die ROHE Messtabelle. Der Generator wendet aber
seit dieser Aenderung `GEMESSENES_PROFIL_M` an, die skalierte.

**Der Test bestand trotzdem** - der Median stieg nur von 1.8 auf 3.9 m. Er
haette also weiter gruen gemeldet, waehrend er etwas anderes prueft als das,
was gebaut wurde. Genau die Art stiller Fehlmessung, die dieses Projekt
mehrfach getroffen hat. Auf die angewandte Vorlage gezogen: **Median zurueck
auf 1.7 m** - das Gelaende trifft die skalierten Profile so genau wie vorher
die rohen.

## Testlage: die Eichung wird BESSER

| Test | vorher | nachher |
|---|---|---|
| `smoke_test_regionen_welt.py` | 5 Befunde | **4** (Samarcia zurueck) |
| `smoke_test_kuestenprofiltreue.py` | 3/3 | 3/3, Median 1.7 m |
| `smoke_test_kuestengebiete.py` | gruen | gruen |

Die Samarcia sass mit Hang 8.5 genau auf der Toleranzgrenze; der kleinere
Kuesteneinfluss bringt sie darunter. Macchia 19.4 -> 18.6.

---

# 2026-08-26 — Voronoi-/Hoehenfaktor-Ansicht, und ein bestaetigter Regelverstoss

## 1. Der Regelverstoss, vom Nutzer gemeldet

*"du hast im uebrigen gegen die regel verstossen eine 2D karte zu erstellen
ohne den shader fuer 3D mitzuschreiben. bei Flussnetzwerk gehen die 3D
karten nicht."*

**Er hat recht, und schlimmer als beschrieben.** Ich hatte am selben Tag
BEHAUPTET, `river_water` gehe "den gewoehnlichen Skalarweg und ist in
beiden Ansichten sichtbar". Das war GESCHLUSSFOLGERT, nicht gemessen.

Die Ursache ist eine neue Auspraegung der bekannten Fehlerklasse - kein
fehlender Methodenname, sondern ein fehlender REGISTEREINTRAG:

    mapped_layer = self._LAYER_NAME_MAP_3D.get(layer_type)

Steht ein Layer dort nicht, ist `mapped_layer` None: es wird nichts ans 3D
gepusht, UND die Sichtbarkeitsschleife danach schaltet ALLE Layer des
Reiters unsichtbar. Im 3D bleibt blankes Gelaende stehen.

**Betroffen war der ganze Flussreiter, nicht nur mein neuer Layer.** Er
meldet sich mit `generator_type = "terrain"` an, aber weder `river_water`
noch `river_order` standen in einer der beiden Listen - `river_order` seit
jeher nicht. `_LAYER_SELECTION_KEYS_3D` hatte ueberhaupt keinen
Fluss-Eintrag.

Neu: `tests/smoke_test_anzeige_register_3d.py`. Er prueft drei Ketten -
Modus -> Register -> Sichtbarkeitsliste -> 3D-Datenplatz. **Bei seinem
ERSTEN Lauf fand er einen weiteren, aelteren Fehler, den ich nicht
verursacht hatte:** `water.evaporation` stand in beiden Registern des
Wasserreiters, fehlte aber auf der 3D-Seite in `layer_visibility`,
`overlay_data` UND in der Renderschleife. Die Verdunstungskarte war im 3D
nie sichtbar. Behoben.

Der Test hatte auch selbst einen Fehler: er verglich MODUSNAMEN statt der
gepushten `layer_type`. Beim Terrain-Reiter heisst der Modus "slope", der
gepushte Layer aber "slopemap" - er meldete "terrain.slope fehlt", und da
hatte der Test unrecht, nicht der Code. Jetzt stehen die Paare
ausdruecklich in der Tabelle.

## 2. Linienstaerke nach Wassermenge

*"du solltest dort auch die dicke der linie langsam steigen lassen mit der
wassermenge. damit es deutlicher ist."*

`r = 0.30 + 0.60 * log10(1 + Menge)`, also 0.6 px beim Bach und 2.0 px beim
Strom. **Nur auf `river_water`, nicht auf `river_mask`** - letztere geht in
die Biomklassifikation (Uferbiome), ein breiterer Fluss haette dort
stillschweigend die Biomverteilung verschoben. Gemessen: `river_mask`
unveraendert 9932 px, `river_water` 13068 px.

## 3. Die Zweipunktmethode - und ein Fehler von gestern

*"hoehenwert vom kuestenprofil sei die mittlere hoehe von 400 bis 700 m
tiefe im hinterland (zweipunktmethode)."*

**Die Methode ist nicht nur besser, sie ersetzt etwas Falsches.** Das
gestern gebaute Gebietssystem rangierte die Hinterlandhoehe nach
`hoehe_faktor` - und der beschreibt das UFER. In VIER von neun Regionen ist
die Reihenfolge dadurch umgekehrt:

| Region | Archetyp | `hoehe_faktor` | h(400-700) |
|---|---|---:|---:|
| Skerrheim | Fjordbucht | 0.30 | **195 m** |
| Skerrheim | Schaerenkueste | 0.50 | 15 m |
| Thalassia | Kreta-Buchten | 0.80 | **202 m** |
| Thalassia | Santorini-Kliff | 1.15 | 120 m |
| Macchia | Cinque-Terre | 1.00 | **252 m** |
| Macchia | Amalfi-Steilkueste | 1.60 | 204 m |
| Morobora | Labrador-Buchten | 0.70 | **90 m** |
| Morobora | Kola-Steilkueste | 1.10 | 58 m |

Neu: `GEMESSENE_HINTERLANDHOEHE` in `core/vektor_kueste.py`, **abgeleitet**
aus `MESS_PROFIL_M_JE_ARCHETYP` statt von Hand gepflegt - eine zweite
Tabelle waere eine zweite Wahrheit ueber dieselbe Messung.

`smoke_test_kuestengebiete.py` meldete die Umstellung korrekt als
Fehlschlag (Rangkorrelation fiel auf +0.06) und prueft jetzt gegen die neue
Groesse: **+0.71 bei Grenze +0.50.**

## 4. Die Voronoi- und Hoehenfaktor-Ansicht

*"kannst du mir die voronoiansicht als erstes bauen? ich will den
hoehenfaktor sehen koennen (3d und 2D)."*

Zwei neue Terrain-Ausgaenge und zwei neue Radioknoepfe im Terrain-Reiter:

* **Hoehenfaktor** (`hinterland_height`): je Pixel die gemessene
  Hinterlandhoehe seines Gebiets in Metern. 4 bis 559 m, auf 85 % der
  Landflaeche gesetzt (der Rest ist der alpine Sonderfall).
* **Voronoi** (`voronoi_map`): die 180 Zellen, aus denen Regionen und
  Gebiete wachsen. Zyklische Farbtafel - der Zahlenwert bedeutet nichts.

**Von Anfang an in BEIDEN Ansichten**, in derselben Aenderung: Eintrag in
`_LAYER_NAME_MAP_3D`, in `_LAYER_SELECTION_KEYS_3D["terrain"]`, in
`layer_visibility`, in `overlay_data` und in der paintGL-Schleife. 2D
faellt von selbst auf `_render_generic_map()` zurueck, das dieselbe
`layer_ranges`-Tabelle liest.

## 5. Testlage

| Test | Stand |
|---|---|
| `smoke_test_anzeige_register_3d.py` | **neu, gruen** (12 Modi, 4 begruendete Ausnahmen) |
| `smoke_test_kuestengebiete.py` | gruen, Rangkorrelation +0.71 |
| `smoke_test_kuestenprofiltreue.py` | 3/3 |
| `smoke_test_regionen_welt.py` | 5 Befunde (unveraendert) |
| `smoke_test_display_methoden_existieren.py` | gruen |

## 6. Entschieden, noch nicht gebaut

* Regionsnamen: **Thalassia, Macchia, Skerrheim, Samarcia** stehen fest.
  Fuenf offen, Nutzer wuenscht mehr Komposita (siehe AUFRAEUMPLAN 4.13).
* Profilmaszstab: verkleinern in BEIDEN Achsen, **ausser bei den flachen
  Typen je Region** (Nutzer: *"dann haben wir zB auch kuerzere straende und
  das mag ich nicht so"*). Noch nicht umgesetzt.

---

# 2026-08-26 — Flussregler gemessen und gekuerzt (Aufraeumplan 4.3)

Nutzervorgabe: *"kannst du die 5 wichtigsten parameter fuer die fluesse
herausfinden und mir diese auf die Flussnetzwerktab seite packen?"*

## Die Messung

Jeder der elf Regler von Minimum zu Maximum, alles andere auf Vorgabe
(384 px, Seed 20260804). ZWEI Groessen, weil ein Flussregler auf zwei Arten
wirken kann - Hoehenaenderung (die Taeler) und Anteil der Flusspixel, die
den Ort wechseln (der Lauf):

| Regler | Hoehe Mittel | groesste | Netz wechselt |
|---|---:|---:|---:|
| SPACING_M | 21.58 m | 494 m | 97.4 % |
| VALLEY_WIDTH | 19.26 m | 467 m | 0.3 % |
| INHERIT_COST | 10.72 m | 690 m | 52.6 % |
| MOUTH_DEPTH_M | 10.26 m | 451 m | 52.2 % |
| COST_STRENGTH | 10.21 m | 557 m | 72.2 % |
| INCISION_SHARE | 3.22 m | 189 m | 0.0 % |
| **VALLEY_FORM** | **0.00 m** | 0 m | 0.0 % |
| MEANDER, DIVIDE_BLEND, PLATEAU_FLATTEN, BORDER_OUTFLOW | 0.00 m | 0 m | 0.0 % |

Die letzten vier stehen ohnehin in `stillgelegte_regler` - die Messung
bestaetigt das.

## Drei stille Fehler, alle gefunden statt vermutet

**1. `VALLEY_FORM` war freigeschaltet und tat nichts.** In
`taeler_eingraben()` ueberschrieb `form_feld = felder.get("talform")` den
uebergebenen Wert BEDINGUNGSLOS. Die Regionsdifferenz (Alpen-V, Skerrheim-U)
ist richtig und gewollt, machte den GUI-Regler aber stumm. Er SKALIERT das
Regionsfeld jetzt, statt es zu ersetzen: bei seiner Vorgabe exakt neutral
(gemessen 0.0000 m Unterschied), darueber alles runder, darunter kerbiger.
Wirkung danach 7.67 m im Mittel, 364 m maximal - vorher 0.00 m.

**2. `INCISION_SHARE` rastete seine eigene Vorgabe weg.** Vorgabe 0.18 bei
Schritt 0.05 - der Regler zeigte beim Aufbau 0.20. Der Wert 0.18 stammt aus
dieser Sitzung; das Raster steht jetzt auf 0.01.

**3. `MAP_DISTANCE_KM` ebenso, und das ist der ernste Fall.** Vorgabe 21.3
bei Schritt 1.0 - der Regler rastete auf 21 km. **Das ist die Weltbreite,
gegen die jede Meterrechnung des Programms geeicht ist.** Das blosse
Oeffnen des Terrain-Reiters haette sie um 1.4 % verstellt, ohne Meldung.
Schritt jetzt 0.1.

Gefunden wurde 2. und 3. nur, weil nach dem ersten Fund ALLE Regler
gegengeprueft wurden statt nur der eine. `smoke_test_parameter_eindeutig.py`
hat dafuer jetzt eine fuenfte Gruppe, mit einer begruendeten Ausnahme
(`EROSION.CONVERGENCE_THRESHOLD`, reine Fliesskomma-Ungenauigkeit).

## Die fuenf auf dem Flussreiter

    Talabstand (m)                 SPACING_M
    Talbreite                      VALLEY_WIDTH
    Taltiefe                       INCISION_SHARE
    Talform (V bis U)              VALLEY_FORM
    Fluesse folgen dem Tiefland    COST_STRENGTH

**Nicht streng nach der Messliste**, und das ist eine Entscheidung:
`INHERIT_COST` und `MOUTH_DEPTH_M` schneiden hoch ab, weil sie den LAUF
verschieben - das Netz sieht danach anders aus, die Landschaft nicht. Sie
sind einmal einzustellen und bleiben im Terrain-Reiter. `INCISION_SHARE`
steht dagegen hier, obwohl sein Mittelwert klein ist: 3.22 m im Mittel bei
189 m Maximum heisst, die Wirkung ist auf die Taeler KONZENTRIERT. **Ein
Mittelwert allein waere fuer diese Art Regler blind** - darum steht die
Maximalspalte mit in der Tabelle.

Die fuenf stehen NUR im Flussreiter, nicht zusaetzlich im Terrain-Reiter.

---

# 2026-08-26 — Eine Wasserkarte, und der Erosionsfilter am Foto gemessen

## 1. Eine Flusskarte statt rot/gruen (Aufraeumplan 4.4)

Nutzervorgabe: *"koennen wir nur eine karte haben die darstellt wie viel
wasser fuer die fluesse berechnet wurde?"*

Neuer Terrain-Output `river_water`: `netz["flaeche"]` (Niederschlag mal
Flaeche, flussabwaerts akkumuliert) aufs Raster gelegt, mit Logskala. Neue
Leitansicht "Wassermenge" im Flussreiter.

**NICHT `flow_map` genommen**, obwohl sie naheliegt: die gehoert zur
WASSER-Stufe (`get_water_data`), der Flussreiter liest Terrain-Daten. Sie
dort zu zeigen hiesse, den Reiter von einer spaeteren Pipelinestufe
abhaengig zu machen.

Die Farbskala war im ersten Anlauf GERATEN (0.02 bis 60) und um mehr als
eine Groessenordnung daneben. Gemessen sind es 0.38 bis 725 (Median 1.33,
p90 17.5, p99 378) - aufgefallen nur, weil die Messung nachgereicht wurde.

**"Baeche (Mikro) gibt es ja auch gar nicht" - die Daten sagen etwas
anderes.** Gemessen liegen 3760 Mikro-Pixel im Raster, MEHR als Makro
(2411). Es fehlte die ANZEIGE: `update_display_mode()` ruft
`overlay_river_generations` per `hasattr` auf, und die gibt es nur auf
MapDisplay2D - in der 3D-Ansicht trifft die Weiche nie zu und es passiert
lautlos gar nichts. Genau die Fehlerklasse aus CLAUDE.md, zum vierten Mal.
Dort steht jetzt eine laute Warnung; die neue Leitansicht "Wassermenge"
geht den gewoehnlichen Skalarweg und ist in BEIDEN Ansichten sichtbar.

## 2. Erosionsfilter gegen das Nutzerfoto (Vorarbeit zu 4.5)

Schalterlage wie verlangt: Flussnetz aus, Kuestentypen aus, Erosionsfilter
an. Gerendert wurde das Nevadin, 8.7 km Ausschnitt bei 21 m/px.

**Vorgeschlagene Werte:**

    erosion_filter_strength        0.60   (Hoechstwert)
    erosion_filter_gully_size_m    1200
    erosion_filter_gully_weight    0.90
    erosion_filter_ridge_rounding  0.00
    erosion_filter_crease_rounding 0.00
    erosion_filter_octaves         7
    erosion_filter_detail          beliebig - wirkungslos, siehe unten

Hang Median 17 -> 26 Grad, p95 43 -> 55 Grad.

**WIRKUNG JE REGLER, ueber den vollen Reglerweg gemessen** (384 px, mittlere
Hoehenaenderung zwischen Minimum und Maximum):

| Regler | Mittel | groesste |
|---|---:|---:|
| GULLY_SIZE_M | 47.04 m | 1149 m |
| GULLY_WEIGHT | 4.27 m | 89.8 m |
| STRENGTH | 3.44 m | 89.2 m |
| OCTAVES | 1.17 m | 26.3 m |
| RIDGE_ROUNDING | 0.45 m | 22.3 m |
| CREASE_ROUNDING | 0.42 m | 22.4 m |
| **DETAIL ("Gully Reach")** | **0.18 m** | 22.6 m |

**Das ist die Messung, die Punkt 4.5 braucht.** `DETAIL` ist 260-mal
schwaecher als `GULLY_SIZE_M` und im Bildvergleich ueber seinen ganzen
Bereich nicht unterscheidbar - der erste Streichkandidat. Die beiden
Rundungen liegen bei 0.45 bzw. 0.42 m und sind zusammenfassbar.

## 3. Was die Regler NICHT leisten koennen

Das Foto zeigt ein FALTENGEBIRGE: lange, parallele Kammketten mit quer
dazu verlaufenden Rinnen. Unser Nevadin entsteht aus ISOTROPEM Rauschen -
rundliche Massive ohne Vorzugsrichtung. Der Erosionsfilter zerfurcht die
Flanken, aber **er kann aus runden Klumpen keine parallelen Ketten
machen**; das Ergebnis liest sich grossflaechig als zerknittertes Papier.

Dafuer braeuchte es Anisotropie im GRUNDGELAENDE (gerichtetes/ridged
Rauschen mit einer regionalen Streichrichtung), nicht einen weiteren
Erosionsregler. Nicht gebaut, nicht beschlossen - als Befund vermerkt.

---

# 2026-08-26 — Kuestengebiete im Hinterland (Aufraeumplan 4.8)

Nutzerentwurf, vom Nutzer vorgezogen: *"gehe die folge durch, aber ich
moechte das du mit dem zuletzt beschriebenen anfaengst."* Zuschnitt:
*"lass uns anfangen nur mit hoehenwerten, also wie hoch die mittlere hoehe
ist. die anderen werte wird von der region vererbt."*

## Was gebaut wurde

`kuestengebiete()` in `core/terrain_weltkarte.py`. Entlang der Kueste tragen
die Voronoi-Zellen den Archetyp, der sie beruehrt; von dort waechst eine
Breitensuche ueber den Zellnachbarschaftsgraphen ins Land, bis jede Region
drei Gebiete hat. Die Gebietshoehe folgt dem `hoehe_faktor` des Archetyps.

Dafuer gibt `voronoi_regionen()` jetzt auch seine Zell-Etiketten zurueck -
bis dahin wurden sie weggeworfen.

**MULTIPLIKATIV, NICHT ADDITIV.** Der erste Entwurf addierte ein Delta. Das
war falsch und die Messung zeigte es sofort: bis zu +-70 m, davon 68 m
NOCH AN DER WASSERLINIE. Ein negatives Delta drueckt Land unter Null, die
Kueste wandert, und die Regionsmittel stimmten anschliessend auch nicht
mehr (Skerrheim +1.06 m), weil sie gegen die alte Landmaske gerechnet
waren. Multiplikativ ist H an der Wasserlinie 0, und 0 mal irgendetwas
bleibt 0 - die Kuestenlinie ist damit EXAKT erhalten.

**Mittelwerttreu je Region**, sonst waere die Eichung hin. Beides zugleich
(Mittel 0 UND an der Wasserlinie 0) geht nur, wenn ein Vielfaches DES
ANLAUFS abgezogen wird, nicht eine Konstante.

## Drei Fehler in der Flaechenquote, alle mit plausiblem Ergebnis

Die Quote nutzt dieselbe Log-Regelung wie die Flaecheneichung in
`voronoi_regionen()`. Sie hatte drei Fehler, und **keiner davon stuerzte ab
oder sah falsch aus** - es kam jedes Mal eine plausible Karte heraus:

| Fehler | Symptom |
|---|---|
| saatloser Archetyp bekam Abstand 1e3 | Log-Regelung schafft hoechstens ~21, er blieb dauerhaft ausgeschlossen |
| Ersatzabstand ueberall GLEICH | der Typ gewinnt alles oder nichts, nie dazwischen (Moher 0 %) |
| letzter statt bester Regeldurchgang | Graphdistanz ist ganzzahlig, die Regelung schwingt statt zu konvergieren |

Gemessen: **Samarcia 0 % / 100 % / 0 % bei DREI vorhandenen Saaten.**
Gefunden nur, weil eine Logzeile Soll gegen Ist je Region meldet - die
steht jetzt dauerhaft drin.

Nach den drei Korrekturen (384 px, Seed 20260804):

| Region | soll | ist |
|---|---|---|
| Clonagh | 25/40/35 | 21/42/38 |
| Skerrheim | 30/40/30 | 28/38/34 |
| Morobora | 30/40/30 | 33/38/29 |
| Estrande | 25/45/30 | 25/46/29 |
| Samarcia | 30/35/35 | 37/38/25 |
| Macchia | 30/35/35 | 40/28/33 |
| Thalassia | 25/40/35 | 34/35/31 |
| Nebelrode | 25/50/25 | 36/50/14 |

## Wirkung und Preis

Mittlere Hoehe je Gebiet, Beispiele: Nebelrode 251/159/79 m, Skerrheim
296/166/163 m, Samarcia 148/120/75 m. Die Rangfolge stimmt in **allen neun**
Regionen (Korrelation Gebietsdelta gegen `hoehe_faktor` +0.81 bis +1.00).

Das Nevadin bleibt zu 91 % beim Sondertyp "alpin" - Nutzervorgabe
*"meistens ist es aber einfach nur alpin"*.

| Test | vorher | nachher |
|---|---|---|
| `smoke_test_kuestenprofiltreue.py` | 3/3 | **3/3** |
| `smoke_test_regionen_welt.py` | 4 Befunde | 5 (Samarcia-Hang 8.9 statt 8.5) |
| `smoke_test_stufen_schalter.py` | gruen | gruen |
| `smoke_test_kuestengebiete.py` | - | **neu, gruen ueber 3 Karten** |

Der eine zusaetzliche Befund ist die Samarcia, und die sitzt genau auf der
Toleranzgrenze: 8.5 besteht, 8.6 faellt durch. Ein Versuch, sie ueber eine
breitere Ueberblendung (1800 m statt 800 m) zurueckzuholen, gelang zwar -
aber um den Preis des Effekts: die Gebietsspanne fiel von +-66 m auf
+-26 m, p10..p90 auf +-3 bis 9 m. **Nicht uebernommen** - eine Eichung, die
man durch Abschalten der Wirkung erkauft, ist keine.

## Vorarbeit am selben Tag

p1 nochmal 15 % naeher (210/245 -> 178/208 m), kostet auf beiden Tests
nichts. Die variable p2-Reichweite aus der Kuestenaehnlichkeit ist gebaut
und gemessen, steht aber auf AUS (`P2_AUS_AEHNLICHKEIT = False`): sie
kostete die Zusicherung *"flache kueste bleibt flach"*. Volle Messreihe bei
`P2_MAX_M` in `core/vektor_kueste.py`. **Mit dem Gebietssystem ist die
Reichweitenfrage neu zu stellen** - jetzt traegt das Gebiet die Kopplung
Kueste -> Hinterland, nicht mehr p2.

## Sichtpruefung steht aus

Alle Zahlen stimmen; ob die Naht im 3D-Bild wirklich verschwindet, hat
noch niemand gesehen.

---

# 2026-08-25 — Abschalthaekchen und die Alpenspitzen

Auftrag in zwei Teilen. Der Plan fuer den Rest steht in
`docs/AUFRAEUMPLAN.md`.

## 1. Drei Abschalthaekchen (Schritt 1 des Aufraeumplans)

*"kannst du mir einmal fuer flussnetzwerk und erosionfilter und
kuestentypen jeweils checkboxen einfuegen, mit denen ich die effekte immer
auch ausschalten kann?"*

Neu im Terrain-Reiter, je ein Haekchen oben in seiner Gruppe:
`river_network_aktiv`, `erosion_filter_aktiv`, `kuesten_archetypen_aktiv`.
Sie laufen als gewoehnliche Parameter (1.0/0.0) ueber denselben Weg wie
jeder Slider und greifen in die **Erzeugung** ein - 2D und 3D zeigen also
zwangslaeufig dasselbe, ohne zweiten Weg (stehende Regel in CLAUDE.md).

`weltfeld()` hat dafuer `kuesten_aktiv=True` bekommen. Ist es False,
laeuft weder der Vektorweg noch `_kuesten_umformen()`, und es gibt keine
Archetypfelder. Das vertraegt die Folgestufe: `_seetiefe_aus_archetyp()`
liest ueber `felder.get()` und steigt ohne sie sofort aus - genau wie beim
schon vorhandenen Rasterweg, der `vektor_kueste` auch nicht setzt.

**Gemessen, was jede Stufe beitraegt** (256 px, mittlere Hoehenaenderung):

| Stufe | Mittel | max |
|---|---:|---:|
| Kuestentypen | 97.7 m | 582 m |
| Flussnetz und Taeler | 17.0 m | 603 m |
| Erosionsfilter | **2.5 m** | 100 m |

**Der Erosionsfilter traegt deutlich weniger bei als erwartet.** Das ist
ein Befund fuer Schritt 5 des Aufraeumplans (sieben Regler fuer 2.5 m).

Neuer Waechter `tests/smoke_test_stufen_schalter.py`: er prueft nicht, ob
die Checkbox da ist, sondern ob sich das **Gelaende** aendert. Ein Haekchen
ohne Wirkung waere sonst von Erfolg nicht zu unterscheiden.

## 2. Die Alpenspitzen - Ursache gefunden und behoben

*"die alpen brauchen hier viiiiiel weniger spitzen. also sowas wie 5 berge
auf der regionsflaeche. hier sind hunderte zu sehen. amplitude ist ok."*

**Ursache: das Oktaventor war einseitig.** Es daempfte Wellen GROESSER als
die Formgroesse, lief nach unten aber offen bis 47 m, gebremst nur durch
`rauheit^k`. Gemessener Anteil des Reliefs in Wellenlaengen <= 375 m:

| Region | form | relief | Anteil | in Metern | grosse Formen/Region |
|---|---:|---:|---:|---:|---:|
| **Nevadin** | 3800 | **1050 m** | 23.9 % | **250 m** | 3.5 |
| Thalassia | 1100 | 404 m | 30.2 % | 122 m | 41.7 |
| Morobora | 3000 | 118 m | 5.5 % | 6 m | 5.6 |

Die GROSSEN Formen stimmten also bereits (3.5 Massive gegen den Wunsch von
5). Der Fehler waren 250 m Amplitude in Formen unter 375 m Breite.

**Behoben durch ein zweiseitiges Tor** (`FEINHEIT_TEILER = 8.0`): Wellen
unter `formgroesse_m / 8` werden zusaetzlich gedaempft. Das haengt
absichtlich an der Formgroesse statt an einer festen Meterzahl - begrenzt
wird das VERHAELTNIS von feinster zu groebster Form.

**Ergebnis (512 px, Seed 20260804):**

| | vorher | nachher |
|---|---:|---:|
| Gipfel im Nevadin | **122** | **24** |
| Hoehenstreuung Nevadin | 168 m | **173 m** |
| Relief unter 375 m | 250 m | **67 m** |

Die Streuung steigt leicht - **die Amplitude bleibt erhalten**, nur die
Nadeln verschwinden. Nebenbei verlieren auch Clonagh (55->23),
Nebelrode (59->25) und Macchia (48->24) ihre Zacken.

**`smoke_test_regionen_welt.py` wird dadurch deutlich besser:**

| | vorher | nachher |
|---|---:|---:|
| Befunde | 8 | **4** |
| Nevadin Hang (soll 27.0) | 36.5 | **30.5 ok** |
| Nebelrode, Samarcia, Morobora | alle DANEBEN | **alle ok** |
| Naht | 1.728 | **1.547** |
| Aufloesungskopplung r | +0.9611 | **+0.9402** |

Dass auch die Aufloesungskopplung faellt, bestaetigt die Diagnose: Oktaven
nahe der Pixelgroesse sind genau das, was eine Landschaft von der Pixelzahl
abhaengig macht.

## 3. NICHT geschafft: mehr Flaeche auf die Talsohle

*"dann brauchen wir height redistribution, so dass ein bisschen mehr auf die
talsohle faellt."*

**Der vorhandene Regler kann das nicht, gemessen ueber seinen ganzen
Bereich.** `potenz` im Nevadin von 0.6 bis 2.5:

| potenz | Median | unter 40 % | Hang |
|---:|---:|---:|---:|
| 0.6 | 457 m | 30.0 % | 19.0 |
| 1.5 (jetzt) | 467 m | 32.3 % | 17.2 |
| 2.5 | 501 m | 20.3 % | - |

Der Talsohlenanteil bewegt sich nur zwischen 30 und 33 %, und nach unten
wird der Hang sogar STEILER. Grund ist die Median-Rueckverschiebung
`t^p - 0.5^p + 0.5` in `weltfeld()`: der Regler ist konstruktionsbedingt
medianerhaltend und kann Flaeche nicht nach unten schieben.

**Was es braeuchte:** eine echte hypsometrische Kurve, die die
Flaechenverteilung ueber der Hoehe formt - ein neuer Mechanismus, kein
Parameterwert. Bewusst NICHT hingebogen; `potenz` steht unveraendert
bei 1.5.

---

# 2026-08-25 — Archetypverteilung ueber viele Karten neu gewichtet

**Nutzervorgabe:** *"es sollte gleichmaessig sein ueber viele maps hinweg.
eine map kann sich von einer anderen unterscheiden. also neu gewichten."*

## Der Befund: fast alles war schon richtig, gemessen wurde nur falsch

Bis hierher war JEDE Aussage ueber die Archetypverteilung an EINEM Seed
gemessen - auch die Fehlschlaege, die ich weiter unten in dieser Sitzung
noch als echte Befunde behandelt hatte. Die Streuung von Karte zu Karte
betraegt 8 bis 21 Prozentpunkte. Ueber **64 Karten** (384 px):

| | vorher | nachher |
|---|---:|---:|
| mittlere Abweichung vom `max_anteil` | 2.9 P | **2.7 P** |
| Archetypen innerhalb von 2 Standardfehlern | 22 von 24 | 23 von 24 |

Zwei Korrekturen an meinen eigenen frueheren Aussagen dieser Sitzung:

* **Kola-Steilkueste "5.53x ueber Soll"** war Rauschen. Ueber 64 Karten:
  29.7 % gegen 30.0 % Soll.
* **Schaerenkueste "0.17x"** ebenso: 38.2 % gegen 40.0 % Soll, und sie
  verschwindet auf keiner Kartenmehrheit.

Nur zwei Archetypen waren belegbar daneben (ueber 2 Standardfehler):
Vendee-Straende -5.1 P und Fjordwand +4.9 P.

## Die Gewichtung: ein Faktor wirkt, einer war wirkungslos

Neu ist `SAAT_BUDGET_KORREKTUR` in `core/vektor_kueste.py` - eine gemessene
Korrektur auf das Meterbudget, innerhalb der Region wieder normiert.

* **Fjordwand 0.86** wirkt: 34.9 % -> 32.3 % (Soll 30 %), damit im Rauschen.
* **Vendee-Straende 1.13 wurde gebaut, nachgemessen und wieder entfernt.**
  Wirkung: 39.9 % -> 39.8 %. Nichts. Der Grund steht in der
  Zuordnungsschleife: die uebrig gebliebenen Stationen gehen an
  `rest = max(archetypen, key=max_anteil)`, und das IST Vendee-Straende
  (45 % gegen 30 % und 25 %). **Wer ohnehin alle Reste einsammelt, ist
  nicht budgetbegrenzt - sein Budget zu erhoehen kann per Konstruktion
  nichts aendern.** Fuer die Nachbarn war der Faktor sogar leicht
  schaedlich (Ile-de-Re +2.5 -> +3.5 P).

Absichtlich NICHT korrigiert wurden die uebrigen 22 Archetypen: ihre
Abweichungen liegen innerhalb des Standardfehlers, und eine Korrektur
darauf kalibriert Rauschen ein.

## Der Test urteilt jetzt ueber 16 Karten statt ueber eine

`smoke_test_archetyp_verteilung.py` hatte an einem Seed geurteilt und
**schlug dadurch auf Rauschen an**. Neue Gruppe `verteilung_ueber_karten`:
16 Karten a 384 px, geprueft wird der Mittelwert. Die Einzelkartenzahlen
bleiben als Anschauung stehen, sind aber keine Pruefung mehr.

Grenzen an fuenf UNABHAENGIGEN 16er-Gruppen gemessen (Mittel 2.4-5.0 P,
schlechtester Einzelwert 8.6-12.2 P, nie ein Ausfall) und mit Reserve
gesetzt: 7 P im Mittel, 16 P einzeln. Scharf genug bleibt es fuer den
Fehler, um den es geht - ein systematisch ausfallender Archetyp liegt
25-50 Punkte daneben.

Archetypen unter 8 Karten mit Kueste werden nicht bewertet: das Nevadin
hat auf 62 von 64 Karten gar keine Kueste, sein Zweistichproben-"Mittel"
sah 60 Punkte daneben aus und war Artefakt.

**Stand: 4/4 Gruppen gruen**, mittlere Abweichung 3.3 P ueber 24
Archetypen.

## Offen geblieben

Vendee-Straende bleibt bei -5.1 P. Die Ursache ist nicht die Quote, sondern
`_segmente_schliessen()` (Einschmelzen unter `MIN_SEGMENT_M`); steht in
`docs/OFFENE_PUNKTE.md`.

---

# 2026-08-25 — Vorbildkueste Dingle ersetzt, Kuestenprofiltreue gruen

**Auftrag:** *"Also erstmal such dir was anderes als Dingle, gibt es etwas
flaches mit Huegel im Hinterland in schottland mit ausreichend Kueste? Oder
nimm England. irgendwo bei lake district oder so. Wir wollen ueberall genug
Profile haben."*

## 1. Neue Vorbildkueste: Sandhead / Luce Bay statt Inch Beach / Dingle

Vier Kandidaten gemessen (COP30 aus `tools/_dem_cache/`, Median ueber alle
Normalenschnitte, relativ zur Uferhoehe):

| Ausschnitt | Kueste | Schnitte | h(150 m) | h(350 m) | h(900 m) | Konturen |
|---|---:|---:|---:|---:|---:|---:|
| **Sandhead / Luce Bay, Galloway** | 8.8 km | 86 | **6.0 m** | **10.4 m** | **23.0 m** | **1** |
| Silecroft / Black Combe, Cumbria | 7.9 km | 79 | 22.3 m | 26.6 m | 19.2 m | 1 |
| Drigg / Seascale, Cumbria | 9.6 km | 97 | 11.4 m | 13.7 m | 10.4 m | 4 |
| Allonby / Solway, Cumbria | 7.7 km | 69 | 4.2 m | 6.5 m | 8.8 m | 1 |
| Inch Beach, Dingle (bisher) | 8.4 km | 96 | 2.6 m | 6.5 m | 51.9 m | 2 |

Sandhead ist der einzige mit BEIDEM - flach an der Wasserlinie und
steigendes Hinterland. Silecroft ist am Ufer gar nicht flach und FAELLT bei
900 m wieder (Tal im Bild). Drigg hat vier Konturen (Aestuar) und kein
Hinterland. Allonby ist sauber, aber voellig flach.

**Der Archetyp heisst jetzt `Luce-Bay-Straende`** (vorher
`Dingle-Straende`), in vier Codestellen: dem Katalog in
`core/terrain_weltkarte.py` und den drei Tabellen in
`core/vektor_kueste.py`.

Neu gemessen wurde nur `MESS_PROFIL_M_JE_ARCHETYP` - `MESS_ARCHETYP_MASSE`
und `MESS_FORM_JE_ARCHETYP` stammen aus dem Regions-Streckensplit, nicht aus
der Archetypkachel, und aendern sich beim Kachelwechsel nicht. Die anderen
26 Profile kamen bitgleich wieder heraus (gegengeprueft), es hat sich also
wirklich nur die eine Zeile geaendert.

## 2. Der eigentliche Fund: der Test mass in der Ueberblendzone

`smoke_test_kuestenprofiltreue.py` blieb nach dem Kachelwechsel rot
("flache Kuesten bleiben flach - Luce-Bay-Straende 54 m"). Die erste
Hypothese - die flachen Archetypen laegen auf kleinen Inseln, wo 150 m nur
2.7 px sind - **war falsch**. Gemessen ueber das Archetypraster, Festland
gegen Insel getrennt (384 px, Seed 20260804):

| Archetyp | soll/ist 150 m | soll/ist 350 m |
|---|---|---|
| Toskana-Straende | 4 / **4** | 6 / **29** |
| Weissmeer-Flachkueste | 5 / **5** | 7 / **31** |
| Luce-Bay-Straende | 6 / **10** | 10 / **30** |
| Foerdenkueste | 2 / **2** | 3 / **17** |
| Ostsee-Flachkueste | 10 / **11** | 17 / **52** |

Bei 150 m trifft JEDER flache Archetyp seine Vorlage, bei 350 m ist JEDER um
das Drei- bis Fuenffache zu hoch - und die Inseln kommen mit 0 px praktisch
gar nicht vor.

**Ursache:** der Test integrierte seinen Fehler ueber `PROFIL_VOLL_M = 350`.
Das ist in der Produktion aber nur noch der RUECKFALLWERT fuer Segmente ohne
gemessenes Profil; jedes echte Segment benutzt `voll_m = _zone_p1(name)` =
210 m (flach) bis 245 m (steil). Seit dem p1-Zug von heute frueh
(Nutzervorgabe *"30% naeher ran"*) lagen damit 140 m Ueberblendzone im
Messfenster - bei 350 m schlaegt das Hinterland zu 41 % durch
((350-210)/(550-210)). **Der Test bestrafte genau die Wirkung, die vorher
angefordert worden war.**

Behoben: der Test integriert jetzt bis zum p1 des jeweiligen Archetyps.

## 3. Quote nach Kuestenlaenge statt nach Stationszahl

Gegengeprueft, weil `max_anteil` im Katalog als "Obergrenze am gesamten
Kuestenumfang" beschrieben ist, aber gegen die Stationszahl gerechnet wurde:

    55.5 % aller Saatstationen liegen auf 14.5 % der Kueste (Faktor 3.8)

Grund ist `MIN_STATIONEN_JE_KONTUR = 16`: jede noch so kleine Insel bekommt
16 Stationen, das Festland eine je 420 m. Jede Station bringt jetzt ihr
Stueck Kueste als Gewicht mit.

**Diese Aenderung wurde gebaut, gemessen, VERWORFEN und wieder eingesetzt** -
weil sich erst am Ende zeigte, was sie wirklich bringt:

| | Mittel | schlechteste | Luce-Bay | regionen_welt |
|---|---:|---:|---:|---:|
| Stationszahl (alt) | 12.5 P | 65.0 P | 32 m rot | 7 Befunde |
| Meterbudget, ueberziehend | 12.8 P | **40.1 P** | **1 m gruen** | **8 Befunde** |
| Meterbudget, nicht ueberziehend | 14.0 P | 65.0 P | 32 m rot | 8 Befunde |

Und gegen `smoke_test_archetyp_verteilung.py`, das die Anteile direkt prueft:

| | Befunde |
|---|---|
| Meterbudget, ueberziehend | **1** (Schaerenkueste 0.17x, Kola-Steilkueste 5.53x) |
| Meterbudget, nicht ueberziehend | 2 (zusaetzlich faellt Fjordwand bei 17 Stationen ganz aus) |

Die nicht ueberziehende Variante sieht sauberer aus und ist in jedem Punkt
schwaecher: wer eine Station ueberspringt, weil sie nicht mehr ins Budget
passt, gibt sie an den NAECHSTEN Archetyp weiter - und die Schleife laeuft
vom steilsten zum flachsten. Die flachen Typen erben genau die Stationen,
die keiner wollte.

**Der Handel, offen benannt:** die Laengenquote behebt einen seit Beginn der
Messreihe roten Archetyp (44 m Hoehe nach 150 m, wo 6 m stehen sollen - ein
Huegel statt eines Strands) und halbiert die schlechteste Quotenabweichung.
Sie kostet dafuer EINEN Befund in `smoke_test_regionen_welt`: Morobora-Hang 10.4
statt 7.5. Die Morobora hat nur ~33 Stationen in der ganzen Region, da schlaegt
jede Umverteilung durch. Naht (1.728) und Aufloesungskopplung (r = +0.96)
bleiben unveraendert. **Wer den Morobora-Befund hoeher gewichtet als das
Strandprofil, dreht die Auswahlschleife in `_saat_setzen()` zurueck** - der
Kommentar dort nennt beide Fassungen.

## 4. Was NICHT behoben ist

`_segmente_schliessen()` schmilzt Segmente unter `MIN_SEGMENT_M` (750 m) in
den laengeren Nachbarn ein. Ein Archetyp mit vielen einzeln verstreuten
Stationen verliert dadurch seine ganze Laenge - **Schaerenkueste und
Algarve-Klippen kommen in BEIDEN Zaehlweisen auf 0.0 % Kuestenlaenge, obwohl
ihnen 16 bzw. 8 Stationen zugeteilt wurden.** Das ist der Rest des
"8 von 27 Archetypen fehlen"-Befunds vom 24.08. und der eigentliche Engpass;
die Quote ist es nicht.

Ebenfalls offen: `Fjordwand` liegt mit 168 m Abweichung weit daneben (24 m
statt 128 m nach 150 m). Das ist der bekannte Hoehenausreisser - eine
466-m-Fjordwand braucht eine Landmasse, die sie traegt, und die gibt es in
einer 21-km-Welt mit schmalen Fjorden selten.

## 5. Testlage

| Test | vorher | nachher |
|---|---|---|
| `smoke_test_kuestenprofiltreue.py` | 2/3, Median 6.1 m | **3/3, Median 2.3 m**, alle 11 flachen gruen |
| `smoke_test_regionen_welt.py` | 7 Befunde | 8 Befunde (Morobora neu) |
| `smoke_test_archetyp_verteilung.py` | 3 rote Pruefungen | **1** (Kola 5.53x neu, Toskana/Luce-Bay/Segmentlaenge behoben) |
| `smoke_test_vektor_kueste.py` | gruen | gruen |
| `smoke_test_kuesten_schnitt.py` | gruen | gruen |
| `smoke_test_kuesten_mesh.py` | gruen | gruen |

**Sichtpruefung steht aus** - der Nutzer konnte nicht schauen.

---

# 2026-08-25 — B.1 Flussnetz-Overlay

**Ausgangslage:** HEAD weiterhin `950a949` (12.08.). Der Nutzer meldete
"habs committed", **der Commit ist aber nicht durchgelaufen** - `git add`
hatte gegriffen (57 Dateien im Index, 18449 Zeilen), `git commit` nicht.
32 weitere Dateien waren gar nicht vorgemerkt, darunter die geaenderten
Kerndateien und zwei Loeschungen. Auf Wunsch des Nutzers vorerst
zurueckgestellt - **bleibt der wichtigste offene Punkt.**

## B.1 erledigt, und dabei zwei stille Fehler gefunden

Der Fluss-Reiter hatte das Overlay laengst. Die eigentlichen Luecken:

1. **`river_order` ohne Farbskala** - Auto-Skalierung, und weil 0 die Karte
   dominiert, sass die Skala im Nichts. Jetzt `("Blues", 0.0, 4.0)`,
   Obergrenze gemessen (Strahler max 4, 75.6 % der Knoten Ordnung 1).
2. **Der Biome-Reiter zeigte ein anderes Flusssystem, und nur in 2D.**
   `overlay_river_network(flow_map)` schnitt sich per 90.-Perzentil eine
   eigene Flussmaske aus dem Wassergenerator, waehrend der Fluss-Reiter
   `river_generation` aus dem Weltflussnetz zeigt - zwei Quellen fuer
   dieselbe Aussage. Und die Methode gibt es nur auf `MapDisplay2D`, im 3D
   fiel die `hasattr`-Weiche lautlos aus. **Dieselbe Fehlerklasse wie am
   24.08. bei `overlay_river_generations`, nur eine Stelle weiter.**

## Die Testluecke dahinter geschlossen

`smoke_test_display_methoden_existieren.py` verlangte, dass ein
Methodenname auf **mindestens einer** Anzeigeklasse existiert. Damit findet
er nur Namen, die es NIRGENDS gibt - der haeufigere Fall ist der halbe:
Methode da, aber nur in einer Ansicht, also Overlay in 2D vorhanden und im
3D lautlos weg.

Neue Gruppe `einseitige_sind_begruendet`: jede einseitige Methode muss in
`NUR_EINE_ANZEIGE` mit Begruendung stehen. 17 Stueck, alle benannt
(`set_sun_direction` - es gibt keine Sonne in einer 2D-Karte;
`overlay_roads` - im 3D echte Bandgeometrie; usw.). Der Test meldet auch
veraltete Registereintraege von selbst und hat dabei sofort zwei gefunden.

**Gegenprobe gemacht:** Eintrag entfernt -> Test schlaegt fehl mit
`set_sun_direction (nur 3D)`, wieder eingesetzt -> gruen.

## Tests

Gruen: `display_methoden_existieren` (3/3), `fluss_overlay`, `display_2d`
(alle 30 Darstellungen), `wege_geometrie`, `shader_paths`.

## Offen

* **Der Commit.** 89 Dateien, letzter Commit vom 12.08.
* **Nichts visuell bestaetigt** seit dem 12.08. - der stehende Blocker.
* B.2/B.3 pruefen, dann A.1 (Layerwert in der 2D-Koordinatenzeile).
* Entscheidung des Nutzers: Anzeige auf `biome_map_super` umstellen?

## Nachtrag 2026-08-25 (Nacht): Kuestenprofile geprueft, Wurzel der Regression gefunden

### Der Einfluss der spaeteren Stufen - gemessen und jetzt bewacht
Nutzervorgabe: *"profile pruefen (der Einfluss durch zb erosion,
erosionfilter, Fluesse, anderer kuesten etc muss natuerlich erkannt werden,
weil das die kueste auch betrifft)"*.

Berechtigt: `smoke_test_kuestenprofiltreue.py` misst auf `weltfeld()` - das
ist **Stufe 1 von vier**. Danach folgen Erosionsfilter, Flussnetz mit Taelern
und Redistribution.

**Gemessen (512 px), Aenderung im Kuestenband unter 350 m:**

| Stufe | p90 | Wasserlinie verschoben |
|---|---:|---:|
| Erosionsfilter | 7.5 m | 0.000 % |
| Fluesse und Taeler | 6.2 m | 0.015 % |
| beide zusammen | 12.1 m | 0.015 % |

Die Messung auf `weltfeld()` ist damit vertretbar - **aber das war bis jetzt
eine Annahme.** Neue Testgruppe `spaetere_stufen_aendern_die_kueste_kaum`
macht eine Messung daraus. Schlagen die Grenzen an, misst der Rest des Tests
ein Gelaende, das so nie angezeigt wird.

### Die Wurzel der Flachkuesten-Regression - gefunden

Beide roten Tests (`kuestenprofiltreue`, `archetyp_verteilung`) melden
dieselben drei Archetypen. Die Kette, Schritt fuer Schritt gemessen:

**1. Wieviel FESTLANDSkueste hat jeder Archetyp?**

    Schaerenkueste       0.0 km      Kykladen-Strand    21.1 km
    Dingle-Straende      1.1 km      Amalfi-Steilkueste 17.3 km
    Toskana-Straende     1.9 km      West-Cork-Buchten  14.2 km

Genau die drei gemeldeten haben fast keine. Toskana hat 10 Segmente, davon
**9 auf Inseln**; Dingle 2, davon 1.

**2. Warum landen sie auf Inseln?**

    Konturen        Kuestenlaenge   Saatstationen   Stationen je km
    6 lange            146.1 km          431             2.95
    69 kleine Inseln    14.1 km          359            25.44

**Kleine Inseln tragen 8.8 % der Kueste, bekommen aber 45.4 % der
Saatstationen - Faktor 8.6 zu dicht.**

**3. Warum ist das entscheidend?** Weil die Quote ueber STATIONEN verteilt
wird, nicht ueber Laenge:

    ziel_anzahl = {a["name"]: max(1, int(round(a["max_anteil"] * n)))}

mit `n` = Zahl der Stationen (`_saaten_bauen`, Zeile 2052). Fast die Haelfte
jedes Kontingents geht damit an Inseln, die keine 10 % der Kueste ausmachen.
Seltene Typen bleiben ganz dort haengen.

**KEIN FEHLER, sondern eine Kollision zweier Anforderungen.** Die Ueberdichte
kommt von `MIN_STATIONEN_JE_KONTUR = 16` - eine bewusste Massnahme gegen
"Tortenstueck"-Naehte auf kleinen Inseln (Kommentar bei `_saaten_bauen`).
Beide Anforderungen sind fuer sich richtig.

**VORSCHLAG (noch nicht umgesetzt, betrifft die Saat-Zuweisung):** die Quote
nach KUESTENLAENGE gewichten statt nach Stationszahl. Inseln behalten dann
ihre dichten Stationen fuer glatte Naehte, ohne die Anteile zu verzerren.

### Zwei eigene Fehlschluesse auf dem Weg, beide korrigiert
1. *"56 % der Segmente verletzen MIN_SEGMENT_M = 750 m, die Regel greift
   nicht."* **Falsch.** Alle 66 kurzen Segmente liegen auf kleinen Inseln -
   genau der Fall, den der Docstring als gewollt beschreibt. Auf den 48
   Festlandsegmenten verletzt KEINES die Mindestlaenge.
2. Die absoluten Profilabweichungen meiner Stufenmessung (Median 40 m) sind
   NICHT mit denen des Tests (3.4 m) vergleichbar - andere Bezugshoehe.
   Aussagekraeftig ist nur der Vergleich der Stufen untereinander.

## Nachtrag 2026-08-25 (Abend): nur noch Haupttaeler, Binnenbecken blau

### "Wir generieren ueberall Fluesse" - Ursache und Behebung
Nutzervorgabe: *"ich wollte damals nur die haupttaeler damit erzeugen. die
abzweige sollten weit unten noch taeler generieren aber dann schwaecht das
schnell ab (je nach regenmenge die angeschlossen ist)."*

**ZWEI EIGENE HYPOTHESEN, BEIDE GEMESSEN UND WIDERLEGT:**
* Schwelle 0.005 auf die angeschlossene Wassermenge: 63.4 -> 63.1 % der
  Landflaeche ueber 20 m ausgehoben. Nichts.
* `breite_faktor` von 0.60 auf 0.12: 54.8 -> 52.4 %. Fast nichts.

**Die Ursache zeigte erst diese Messung: 23.5 % der LANDFLAECHE sind
Flusspixel, Median-Abstand zum naechsten Lauf 1 Pixel (55 m).** Im Talprofil
ist `t = abstand/talbreite ~ 0.1`, damit `profil = (1-exp(-t))**form ~ 0.03`
- es wirken ueber 95 % des Talsogs, ueberall. Bei so kleinem Abstand ist die
Talbreite gleichgueltig.

**Zwei Hebel, beide auf `gebiet`** (= `netz["flaeche"]`, akkumuliert
Niederschlag mal Flaeche - also genau die "angeschlossene Regenmenge"):

1. `TALTIEFE_EXPONENT` 0.30 -> 0.70. Die 2220 Ordnung-1-Knoten (75 % aller
   Knoten) bekamen vorher 15.9 % der vollen Taltiefe, jetzt 1.4 %. Ordnung 3
   behaelt 21.5 %, Ordnung 4 36 %, Hauptstrom 99.6 %.
2. `TALTIEFE_MINDESTWASSER = 0.07` - darunter zieht ein Knoten gar kein Tal.
   Die Laeufe bleiben im Netz, werden gezeichnet und tragen Wasser.

**Ergebnis gegen den Stand von heute frueh:** mediane Abtragung 41.8 -> 7.7 m,
Land ueber 20 m ausgehoben 68.5 -> 37.2 %, unberuehrtes Land 8.4 -> 35.4 %.

Der Wert 0.07 kam nach einer Sichtpruefung ("etwas zu viel jetzt, dazwischen
waere gut") aus 0.05/0.10; die volle Tabelle steht im Kommentar.

### Eigener Fehler zurueckgedreht: VALLEY_WIDTH
Ich hatte ihn auf 0.60 gesetzt. **Die Beschreibung des Reglers sagte es
bereits: "0.5 heisst wirklich bis zur Mitte zwischen zwei Laeufen"** - 0.60
liegt darueber, es bleibt keine Hochflaeche. Zurueck auf 0.35. Breitere
Taeler fuer die grossen Fluesse kommen aus dem KONTRAST
(`TALBREITE_UNTERGRENZE` 0.22 -> 0.12, Exponent 0.40 -> 0.55): Hauptstrom
688 statt 402 m, feinste Baeche 104 statt 116 m, Kontrast 3.47x -> 6.63x.

### Binnenbecken unter 0 m waren Landbiome
Nutzerbild: die 0-m-Kontur sichtbar, das Becken darin mit Waldbiomen.

**Ursache:** `_detect_ocean_connectivity()` ist eine Flutfuellung VOM
KARTENRAND - eine abgeschlossene Senke erreicht sie nie, und wenn die
Wassersimulation sie auch nicht als See fuellt, faellt sie durch beide
Raster. Gemessen: 83 Pixel in 23 Becken, bis -73.5 m tief.

Behoben in der KLASSIFIKATION (nicht in der Anzeige), damit 2D und 3D
dieselbe Karte bekommen. Zugewiesen als **See, nicht Ozean** - die Becken
sind per Definition nicht mit dem Meer verbunden, und Seewege sowie die
Seegliederung lesen `ocean`.

**Dabei ein zweiter, groesserer Fehler gefunden:** in `biome_map_super` lagen
**3142 `river_bank`- und 467 `lake_edge`-Pixel unter dem Meeresspiegel** -
Flussufer auf dem Meer. Die Abstandsfelder bekommen die Heightmap nicht zu
sehen und reichen in beide Richtungen. In `biome_map` fiel es nie auf, weil
dort die harten Wasserzuweisungen gewinnen; erst das Supersampling macht
Pixel daraus. Beide Karten jetzt: 0 Nicht-Wasser-Pixel unter 0 m, Ufer an
Land erhalten.

## Nachtrag 2026-08-25 (spaet): Klippenbefund, Fluss-Splines, stehende 3D-Regel

### Warum man die Klippen an der Kueste nicht sieht - GEKLAERT
Nutzerfrage: *"wird die slopemap vor oder nach den vektoren berechnet?"*

**Die Reihenfolge war NICHT das Problem.** `terrain.redistribution` ruft
`weltfeld()`, und dort wird die Vektorkueste angewandt; `erosion.slope` und
die Klippenrechnung im Biom bekommen beide die KOMBINIERTE Heightmap.
Nachgemessen: die kombinierte war sogar bitgleich mit
`terrain.redistribution/heightmap` (Erosion per Schalter aus). Die
Klippenrechnung benutzt ausserdem gar nicht die `slopemap`, sondern ihren
eigenen Gradienten.

**Die echte Ursache sind zwei Dinge:**

1. Das Profil ist FEINER ALS DAS RASTER. Die gemessenen Kuestenprofile sind
   nur auf ihren ersten 50-100 m steil (Moher 48.9 Grad, Amalfi 27.4,
   Santorini 26.5, Algarve 24.9, Dingle 1.9). Und 50 m sind 0.30 px bei
   128 px, 1.20 px bei 512, 2.40 px bei 1024.
2. Die meisten Archetypen sind GAR NICHT SO STEIL - von den fuenf oben
   ueberschreitet nur Moher 45 Grad.

Gemessen, Wasserlinie (<200 m) gegen Inland (>1 km), Anteil ueber 45 Grad:
256 px 0.16 % gegen 2.80 %, 512 px 0.73 gegen 6.63, 1024 px 1.93 gegen 11.43.
**Die Kueste ist der FLACHSTE Teil der Karte**, der p90 bleibt bei allen
Groessen zwischen 26 und 30 Grad.

Folgerung, ausfuehrlich im Docstring von
`_calculate_cliff_probabilities()`: Klippen sind bei diesem Massstab ein
GEBIRGS-Merkmal. Wer Kuestenklippen sehen will, muss `cliff_slope` auf etwa
30 Grad senken - dort liegt der gemessene p90.

### Fluesse als Spline statt als Streckenzug
Nutzerbefund: *"die fluesse sind hier sehr zackig gezeichnet ... kann man
das mit splines aufweichen?"*

Beide Stellen liefen die Kanten des Knotengraphen als GERADE SEHNEN ab -
`taeler_eingraben()` UND der Maskenaufbau in `terrain_generator.py`. Neu:
`hauptkinder()` und `kantenpunkte()` in `core/terrain_weltfluesse.py`,
zentripetales Catmull-Rom (alpha 0.5). Beide Stellen benutzen jetzt
dieselbe Funktion - sonst laege der gezeichnete Fluss neben seinem Tal.

Zentripetal und nicht uniform, weil die Knotenabstaende sehr ungleich sind
(150 bis 1200 m); uniformes Catmull-Rom bildet dort Schleifen. Die Kurve
geht EXAKT durch die Knoten, weil an ihnen die Sohlenhoehen haengen
(nachgemessen: groesste Abweichung 0.000000 m).

**Gemessen am Hauptstrom (384 px, 40 Knoten): groesster Knick 143.5 -> 33.3
Grad, Knicke ueber 30 Grad von 24 auf 2.** Lauflaenge +1.9 %.

**Eigener Messfehler auf dem Weg:** die erste Messung summierte die
Drehwinkel ueber alle Abtastpunkte und meldete die Spline als SCHLECHTER
(6840 -> 9146 Grad). Falsches Mass - eine Spline verteilt die Drehung auf
viele kleine Schritte, sie beseitigt sie nicht. Entscheidend ist der
GROESSTE Knick, nicht die Summe.

### STEHENDE REGEL: was in 2D sichtbar ist, gehoert auch in 3D
Nutzervorgabe: *"kann man irgendwo festhalten dass wenn du etwas umsetzt es
immer auch in 3D gleich umgesetzt wird? weil sonst muss ich das immer
wieder sagen."*

Steht jetzt in **CLAUDE.md** als eigener Abschnitt, mit der Tabelle der drei
Faelle, in denen genau das schiefging (24.08. `overlay_river_generations`,
25.08. `overlay_river_network` und `overlay_settlements`, alle drei im
Biome-Reiter bzw. Fluss-Reiter, alle drei lautlos).

**Sofort umgesetzt:** der Siedlungs-Haken im Biome-Reiter zeichnet jetzt
auch in 3D (RGBA-Skin ueber `update_overlay_data`, derselbe Weg wie
`SettlementTab.apply_3d_overlays`).

**Bewacht von** `smoke_test_display_methoden_existieren.py` mit zwei
getrennten Registern:
* `NUR_EINE_ANZEIGE` - einseitig und richtig so (12 Eintraege)
* `FEHLT_IM_3D` - einseitig und eine SCHULD (5 Eintraege): `overlay_regions`,
  `overlay_region_grid`, `overlay_city_boundary_contour`,
  `set_contour_reference_heightmap`, `overlay_plot_boundaries`.
  Die Liste darf nur schrumpfen.

**Eigener Fehler im Waechter, durch die Gegenprobe gefunden:** die fuenf
Schuldeintraege standen zuerst in BEIDEN Registern, damit war die
Schuldliste reine Zierde - sie herauszunehmen aenderte nichts. Jetzt
schlaegt der Test bei Ueberschneidung fehl. Gegenprobe wiederholt: Eintrag
entfernt -> FAIL, wieder eingesetzt -> PASS.

### p1 der Kuestenprofile 30 % naeher an die Kueste
Nutzervorgabe: *"die kuestenprofile sind teilweise etwas zu agressiv, ich
wuerde p1 noch naeher an die kueste ziehen (30% naeher ran)"*.
`PROFIL_VOLL_FLACH_M` 300 -> 210, `PROFIL_VOLL_STEIL_M` 350 -> 245. p2
bleibt, die Uebergangszone wird dadurch laenger und sanfter.

**NEBENWIRKUNG, ehrlich benannt:** `smoke_test_kuestenprofiltreue.py` meldet
seither Toskana-Straende zusaetzlich (18 m) - vorher nur Dingle (57 m).
Plausibler Grund: das gemessene Profil setzt sich kuerzer durch, danach
uebernimmt das Rauschgelaende, und das ist an einem FLACHEN Archetyp hoeher
als die Vorlage. Der Wunsch "weniger aggressiv" und die Zusicherung "flache
Kuesten bleiben flach" ziehen also gegeneinander. **Beide Tests waren schon
vor dieser Aenderung rot** (siehe TESTBERICHT Abschnitt 3), eine saubere
Zuordnung ist daher erst nach der Klaerung jener Regression moeglich.

### NOCH OFFEN aus dieser Runde
* **Binnenbecken unter 0 m wird mit Biomen statt blau gezeichnet** - der
  Nutzer hat ein Bild geschickt, in dem die 0-m-Kontur sichtbar ist, das
  Becken darin aber Landbiome traegt. In 2D UND 3D zu beheben.
* **Taeler: breiter und flacher, dicke Fluesse deutlicher.** Gemessen:
  `river_valley_width` 0.35 -> 0.70 verdoppelt `breite_feld` (445 -> 891 m),
  `river_incision_share` 0.30 -> 0.15 senkt die mediane Abtragung von 44 auf
  34 m. Der Kontrast dick/duenn steckt in zwei fest verdrahteten Zahlen
  (`untergrenze = 0.22`, Exponent 0.40) - noch nicht geaendert.

## Nachtrag 2026-08-25: flow_map, cliff_slope, voller Testlauf

### flow_map: Skala war um zwei Groessenordnungen daneben
Gemessen an der vollen Pipeline (`flow_accumulation` aus water.flow_network -
`flow_map` ist nur der Name auf dem WaterData-Objekt): p50 = 581, max = 6408.
Mit der alten Skala 0..50 sassen **98.26 % aller Wasserpixel ueber dem
Maximum** und bekamen dieselbe Farbe; nur 1.7 % lagen im Verlauf.

Jetzt **logarithmisch 20..8000**. Linear 0..3500 haette p10 auf 4 % und p50
auf 17 % der Skala gedraengt - alle kleinen Laeufe saehen gleich aus. Log
verteilt: p1 bei 10 %, p50 bei 56 %, max bei 96 %. 2D und 3D teilen sich
dieselbe Tabelle, eine Aenderung wirkt in beiden.

### cliff_slope 60 -> 45 Grad
Bei 60 Grad gab es auf der ganzen Karte **eine einzige Klippe**. Kein
Rechenfehler, sondern eine Schwelle, die das gerasterte Gelaende nicht
erreicht: 60 Grad brauchen bei 117 m/px einen Hoehensprung von 203 m zwischen
Nachbarpixeln.

Gemessen auf der Heightmap, mit der die Biom-Stufe wirklich rechnet (NICHT
auf dem steileren Rohgelaende): 30 Grad = 18.68 % des Landes, 45 = 4.58 %,
60 = 0.15 %. **Gegenprobe am anderen Ende der Kette: `cliff` in
`biome_map_super` von 1 auf 159 Pixel** (0.24 % der Karte, vergleichbar mit
`beach` mit 312).

Ehrliche Einschraenkung im Code vermerkt: der Anteil haengt an der
Aufloesung (bei 1024 px ueberschreiten 9.4 % der Landflaeche 45 Grad gegen
4.4 % bei 256 px).

### Voller Testlauf und neuer Testbericht
Neu: `tools/testlauf.py` - je Datei ein eigener Prozess (sonst verfaelscht
die numpy-Allokator-Nachwirkung aus der Perf-Sitzung die Zeiten), erkennt
alle drei Fehlerformate der Suite.

**50 von 61 bestanden, 784 s.** `docs/TESTBERICHT.md` komplett neu - die alte
Fassung kannte 40 der heute 61 Dateien.

**Der Lauf hat gefunden, was ein statischer Audit nicht finden kann.** Vorher
wurden alle 61 Dateien statisch geprueft: alle importieren sauber, keine
verweist auf eine geloeschte Datei. `smoke_test_settlement_roads.py` sah
dabei gesund aus und prueft trotzdem seit Wochen **quadratische** Hangkosten,
waehrend der Code seit der Nutzervorgabe **exponentiell** rechnet (500.000
gegen erwartete 2.5 - der Testpunkt liegt bei 45 Grad und traf die Sperre).
Behoben, jetzt zwei Zusicherungen: Sperrwert oberhalb der Grenze,
exponentielle Kurve darunter, plus die Bedingung, dass sie schneller als
linear waechst.

### DER WICHTIGSTE BEFUND: zwei Tests sind seit gestern rot
Die Uebergabe vom 24.08. (23:47) fuehrt beide als gruen:
`smoke_test_kuestenprofiltreue.py` 2/2 und
`smoke_test_archetyp_verteilung.py` 3/3. Heute beide rot, und **beide
betreffen flache Kuesten** - Dingle-Straende taucht in beiden auf (57 m statt
flach; 0.17-facher Anteil). Das sieht nach EINER Ursache aus.

Diese Sitzung hat keinen Terrain-, Kuesten- oder Archetypcode angefasst
(nur Anzeige, Farbskalen, Parameterregister, Tests, tools) - und
`cliff_slope` wurde erst NACH dem Lauf geaendert. **Vermutung, ungeprueft:**
die Aenderung stammt vom Ende der Sitzung vom 24.08., nachdem die Uebergabe
geschrieben war, oder die Tests sind nicht deterministisch.

### Weitere Befunde aus dem Lauf
* **`regionen_welt` ist schlechter geworden**, nicht nur weiterhin rot: Naht
  1.804 gegen 1.540 in der Uebergabe. Und eine Zeile, die dort fehlt:
  `Landschaft haengt an der Pixelzahl (r = +0.9603)` - ein Massstabsproblem,
  kein Eichungsproblem.
* **`erosion_map`/`sedimentation_map` haben dasselbe Skalenproblem wie
  flow_map**, nur andersherum: typischer Wert 0.22 bzw. 0.32 liegt UNTER der
  Skalenuntergrenze 0.5. Beide Karten zeigen fast ueberall die unterste Farbe.
* **`erosion_gpu_parity`: Faktor 385 zwischen GPU und CPU** (38.5 gegen 0.1 m).
* Der alte Bericht fuehrte drei Erosionsfehlschlaege auf `EROSION_AKTIV`
  zurueck - **geprueft: keine der drei Dateien erwaehnt den Schalter.**

### B.1 Flussnetz-Overlay (davor in dieser Sitzung)
`river_order` hatte keine Farbskala; der Biome-Reiter zeigte ein ANDERES
Flusssystem als der Fluss-Reiter (`flow_map` per 90.-Perzentil statt
`river_generation`) und nur in 2D - `overlay_river_network` gibt es nur auf
MapDisplay2D, im 3D fiel die hasattr-Weiche lautlos aus. Dieselbe
Fehlerklasse wie am 24.08.

Testluecke geschlossen: `smoke_test_display_methoden_existieren.py` verlangte
nur, dass ein Name auf MINDESTENS EINER Anzeige existiert. Neue Gruppe
`einseitige_sind_begruendet` - 17 einseitige Methoden, alle namentlich
begruendet. Gegenprobe gemacht.

### B.2/B.3 geprueft - nichts zu bauen
`smoke_test_flussstufen.py` bestaetigt alle vier Wasserstufen in beiden
Biomkarten. Straende 312 Pixel, aber **nur in `biome_map_super`**.

Fuer die offene Entscheidung gemessen - nur in der feinen Karte vorhanden:
cliff 1 (jetzt 159), beach 311, lake_edge 3177, river_bank 4878,
snow_level 1, alpine_level 1207. Zusammen rund **14.6 % der Karte**.
`snow_level` ist zu Recht leer (2000 m Schwelle, Gelaende bis ~1016 m).
**Der Biome-Reiter hat den Umschalter bereits** - es geht nur um die
Voreinstellung.

---

# 2026-08-24 (Teil 2) — Leistung, Kuestenprofile in Metern, Fluesse nach Wasser

**Ausgangslage:** HEAD `950a949`, uncommitted auf `main`. Kein Worktree.

## 1. Ein Wicklungsfehler zerschnitt das 3D-Gelaende in Streifen

Nutzerbefund mit Bild: *"warum habe ich kein normal aussehendes mesh mehr
... von unten ist es uebrigens nicht unterbrochen."*

Der letzte Halbsatz war der ganze Beweis. `gui/widgets/kuesten_schnitt.py`
wickelte die Dreiecke gegen den Uhrzeigersinn (signierte Flaeche +0.5), das
Gitter in `map_display_3d` im Uhrzeigersinn (-0.5). Mit
`glFrontFace(GL_CW)` + `glCullFace(GL_BACK)` wird dann exakt verkehrt herum
gecullt: alles zur Kamera hin verschwindet, sichtbar bleiben nur die
abgewandten Rueckhaenge.

**Warum der Test das nicht fand:** er prueft, dass die Wicklung EINHEITLICH
ist. Die richtige Frage war, ob sie zum Gitter PASST. Neue Gruppe
`wicklung_wie_gitter` in `tests/smoke_test_kuesten_schnitt.py`.

## 2. Leistung — vier Knoten sind 84 % der Ladezeit

Neu: `managers/teilschritte.py` (Teilzeiten INNERHALB eines Knotens, fuer
Log und Ladebalken) und `tools/pipeline_kritischer_pfad.py`.

**Parallelitaet zwischen Knoten bringt 7 %** — der Abhaengigkeitsgraph ist
praktisch eine Kette (25 Ebenen, breiteste 3 Knoten, mittlere Breite 1.5).
Alles muss INNERHALB der vier grossen Knoten passieren.

Umgesetzt, alle A/B im selben Prozess gemessen, alle Ergebnisse bitgleich
bzw. Punkt-fuer-Punkt identisch:

| Massnahme | Faktor |
|---|---:|
| A-Stern mit numba (`core/wegsuche_schnell.py`) | 14-16x |
| `archetyp_felder`: Doppelarbeit + Tabellen-Nachschlag | 23x |
| `taeler_eingraben`: Stuetzstellen vektorisiert | 7.7x |
| Erosionsfilter bandweise (sequenziell) | 3.0x |
| Nahmaske in `hoehe()` | 2.5x |
| `poisson_points`: numpy-Skalarzugriffe raus | 2.4x |

`settlement.pathfinding` 60 s -> 3.8 s (Median je Route 0.85 s -> 0.055 s).

**FUENF Vorschlaege wurden durch Messung WIDERLEGT** und sind im Code an
Ort und Stelle mit ihrem Messwert kommentiert, damit sie nicht wiederholt
werden:

* `K_KANDIDATEN` von 24 auf 12 senken — der MEDIAN-Punkt braucht 18.
* `_auf_strecken` in einen Durchgang ziehen — langsamer (0.826 gegen
  0.690 s), die Zwischenfelder waeren 111 MB statt zweimal 55 MB.
* Die Kandidatenschleife auf 22 Kerne verteilen — durch numba
  gegenstandslos.
* Den 5x5-Nachbarschaftstest in `poisson_points` vektorisieren — 1.7 bis
  3x LANGSAMER, es sind nur ein bis fuenf belegte Nachbarn und die
  Schleife bricht beim ersten Treffer ab.
* **Der parallele Erosionsfilter.** 5x schneller (9.9 -> 1.3 s), aber JEDE
  nachfolgende numpy-Rechnung im selben Prozess dauerhaft **2.4x
  langsamer** (Referenzlast 1.47 -> 3.55 s, ohne Erholung). `weltfluesse`
  stieg dadurch von 8.4 s ueber 17.9 s auf 31.7 s — der Knoten wurde als
  Ganzes langsamer, obwohl sein teuerster Teilschritt schneller war.
  Ursache ist die gleichzeitige Allokation grosser Felder aus 22 Threads.
  Die sequenzielle Fassung bringt 3.0x ohne die Nebenwirkung; der Gewinn
  kam aus der Cache-Lokalitaet, nicht aus der Parallelitaet.
  `tests/smoke_test_erosionsfilter_baender.py::keine_nachwirkung` faengt
  den Fall kuenftig — an der Zeit des Filters allein ist er NICHT zu sehen.

**MESSWARNUNG:** diese Maschine schwankt um Faktor 2-3. Dieselbe feste
Referenzlast (5x `np.fft.fft2` auf 2000x2000) mass zwischen 0.80 s und
1.91 s ohne erkennbare Fremdlast (CPU 18 %, 12.9 GB RAM frei, kein
Swapping). Derselbe unveraenderte Knoten `terrain.redistribution` mass an
einem Nachmittag 43, 48, 58 und 96 s. **Nur A/B im selben Prozess ist
belastbar.**

Details: `docs/PERFORMANCE_2026-08-23.md`.

## 3. Kuestenprofile: von normiert auf METER

Anlass: *"ein strandgebiet soll nicht so stark an noise angepasst werden
wie klippen ... flache kueste bleibt flach, hohe kueste ist
hoehenvariabel."* Die Weissmeer-Flachkueste stand bei **187 m**, ihre
Vorlage sagt 15-30 m.

**Die Ursache:** die Zielhoehe entstand als
`max(hinterland * ueberhoehung + SOCKEL, MIN_ANTEIL_KATALOG * katalog)`,
und der erste Term gewann fast immer. Bei 150 m Rohgelaende hinter der
Kueste sind das 150 * 1.06 + 12 = 171 m — genau der gemessene Wert. Das
rohe Rauschgelaende bestimmte die Kuestenhoehe praktisch allein.

**Zwei Sackgassen zuerst**, beide gemessen, beide verworfen:

1. Die Bandeinblendung daempfen. Wirkungslos, weil im Plateau ohnehin
   `staerke = 1` gilt und `rest` dort 0 ist.
2. Die Hinterlandkopplung an den KATALOGWERT binden. Die Straende wurden
   deutlich besser (Toskana 0.374 -> 0.031), aber die Foerdenkueste fiel
   von RMS 0.040 auf 0.307: der Katalogwert ist je REGION tabelliert und
   trennt Strand und Klippe nicht — Kola-STEILkueste 30 m,
   Kykladen-STRAND 69 m.

**Der Umbau** nach Nutzerspezifikation: Profil in METERN statt gestreckt,
feste Zonen (0-350 m volles Profil, 350-500 m Uebergang ins Rauschen,
0 bis -200 m ins Ozeanprofil), Einfluss nur innerhalb der eigenen Insel
(`ndimage.label`), Strahltiefe je Segment variierbar angelegt.

**Neue Datengrundlage:** 27 Archetypen, jeder aus SEINER eigenen
Vorbildkueste (`tools/archetyp_vorbilder.py`, 22 DEM-Kacheln neu
abgerufen). Vorher gab es 11 Strecken fuer 27 Archetypen, und die Schnitte
einer Region wurden nach Steilheit gedrittelt — das ergab drei Kurven mit
identischer FORM und nur verschiedener HOEHE (normiert 0.28/0.26/0.30 nach
50 m). Dazu ein Zirkelschluss: nach h(150 m) sortieren und dann h(150 m)
messen trennt zwangslaeufig nur in dieser Groesse.

Vier Ausschnitte mussten nachgebessert werden; bei zweien wurden je vier
Kandidaten gemessen, bevor einer eingetragen wurde:
Isonzo-Muendung (0 m ueber 900 m, reines Schwemmdelta) -> Roja bei
Ventimiglia; Rybachi (Kachel zu 100 % an Land) -> Teriberka Ostkap;
Ponta da Piedade (flacher als die Nachbarbucht) -> Cabo de Sao Vicente;
La Concha (Berge im Ruecken) -> Donana.

**Ergebnis:** flache Kuesten treffen ihre Vorlage auf 0-2 m, Median ueber
20 Gruppen 2.6 m. Weissmeer-Flachkueste 187 m -> 8 m (Soll 5 m).

**Drei Fehler beim Umbau, alle gemessen und behoben:**

1. **Die Mittelung war zu breit.** Im Plateau ist das Blendgewicht fuer
   JEDEN Abschnitt exakt 1 — bei 350 m Zone wurde eine Fjordwand
   gleichberechtigt mit der flachen Schaerenkueste nebenan gemittelt.
   Jetzt zwei getrennte Gewichte: `max` fuer WIEVIEL Kuesteneinfluss, ein
   quadratisch fallendes fuer WELCHE Kueste. Median 8.2 -> 3.4 m.
2. **Die Kuestenlinie verschob sich** um 0.24-0.35 % der Pixel — die
   Meterprofile beginnen bei 0, der alte Sockel von 12 m fehlte. Statt
   einen Sockel aufs Profil zu legen (der die flachen Archetypen
   verfaelscht haette) wird das Vorzeichen erzwungen.
3. **Der Test mass das Falsche.** Er verglich normierte Formen — eine
   Kueste mit richtiger Form und voellig falscher Hoehe war gruen. Genau
   so blieb die Weissmeer-Flachkueste bei 187 m unentdeckt. Er misst jetzt
   Meter.

**Nicht loesbar mit diesen Daten:** ein gemessenes Unterwasserprofil.
COP30 ist ein Oberflaechenmodell und setzt offenes Meer auf 0; alle 27
Archetypen liegen zwischen -0.1 und -1.8 m und aendern sich ueber 250 m
nicht. Der Uferuebergang blendet deshalb von 0 auf die Seegrad-Tiefe —
vorher stand dort eine 10-m-Stufe direkt an der Wasserlinie.

**OFFEN UND NACH VIER GEPRUEFTEN HYPOTHESEN UNGEKLAERT:** die drei
steilsten Archetypen (Fjordwand 22 statt 120 m, Cinque Terre 24 statt 99,
Amalfi 42 statt 80).

**Der Kern des Raetsels:** `hoehe()` liefert AN DEN FJORDWAND-SAATPUNKTEN
richtige Werte. Gemessen ueber 14 Stationen und 102 Landrichtungen bei
d = 150 m: **Median 144 m** bei Soll 120 m - also sogar darueber. Der Test
misst an denselben Archetypen 22 m. Dieselbe Funktion, verschiedene
Ergebnisse.

Vier Hypothesen wurden gemessen und WIDERLEGT:

1. **Der Hoehendeckel greift.** Nein. `max(ZIEL_JE_HINTERLAND * d_max,
   h_max)` liegt im Median bei 2740 m, im Minimum bei 83 m.
2. **Die Kueste greift zu kurz.** Auf Nutzerwunsch wurde der
   Skerrheim-Griff um 50 % vertieft (`KUESTEN_TIEFE_JE_REGION`, p1/p2 von
   350/500 auf 525/750). Die Fjordwand blieb bei 23 m.
3. **Die Mischung mit flachen Nachbarn zieht herunter.** Plausibel - das
   Skerrheim hat 2106 Schaerenkuesten-Pixel (h(150 m) = 14 m) gegen 1299
   Fjordwand-Pixel. Der Mischexponent wurde von 2 auf 4 und 6 erhoeht:
   der MEDIAN ueber alle Gruppen verbesserte sich von 2.2 auf 1.8 m, die
   Fjordwand blieb bei 22 m.
4. **Die Basis sinkt landeinwaerts.** Trifft an einzelnen Stationen zu
   (105 m bei 50 m, 59 m bei 150 m), erklaert aber nicht, warum die
   Hoehenfunktion an denselben Stellen 144 m liefert.

**GEFUNDEN 2026-08-24 (spaeter am Tag): `archetyp_felder()` rechnete die
Staerke mit der ALTEN Reichweite.** Dort stand
`clip(1 - d / reichweite_m)^2`; `reichweite_m` ist die Groesse aus dem
frueheren Modell (110-349 m je Region), waehrend `_hoehe_block()` seit dem
Zonenumbau mit `voll_m`/`uebergang_m` rechnet (350/500 m, im Skerrheim
525/750 m). Bei einem Pixel 392 m vor der Kueste wurde
`1 - 392/296` negativ und auf 0 geklemmt.

Gemessen VOR dem Fix: Fjordwand-Pixel `staerke` 0.07, Schaerenkueste 0.00.
NACH dem Fix: beide 1.00, und 52 % der Landpixel haben ueberhaupt eine
Staerke.

**Die Hoehenfunktion war davon NICHT betroffen** - sie rechnet
eigenstaendig. Betroffen war alles, was `kuesten_staerke` LIEST:
`_seetiefe_aus_archetyp()` fuer die Meerestiefe (5 Fundstellen) und die
2D-Anzeige im Kuestentypen-Modus (3 Fundstellen).

**Der Fjordwand-Befund relativiert sich damit.** An den Pixeln, die
tatsaechlich als Fjordwand markiert sind, liefert die Hoehenfunktion 218 m
bei einem Sollwert von 392 m - nicht die 22 m, die
`smoke_test_kuestenprofiltreue` meldet. Der Test misst entlang der
Kuestennormalen und ordnet ueber `kuesten_archetyp` zu; mit der falschen
Staerke griff diese Zuordnung ins Leere.

**Ebenfalls gemessen:** ohne die Vektorkueste trifft das Skerrheim seine
Vorbild-Hoehenverteilung fast exakt (h_p25 0.29 gegen 0.30 bei Geiranger),
mit ihr faellt sie auf 0.02. Die Kuestenlinie selbst ist in beiden Faellen
identisch - es ist allein die Hoehenverteilung in Kuestennaehe.

### ES GAB KEINE GROSSEN FLUESSE — 2026-08-24

Beim Pruefen von Block B.2/B.3 (*"die grossen fluesse koennen als overlay
bei biome drin sein"*) gezaehlt: `water_biomes_map` enthielt
**ausschliesslich Creeks und Seen**. Die Stufen `river` (2) und
`grand_river` (3) kamen NIE vor - auf keiner Karte.

```
creek-Schwelle                        1716.77
river braucht  4x                     6867.08
grand  braucht 20x                   34335.41
groesster Abfluss der ganzen Karte     6408.09     <-- verfehlt beide
```

**URSACHE - zwei Masstaebe vermischt.** Die Creek-Schwelle ist ein
PERZENTIL der wasserfuehrenden Zellen und passt sich der Verteilung an.
Die Faktoren fuer River und Grand River waren dagegen ABSOLUTE Vielfache
davon. Zwischen einem hohen Perzentil und dem Maximum liegt in einer
Abflussverteilung aber systematisch weniger als eine Groessenordnung -
hier Faktor 3.73. Ein absoluter Faktor 20 darauf ist nicht streng,
sondern unerfuellbar.

**ABHILFE:** die Faktoren wirken jetzt auf die HAEUFIGKEIT. Ein River ist
der Lauf, der zu den obersten (abundance / 4) gehoert, ein Grand River zu
den obersten (abundance / 20). Das ist die Bedeutung, die die Namen immer
schon nahelegten - "vier mal so selten", nicht "vier mal so viel Wasser".

| | vorher | nachher |
|---|---:|---:|
| creek / river / grand_river | 1495 / **0** / **0** | 1119 / 299 / 77 |
| Anteil mindestens River | 0 % | 25.2 % (Soll 25.0) |
| Anteil Grand River | 0 % | 5.2 % (Soll 5.0) |

**WARUM ES NIEMAND GEMERKT HAT:** die Klassifikation lief fehlerfrei
durch und lieferte eine plausible Karte voller Baeche. Nichts stuerzte
ab, nichts warnte. Dasselbe Muster wie beim Archetyp-Verteilungsbug
weiter unten - ein Ergebnis, das von einem richtigen nicht zu
unterscheiden ist, solange niemand nachzaehlt.
`tests/smoke_test_flussstufen.py` zaehlt jetzt nach.

**NEBENBEI GEKLAERT: die Straende fehlen NICHT.** Es gibt zwei
Biomkarten mit unterschiedlicher Aufgabe:

  * `biome_map` - die grobe. Enthaelt die Wasserstufen, aber KEINE
    Wahrscheinlichkeits-Biome (beach, cliff, lake_edge, river_bank,
    snow/alpine_level). Alle sechs sind dort 0.
  * `biome_map_super` - die feine. Erst das Supersampling setzt die
    Wahrscheinlichkeiten in Pixel um; dort liegen 311 Strandpixel.

Wer Straende sehen will, muss also `biome_map_super` anzeigen. Das ist
Bauart, kein Fehler - und die Vermutung in ANZEIGE_UND_SEEN.md, es sei
ein reines Anzeigeproblem, war fuer die STRAENDE richtig und fuer die
GROSSEN FLUESSE falsch.

### FLAECHENEICHUNG: alle neun Regionen auf ihren Zielwert — 2026-08-24

Nutzerentscheidung nach dem Verteilungsfix: *"lass uns zielwert 0.8 fuer
fjordland festlegen, aber dann muessen wir noch etwas mehr flaeche
bekommen ... insgesamt bekommt halt jede region einen bestimmten
zielwert der erreicht werden soll."*

`ZIELWERT["Skerrheim"] = 0.80` (wie das Nevadin - Skerrheim verliert
DOPPELT, erst ein Drittel ans Wasser, dann die Haelfte des Rests an zu
steile Haenge).

Der Ausgleich laeuft ueber `flaeche_soll`. Der Parameter gab es schon,
aber nur drei Regionen hatten ihn gesetzt. Von Hand ist er kaum
einzustellen: **es ist ein Nullsummenspiel** - der Index misst gegen den
MEDIAN, also druckt jede wachsende Region alle anderen nach unten, und
jede Runde kostet eine volle Weltberechnung.

Deshalb `tools/flaeche_eichen.py`: ein logarithmischer Regelkreis, aus
demselben Grund logarithmisch wie die Eichung in `voronoi_regionen()` -
ein linearer Schritt schwingt, weil der Vorteil multiplikativ wirkt.

```
Runde 0: groesste Abweichung 0.261, mittlere 0.127
Runde 1: groesste Abweichung 0.204, mittlere 0.075
Runde 2: groesste Abweichung 0.102, mittlere 0.038  -> fertig
```

| | vorher | nachher |
|---|---:|---:|
| `smoke_test_regionen_fairness` | **1/2 Gruppen** | **2/2 Gruppen** |
| Spanne beste/schlechteste Region | 2.10x | **1.25x** |
| Skerrheim Index (Ziel 0.80) | 0.64 | **0.80** |
| Skerrheim Flaeche / nutzbar | 9652 / 33 % | 10920 / 50 % |

Die eingeregelten Werte stehen jetzt fest in `REGIONEN`; die ZIELWERTE
bleiben bewusst in `tests/smoke_test_regionen_fairness.py`, weil sie eine
Entscheidung ueber das Zielbild sind und keine ueber die Rechnung.

### URSACHE GEFUNDEN UND BEHOBEN — 2026-08-24

Die Fjordwand erreichte 23 m, wo ihr Profil 128 m verlangt. Nach dem
Ausschluss von Hoehendeckel, Reichweite, Mischung, Inselschluss und
`staerke` blieb nur, `_hoehe_block()` an einem einzelnen Punkt
mitzurechnen. Das Ergebnis war eindeutig:

```
Station 117, Punkt 150 m landeinwaerts, Archetyp Fjordwand
   hoehe():  14.6 m
   Platz 0: d=149.2 m   Profil h(150) = 14.7 m      <-- Schaerenkueste!
   Platz 1: d=443.3 m   Profil h(150) = 14.7 m
   Platz 2: d=505.1 m   Profil h(150) = 14.7 m
```

Die Hoehenfunktion war die ganze Zeit RICHTIG - sie bekam nur das falsche
Profil. Die Station gilt als Fjordwand, aber jeder Kuestenabschnitt in
ihrer Naehe traegt das Schaerenkuesten-Profil.

**Die Zaehlung zeigte, wie gross das Problem wirklich war:**

| | vorher | nachher |
|---|---:|---:|
| Archetypen ohne ein einziges Segment | **8 von 27** | 0 von 27 |
| Fjordwand: Saatstationen → Kuestenanteil | 13 → **0.4 %** | 13 → 5.2 % |
| Algarve-Klippen: Saatstationen → Anteil | 8 → **15.5 %** | 8 → 1.6 % |
| schlimmste Verzerrung | **18x** | 3.6x |
| Segmente / Medianlaenge | 115 / 169 m | 148 / 502 m |

Acht Archetypen - Fjordbucht, Foerdenkueste, Kotor-Steilfjord,
Labrador-Buchten, San-Sebastian-Bucht, Costa-Brava-Buchten,
Dalmatien-Klippen, Alpine-Flussmuendung - kamen auf der fertigen Karte
**ueberhaupt nicht vor**. Die Arbeit, alle 27 Profile aus echten
Vorbildern zu vermessen, war fuer knapp ein Drittel davon wirkungslos.

**URSACHE:** Die Zuweisung in `_saat_bauen()` sortierte die Stationen
einer Region nach Score und schnitt oben ab - ohne jeden Bezug darauf,
welche Station neben welcher liegt. `jitter` war weisses Rauschen je
Station. Ein Typ landete damit in Laeufen von ein bis zwei Stationen,
also 420-840 m. `MIN_SEGMENT_M` verlangt 750 m, und das Einschmelzen gibt
einen zu kurzen Lauf an den LAENGEREN Nachbarn weiter. Wer schon lang
war, wurde laenger - ein sich selbst verstaerkender Prozess, an dessen
Ende ein Archetyp mit 0.8 % der Stationen 15.5 % der Kueste hielt.

**ABHILFE** (`SAAT_KOHAERENZ_STATIONEN = 3.5`): Jitter und lokale Hoehe
werden entlang der Bogenlaenge geglaettet, Kontur fuer Kontur
(`_bogen_glaetten`, `_bogen_rauschen`). Benachbarte Stationen bekommen
dadurch aehnliche Scores und denselben Typ. Die ZIELANTEILE sind
unberuehrt - `ziel_anzahl` und die Auswahlschleife sind unveraendert, es
aendert sich nur, WELCHE Stationen ein Typ bekommt, nicht wieviele.

**DER WERT WURDE GEMESSEN, nicht geraten** (512 px, Seed 20260804):

| sigma | Ausfaelle ab 8 Stationen | Segment-Median |
|---:|---|---:|
| 2.5 | 2 (Fjordbucht, Moher-Klippen) | 505 m |
| **3.5** | **1 (Algarve-Klippen, 8 Stationen)** | 502 m |
| 5.0 | 2 (Algarve, Fjordbucht) | 397 m |

Nach oben wird es wieder schlechter: zu lange Laeufe lassen einem
seltenen Typ keine Luecke mehr, in die er passt.

**Ergebnis:** alle vier zuvor auffaelligen Archetypen treffen jetzt ihr
Profil - Fjordwand 129/128 m (vorher 23), Moher-Klippen 134/146 m (vorher
gar nicht vorhanden), Fjordbucht 44/35 m (vorher gar nicht vorhanden),
San-Sebastian-Bucht 15/16 m (vorher 5). `smoke_test_kuestenprofiltreue`
beide Gruppen gruen, Median 3.1 m, flache Kuesten 14 von 14.
`BEKANNTE_ABWEICHUNGEN` ist jetzt LEER - die San-Sebastian-Bucht (5
statt 16 m) hatte dieselbe Ursache; der dort notierte Verdacht
"Landmasse der Samarcia" war falsch.

**Was dabei SCHLECHTER wurde, ehrlich notiert:**
  * `smoke_test_regionen_fairness`: Skerrheim 0.72 → 0.64. Das ist
    inhaltlich richtig - es gibt jetzt echte Fjordwaende, und die sind
    steil (52 %). Der Kuestenbonus (1.08) gleicht das nicht aus. Eine
    KALIBRIERUNGSfrage, keine Fehlfunktion: der Zielwert 1.00 fuer eine
    Fjordlandschaft ist zu hoch, oder der Kuestenbonus zu schwach.
  * `smoke_test_regionen_welt` Naht 1.474 → 1.540 (Grenze 1.25). Der
    Test war vorher schon rot, mit denselben Befunden. Die Kuestenformen
    praegen jetzt staerker, und diese Metrik mischt Kuestenformen mit
    Regionsnaehten.

**Die Lehre, zum dritten Mal in diesem Projekt:** eine Kette aus lauter
einzeln richtigen Teilen kann als Ganzes falsch sein. Profil, Mischung,
Reichweite, Deckel und Staerke waren jeder fuer sich korrekt - der Fehler
sass in der Zuordnung DAZWISCHEN, und keine der Einzelpruefungen konnte
ihn sehen. Gefunden wurde er erst, als eine Messung Archetyp-Anteile
gegen Kuestenlaengen-Anteile hielt, also zwei Enden der Kette
gegeneinander. `tests/smoke_test_archetyp_verteilung.py` haelt das
jetzt fest.

### DER MESSAUFBAU WAR FALSCH — aufgeloest 2026-08-24

Alle Messungen dieser Runde, die `hoehe()` "richtige" Werte an den
Fjordwand-Stationen bescheinigten (144 m bei Soll 120), waren **an der
falschen Karte**. Sie bauten die `VektorKueste` so:

```python
vd.VEKTOR_KUESTE_AKTIV = False
H0, f = weltfeld(512, SEED)          # <-- laeuft den ALTEN Rasterpfad!
vk = VektorKueste(H0, ...)
```

Mit dem Schalter auf False laeuft `weltfeld()` durch
`_kuesten_umformen()` - den alten Pixelmasken-Pfad. Das Gelaende, auf dem
die Kueste dann gebaut wurde, ist ein anderes als im echten Lauf.

**Mit der `VektorKueste` aus dem ECHTEN `weltfeld()`** (sie liegt dort als
`felder["vektor_kueste"]`) liefert `hoehe()` bei d = 150 m nur **14 m**
statt der Soll-128 m. Der Test hatte die ganze Zeit recht; die
Fjordwand ist wirklich zu flach.

**Was daraus fuer kuenftige Messungen folgt:** immer
`felder["vektor_kueste"]` aus dem echten Lauf nehmen, nie eine selbst
gebaute. Dieselbe Lehre wie in smoke_test_kuesten_schnitt (dort steht sie
seit dem 2026-08-22 im Kopf) - sie hat sich hier unabhaengig wiederholt,
weil ich sie beim Messen nicht angewandt habe.

**Der Widerspruch ist damit aufgeloest, die URSACHE aber weiter offen.**
Bekannt ist jetzt: alle Einzelteile pruefen sich einzeln als richtig -
das Profil liefert bei d = 150 m die erwarteten 128 m, alle drei
Mischplaetze sind Fjordwand (keine Verduennung durch flache Nachbarn),
der Hoehendeckel greift nicht (d_max 2740 m), das Schliessen zur
Inselmitte kuerzt 150 m auf 146 m, und `staerke` ist nach dem Fix 1.00.
Trotzdem kommen 14 m heraus. Der naechste Schritt ist, `_hoehe_block()`
an einem einzelnen Fjordwand-Punkt Zeile fuer Zeile mitzurechnen, statt
weiter Hypothesen zu pruefen.

**Weiterhin offen** ist auch die Zuordnung in `archetyp_felder()`:
die Pixel, die dort als "Fjordwand" markiert sind, liegen im Median 335 m
von der Kueste (dort sagt das Profil rund 270 m) und haben H = 41 m. Die
Fjordbucht - der FLACHSTE Archetyp des Fjordlands - bekommt dagegen Pixel
mit H = 201 m. Entweder stimmt diese Zuordnung nicht mit der ueberein, die
`hoehe()` intern trifft, oder der Test misst ueber sie an den falschen
Stellen. **Das ist zu pruefen, bevor weiter am Modell gedreht wird.**

MISCH_EXPONENT = 4 wurde uebernommen (Median 2.2 -> 1.9 m), der
Skerrheim-Tiefenfaktor ebenfalls (Nutzerwunsch, schadet nicht).

## 4. Fluesse folgen jetzt dem Wasser, nicht der Knotenzahl

Nutzerbefund: im Skerrheim fehlen grosse Fluesse.

**Die Ursache stand in einer Zeile** (`terrain_weltfluesse.py:377`):
`flaeche = np.ones(n)`. Jeder Knoten trug 1 bei, egal ob dort 471 mm oder
1967 mm fallen. Was das Modell "Einzugsgebiet" nannte, war die FLAECHE in
Knoten, nicht die Wassermenge. Die Korrelation zwischen Wassermenge und
Flussgroesse war NEGATIV: das Skerrheim hatte die meiste Wassermenge
(13.0 Mio) und den kleinsten Hauptfluss (123), die Samarcia die wenigste
(4.3 Mio) und einen dreimal groesseren.

Umgesetzt: Niederschlag als Knotengewicht (Block 1.1) und die Regionsquote
nach Nutzervorgabe (Block 2, Skerrheim 100 %, Morobora und Atlantik je 66 %).
Skerrheim 123 -> 215 -> **700**.

**Drei Fehler beim Bau der Quote:** Doppelmultiplikation (2275 statt 700),
Zufluesse oberhalb mitskaliert (dort fliesst kein zusaetzliches Wasser),
multiplikativ statt additiv (blaehte die Estrande auf 2396 — mehr
Wasser im Oberlauf heisst flussabwaerts eine KONSTANTE Zugabe).

**Die Ursache des Skerrheim-Rueckstands bleibt offen.** Drei Hypothesen
wurden gemessen und widerlegt — zerstueckelte Landmasse (Skerrheim liegt
zu 100 % in einem Stueck), kurze Fliesswege (es hat mit 832 m den
ZWEITLAENGSTEN mittleren Kuestenabstand), zu wenig Buendelung (die Metrik
ist nicht schluessig). Sie stehen in `docs/FLUESSE_UND_WASSER.md`, damit
sie niemand erneut prueft.

**Block 3 (Seen als Sammler) ist fuer das Skerrheim wirkungslos:** die
Karte hat 11 Binnenseen ueber 4 Pixel, und keiner liegt dort.

Ordnung und weitere Bloecke: `docs/FLUESSE_UND_WASSER.md`.

## Offen am Ende dieser Sitzung

* **VISUELL NICHT BESTAETIGT.** Die gesamte Kuesten- und Flussarbeit ist
  headless gemessen. Niemand hat sie im laufenden Programm gesehen. Das
  ist der Blocker.
* `smoke_test_regionen_welt`: Naht-Kennzahl 1.474 (Grenze 1.25).
  **Nachgemessen 2026-08-24, A/B im selben Prozess:**

  | | Naht | Inneres | Verhaeltnis | Befunde |
  |---|---:|---:|---|---:|
  | `VEKTOR_KUESTE_AKTIV=False` | 1.247 | 1.175 | 1.06 ok | 10 |
  | `VEKTOR_KUESTE_AKTIV=True` | 1.474 | **0.621** | 2.37 FEHLER | **7** |

  Bemerkenswert: MIT Vektorkueste hat der Test INSGESAMT WENIGER Befunde
  (7 gegen 10) - die Regionseichung ist als Ganzes besser geworden. Nur
  die Nahtpruefung kippt.

  **Die Ursache ist eingegrenzt, aber nicht abschliessend geklaert.** Die
  Naht selbst aendert sich kaum (1.247 -> 1.474); das INNERE faellt von
  1.175 auf 0.621. Gemessen wirkt die Vektorkueste exakt in ihrer Zone
  und nirgends sonst:

  | Abstand von der Kueste | ohne VK | mit VK | Faktor |
  |---|---:|---:|---:|
  | 0-100 m | 0.581 | 0.176 | 0.30 |
  | 100-250 m | 0.324 | 0.092 | 0.28 |
  | 250-500 m | 0.322 | 0.285 | 0.89 |
  | **ab 500 m** | 0.202 | 0.202 | **1.00 (bitgleich)** |

  Der Hang in der Kuestenzone faellt, weil die gemessenen Profile die
  Wahrheit sind: die meisten Archetypen SIND flach (Kykladen 1 m,
  Toskana 4 m, Dingle 2 m nach 150 m), nur wenige sind Klippen. Das
  "Innere" des Tests ist als `max(gewichte) > 0.85` definiert und
  schliesst Kuestennaehe nicht aus - der Test wurde gebaut, als die
  Kuestenzone 110-349 m breit war, jetzt sind es 500 m.

  **NACHGEMESSEN mit der ECHTEN Testmetrik** (p99.5 der Steigung auf dem
  Gauss-geglaetteten Feld, dieselben Masken wie im Test):

  | | Naht p99.5 | Innen p99.5 | Verhaeltnis |
  |---|---:|---:|---:|
  | ohne Vektorkueste | 1.273 | 1.020 | **1.25** |
  | mit Vektorkueste | 1.500 | 0.713 | 2.11 |

  **DER TEST STAND SCHON VORHER EXAKT AUF DER GRENZE.** 1.25 bei einer
  Grenze von 1.25 - null Reserve. Er war nicht "gruen", sondern "gerade
  eben noch gruen", und jede Geländeaenderung kippt ihn.

  Beide Seiten bewegen sich: die Naht steigt um 18 %, das Innere faellt
  um 30 %. Erklaerbar ist beides:

  * **Das Innere faellt**, weil die extremsten Landhaenge oft
    KUESTENhaenge sind. `innen_band` schliesst nur das Nahtband aus,
    nicht die Kueste. Die gemessenen Profile sind ueberwiegend flach
    (Kykladen 1 m, Toskana 4 m nach 150 m), also sinkt das p99.5.
  * **Die Naht steigt**, weil die Vektorkueste manche Klippen STEILER
    macht als das alte Modell (Moher: 145 m ueber 150 m). Laeuft eine
    Regionsgrenze durch eine Klippenregion, wird sie steiler gemessen.

  Ein Gegencheck nur ueber 500 m von der Kueste entfernt zeigt denselben
  Anstieg der Naht (1.060 -> 1.304). Das liegt an der Gauss-Glaettung
  (sigma 120 m, Einfluss bis rund 360 m) - sie zieht Werte aus der
  Kuestenzone in die Auswertung hinein.

  **NICHT BEHOBEN, und die Grenze wurde NICHT angehoben.** Eine
  Testgrenze zu lockern, weil das Ergebnis nicht passt, macht den Test
  wertlos. Der eigentliche Befund ist, dass die Metrik Kuestenformen und
  Regionsnaehte vermischt: sie misst den steilsten Hang im
  "Regionsinneren" und trifft damit oft eine Klippe. Wer das aufloest,
  sollte die Kuestenzone aus BEIDEN Masken nehmen und die Glaettung
  entsprechend kuerzen - dann misst der Test wirklich Naehte.
* `smoke_test_settlement_roads`: rot, aber VORBESTEHEND — `bau_kostenfeld`
  wurde auf eine exponentielle Hangformel umgestellt, der Test prueft noch
  die alte quadratische Erwartung.
* `docs/KUESTENMODELL.md` beschreibt noch das alte Modell (Reichweite,
  normierte Formen) und ist nachzufuehren.

---

# 2026-08-24 — Wegoptik, dann Abarbeitung der 20 einfachsten offenen Punkte

**Ausgangslage:** HEAD `950a949` (2026-08-12), Arbeit laeuft uncommitted im
Hauptcheckout auf `main`. Kein Worktree.

## Teil 1: Wege besser auf dem Mesh darstellen

Nutzerauftrag: zehn Methoden vorschlagen, davon die sinnvollen umsetzen.
Umgesetzt wurden **Methoden 1-3** plus ein Nebenbefund.

### 1. `glPolygonOffset` statt Weltversatz
`gui/widgets/map_display_3d.py` `_render_wegbaender()`. Vorher hielt allein
`SCHWEBE_ANTEIL` das Band ueber dem Gelaende - ein Versatz in der WELT, bei
flachem Blickwinkel sieht man darunter. Jetzt `glPolygonOffset(-2.0, -4.0)`,
das nur im Tiefenpuffer wirkt. `SCHWEBE_ANTEIL` 0.0006 -> 0.0001 als
Sicherheitsnetz.

### 2. Weicher Rand statt Polygonkante
Neues Vertexattribut `deckung` (Vertexformat 7 statt 6 float), 1 auf der
Fahrbahn, 0 an den Kanten. `wegband.frag` multipliziert es in die Alpha
(mit `sqrt`, sonst wirkt der deckende Teil zu schmal) und verwirft Fragmente
nahe 0. Die lineare Interpolation zwischen Schulter und Kante erzeugt den
Verlauf von allein.

### 3. Querprofil statt flachem Rechteck
`gui/widgets/wege_geometrie.py`: fuenf Bahnen je Wegpunkt
(Kante/Schulter/Scheitel/Schulter/Kante), Woelbung `h(t) = w * (1 - t^2)`.
Geometrische Woelbung klein (`WOELBUNG_ANTEIL = 0.02`, gemessen 2.1 m), die
NORMALEN aber um `NORMALEN_WOELBUNG = 0.40` verkippt - absichtlich viel
staerker, damit die Schattierung die Woelbung zeigt und der Umriss sie nur
andeutet.

### Nebenbefund: GL-Puffer wurden JEDEN FRAME neu angelegt
`_render_wegbaender()` legte VAO/VBO/EBO pro Frame an, lud die kompletten
Baender hoch (gemessen 1.1 MB bei 40 Wegen) und loeschte alles wieder. Kein
Leck, aber der Python-Cache darueber sparte nur das Rechnen, nicht die
Uebertragung. Jetzt haengen die Puffer am Cache;
`_wegband_puffer_freigeben()` raeumt beim Geometriewechsel und in
`_cleanup_mesh_buffers()`.

### Zwei Messbefunde beim Umbau
* **Hoehe stuetzte sich auf drei Querstellen** (Mitte, linker, rechter Rand) -
  ein Grat DAZWISCHEN wurde uebersehen. Jetzt `QUER_STUETZSTELLEN = 9`.
* **Die Laengsglaettung schneidet an Kuppen ein**, gemessen 0.7 m bei 256 px.
  Kein Fehler - das tut eine echte Trasse auch, und mit dem Tiefenversatz ist
  es unsichtbar.

### GELOCKERTE ZUSICHERUNG - bitte nachvollziehen
`tests/smoke_test_wege_geometrie.py`: die Bedingung "kein Vertex unter dem
Gelaende" wurde zu "hoechstens `EINSCHNITT_MAX_M = 2.0` m Einschnitt, im
Median darueber" **gelockert**. Begruendung steht im Test. Wer das anders
sieht, muss die Laengsglaettung abschalten oder den Weltversatz
zurueckholen.

### Auch behoben: stiller Rueckfall
`_render_wegbaender()` stieg ohne `wegband_shader_program` kommentarlos aus -
"Shader liess sich nicht uebersetzen" war von "es gibt keine Wege" nicht zu
unterscheiden. Meldet jetzt einmal laut.

**Tests:** `smoke_test_wege_geometrie.py` 6/6 gruen (drei Zusicherungen an das
neue Profil angepasst), `smoke_test_shader_paths.py` um eine vierte
Zusicherung erweitert (Varyings vert<->frag, Attributplaetze lueckenlos).
Deren erste Fassung war selbst loechrig: `dict()` auf (typ, name)-Paare macht
den TYP zum Schluessel, `FragPos` verschwand hinter `Normal`.

**NICHT bestaetigt:** wie es aussieht. OpenGL laesst sich headless nicht
pruefen.

### Nachbesserung 2026-08-24 nach Sichtprüfung (Nutzerbefund)

Nutzer: *"sieht besser aus. aber verschwindet noch immer bei vielen
kamera-bewegungs-aktionen. dann kommt es auf distanz zb auch mal vor, dass
der weg nicht komplett dargestellt wird sondern so gestueckelt sein kann ...
wege sind sehr windy ... nur 60% von der dicke ... nicht markierbar."*

**Verschwinden und Stueckeln war NICHT die Wegdarstellung, sondern die
Projektion.** In `_update_projection_matrix()` standen fest `near=0.1,
far=2000.0` mit dem Kommentar, das Verhaeltnis sei "fuer einen 24-Bit-
Tiefenpuffer unkritisch". Das ist falsch - die Genauigkeit haengt fast allein
an der NEAR-Plane. Nachgerechnet (Welt 10 Einheiten breit, 24 Bit):

| Kameraabstand | Tiefenaufloesung |
|---:|---:|
| 17 (Vorgabe) | 0.000172 Welteinheiten |
| 30 | 0.000536 |
| 60 | 0.002146 |

Das Wegband schwebt 0.001 Einheiten. **Ab Kameraabstand ~45 ist die
Tiefenaufloesung groesser als der Abstand** - genau das Symptom. Jetzt wandern
Near und Far mit (`near = Abstand/20`, `far = Abstand + 200`), das ist 8.6x
bis 30.8x genauer. Polygon-Offset zusaetzlich auf (-3, -6).

**Wiggle:** das A*-Routing laeuft auf einer 8er-Nachbarschaft, eine schraege
Strecke wird dort zur Treppe mit 45-Grad-Wechseln. Bisher wurde nur die HOEHE
geglaettet, nicht der Verlauf. Neu `GLAETTUNG_XY_FENSTER = 7` mit zwei
Durchgaengen (zwei kleine statt eines grossen Fensters - gleiche Glaette bei
halbem Versatz), Endpunkte fest. Gemessen: Gesamtdrehung **4410 -> 83 Grad
(53x ruhiger)**, groesster Versatz 1.0 px, Endpunkte exakt getroffen.

**Breite auf 60 %:** 35/26 m -> 21/16 m, `MINDEST_BREITE_PX` 2.5 -> 1.5. Bei
512 px sind das 62 m statt 104 m.

**Markierbar:** die Auswahl funktionierte bereits (Klick -> Treffer -> Text im
Seitenfeld), aber **im Bild passierte nichts** - der Shader kann die
Einfaerbung seit dem 2026-08-16, sie wurde nur nie eingeschaltet. Jetzt gibt
`treffer_suchen()` den `index` mit zurueck, das Display merkt sich den
gewaehlten Weg und zeichnet dessen Indexbereich ein zweites Mal mit
`ausgewaehlt = 1`. Zweiter Draw-Call statt Vertexfarbe: die Auswahl aendert
sich pro Klick, die Geometrie nicht.

**Woelbung:** der Nutzer sieht sie nicht und haelt sie bei dieser
Darstellungsart fuer verzichtbar. Bleibt drin, kostet nichts.

---

## Teil 1b: Kuestenprofile — gibt es die, und prueft sie jemand?

Nutzerfrage: *"sag mal ob du zugriff auf die ganzen kuestenprofile hast die
es haben soll und ob es einen test gibt ob die kuesten so aussehen wie die
profile?"*

**Die Profile gibt es**, in `core/vektor_kueste.py`: `MESS_FORM_JE_ARCHETYP`
(27 Archetypen, je 17 Stuetzstellen), `MESSWERTE_JE_REGION` (Klippenhoehe
p10/p90) und `MESS_REICHWEITE_M` - alle aus echten DEMs gewonnen.

**Geprueft hat sie niemand.** `smoke_test_vektor_kueste.py` prueft sechs
Dinge, und alle sechs sind STRUKTURELL (eine Funktion, Rasterfreiheit,
Aufloesung, Determinismus, Lage der 0-Linie, Randfaelle). Keines vergleicht
die entstandene FORM mit der Vorlage.

**Neu: `tests/smoke_test_kuestenprofiltreue.py` (2/2 gruen).** Misst mit
denselben Funktionen, mit denen die Tabellen entstanden sind
(`tools/kuestenlaengsschnitt.py`), gruppiert nach (Region, Archetyp).

Ergebnis ueber 20 Gruppen:

* **Form: Median-RMS 0.121** auf der 0..1-Skala. Klippenarchetypen treffen gut
  (Kola 0.028, Kotor 0.031, Foerdenkueste 0.040, Santorini 0.063, Amalfi
  0.063). **Straende und Flachkuesten treffen schlecht** (Toskana-Straende
  0.374, Weissmeer-Flachkueste 0.320, Dingle 0.318, Kykladen-Strand 0.317) -
  eine markante Klippe setzt sich gegen das Rauschgelaende durch, ein Strand
  geht darin unter.
* **Hoehe: vier Ausreisser**, namentlich in `BEKANNTE_HOEHENABWEICHUNGEN`
  gefuehrt. **Ursache ist ein Modellfehler in der TABELLE, kein Rechenfehler:**
  `MESSWERTE_JE_REGION` ist je REGION tabelliert und stammt aus je EINER
  Vorbildkueste, waehrend jede Region DREI Archetypen unterschiedlichen
  Charakters hat. Die Morobora-Hoehen kommen von den Stockholmer Schaeren
  (15-30 m), ihre Archetypen heissen aber "Kola-Steilkueste" und
  "Weissmeer-Flachkueste" - eine Steilkueste kann das Schaerenband gar nicht
  einhalten. **Der saubere Weg waere, die Hoehen je ARCHETYP zu messen statt
  je Region.**

Eigener Messfehler dabei gefunden und behoben: die erste Fassung nahm die
Klippenhoehe als Maximum ueber die GANZE Landseite - damit misst man das
Hinterland mit, und alle Regionen sahen zu hoch aus. Jetzt nur innerhalb der
Profilreichweite, gegen die Uferhoehe gerechnet.

---

## Teil 2: Die 20 einfachsten offenen Punkte

### [x] 12.3 + 12.4 — Doppelte und unerreichbare Parameter
`erosion_strength` steht in class EROSION (0.0-2.0, Vorgabe 0.5) UND class
WATER (0.1-5.0, Vorgabe 2.5). **Nicht geloescht** - `core/water_generator.py`
liest die Schluessel weiter ueber `parameters.get(...)`, ein Loeschen wuerde
nur die dokumentierte Spanne entfernen und den stillen Rueckfall auf die
Literalwerte im Generator hinterlassen.

Stattdessen: neues Register `DOPPELTE_SCHLUESSEL` in
`gui/config/value_default.py`, und die stillgelegten Droplet-Regler
(`erosion_passes`, `sediment_capacity_factor`, `settling_velocity`,
`thermal_erosion_strength`, `evaporation_base_rate`, `diffusion_radius`)
tragen jetzt eine Begruendung in `stillgelegte_regler()`.

**Neuer Test `tests/smoke_test_parameter_eindeutig.py` (4/4 gruen)** - er hat
sofort einen ZWEITEN Doppelschluessel gefunden, von dem in 12.3 nichts stand:
`octaves`. Nachgeprueft: harmlos, nur der Attributname ist gleich
(`TERRAIN.OCTAVES` laeuft als `octaves`, `EROSION_FILTER.OCTAVES` als
`erosion_filter_octaves`). Steht mit dieser Erklaerung im Register.

**Eigener Messfehler dabei:** die erste Testfassung verglich Attributnamen
statt Parameterschluessel und meldete `octaves` faelschlich als Kollision.
Der Test prueft jetzt beides getrennt.

### [x] 12.2 — Erosionsreiter kennzeichnen
`gui/tabs/erosion_tab.py`: `_create_stilllegungs_hinweis()` zeigt einen
gelben Hinweisstreifen, solange `EROSION_AKTIV` False ist. **An den Schalter
gekoppelt, nicht fest verdrahtet** - wird er True, verschwindet der Streifen
von selbst.

### [x] 6.3 — WAR SCHON ERLEDIGT
Der Wetter-Reiter hat bereits zwei getrennte Zeilen (Messgroesse /
Atmosphaere-Schicht), der Docstring von `create_visualization_controls()`
nennt 6.3 ausdruecklich. Nur der Haken in der Liste fehlte.

### [x] 6.26 — Umschalten von Settlements/Roads
**Gemessen:** `update_display(heightmap)` bei 512 px = 150.3 ms, davon
`canvas.draw()` allein 90.6 ms. Beim Umschalten einer Checkbox wird ZWEIMAL
gezeichnet (Basiskarte, dann Overlays).

**Fix:** alle 17 `self.canvas.draw()` in `gui/widgets/map_display_2d.py` auf
`draw_idle()` umgestellt. Qt fasst mehrere Anfragen zu einem Durchgang
zusammen. **150.3 ms -> 42.2 ms**, alle 30 Darstellungen
(`smoke_test_display_2d.py`) weiter gruen.

WICHTIG: die Modulfunktionen `rasterize_*` zeichnen auf EIGENE
Offscreen-Canvases und lesen direkt danach `buffer_rgba()` - die brauchen
`draw()` synchron und wurden NICHT umgestellt.

### [x] 6.22 — Monats-Wetterkarten
`_on_month_cycle_tick()` prueft jetzt `viewport_widget.isVisible()` und
zeichnet nur, wenn der Reiter vorn ist. Der Monatsindex laeuft dabei bewusst
nicht weiter, damit die Anzeige beim Zurueckwechseln nicht springt. Gleiche
Ueberlegung wie 6.14.

### [x] 3b.9 — Rechenzeit der Vektor-Kueste, jetzt gemessen

| px | Vektor | Raster | Faktor |
|---|---:|---:|---:|
| 256 | 1.41 s | 0.65 s | 2.18x |
| 384 | 2.13 s | 1.02 s | 2.09x |
| 512 | 3.19 s | 1.61 s | 1.98x |
| **1024** | **10.67 s** | **5.39 s** | **1.98x** |

**Antwort: der Faktor bleibt konstant bei ~2x**, er explodiert nicht mit der
Aufloesung. (Die Doku nannte fuer 384 px 3.26/1.25 s - andere Maschinenlast
oder aelterer Stand; das Verhaeltnis stimmt ueberein.)

### [x] 13.1 — Export auf feste Weltgroesse
`gui/utils/map_export.py`: `EXPORT_KANTENLAENGE_PX = 2048`, jeder Layer wird
beim Export darauf gebracht. **Kategorien und Farbbilder per naechstem
Nachbarn**, skalare Felder bilinear - zwischen Biom 3 und Biom 7 liegt kein
Biom 5. Manifest fuehrt jetzt `export_kantenlaenge_px`, `welt_km`,
`meter_pro_pixel` und je Layer einen `groesse`-Vermerk ("nativ" /
"bilinear 512->2048" / "naechster Nachbar ...").

### [x] 13.2 — Daempfungsmaske fuers Engine-Rauschen
`daempfungsmaske()` in `map_export.py`, exportiert als
`noise_damping_mask.png`. 0 = nicht rauschen, 1 = volle Wildnis. Quellen:
`city_mask`, `street_mask`, `house_parcel_map`, `roads`/`sea_roads` und ein
Uferstreifen um die Wasserlinie. Radien in METERN
(`WEG_SCHUTZ_M = 45`, `UFER_SCHUTZ_M = 35`, `BAU_SCHUTZ_M = 25`,
`UEBERGANG_M = 60`), nicht in Pixeln - sonst haengt die Korridorbreite an der
Exportaufloesung.

**Neuer Test `tests/smoke_test_export_2048.py` (3/3 gruen).**

### [ ] 3b.8 — BEWUSST NICHT UMGESETZT
Auftrag war, `_kuesten_umformen()` (~200 Zeilen) als toten Code zu loeschen.
**Ist kein toter Code:** es ist der `else`-Zweig von `VEKTOR_KUESTE_AKTIV`,
und der Schalter existiert laut Kommentar in `core/terrain_weltkarte.py`
genau deshalb, weil die Regionseichung (3.9, noch offen) daran haengt. Den
Vergleichsmassstab zu loeschen, solange 3.9 ungeklaert ist, waere falsch.
**Erst 3.9 klaeren, dann loeschen.**

### [x] 13.5 — Vektordaten exportieren
`vektordaten()` in `map_export.py`, geschrieben als `vektor.json`. Enthaelt
Wege, Seewege, Grundstuecksgrenzen und Ortslagen (mit Typ, Rang, Kultur,
Haeuserzahl, Radius). **Koordinaten in METERN, nicht in Pixeln** - der
Pixelwert haengt an der Kartengroesse, Meter haengen an der Welt.

**Fluesse fehlen, mit Begruendung im Code und im JSON-Feld `fehlt`:**
`core/terrain_weltfluesse.flussnetz()` baut sehr wohl einen Knotengraphen,
aber `core/terrain_generator.py:1707` behaelt daraus nur die Raster
`river_mask`/`river_order` und wirft den Graphen weg. Aus dem Raster wieder
Linienzuege zu machen waere Arbeit mit eigenen Fehlerquellen fuer etwas, das
vorher schon vorlag. **Der richtige naechste Schritt ist, den Graphen in den
Ausgaben zu behalten** - ein Eingriff in den Terrain-Generator, nicht in den
Exporteur.

### [x] 6.19 — Grenze der Heightmap dort dokumentiert, wo man sucht
Der Punkt war ausfuehrlich in `OFFENE_PUNKTE.md` beschrieben, aber nicht im
Code. Jetzt steht im Kopf von `gui/widgets/adaptive_terrain_mesh.py`, warum
dieses Modul die 90-Grad-Kueste PRINZIPIELL nicht beheben kann (eine
Heightmap speichert je (x,y) genau einen Wert) und wohin man stattdessen
schaut: `terrain_remesh.py` (hilft ueber die Flaeche, nicht an der
Kuestenlinie) bzw. Zwangskante/Vektorweg.

### [ ] 7.4 — NICHT UMSETZBAR, Angaben fehlen
"Vier Schwellwertkipper aus float32 als Toleranz fuehren". **Welche vier,
steht nirgends** - weder im Eintrag (eine Zeile ohne Details) noch im Code
(`grep` nach "Kipper"/"kipp" findet ausserhalb dieser Zeile nichts, und keiner
der Paritaetstests fuehrt eine solche Liste). Ohne die Angabe waere jede
Toleranz geraten. **Braucht die Messung, aus der die vier stammen.**

---

## Testlage nach dieser Sitzung

Alle acht beruehrten Testdateien gruen:
`smoke_test_wege_geometrie`, `smoke_test_shader_paths`,
`smoke_test_parameter_eindeutig` (neu), `smoke_test_export_2048` (neu),
`smoke_test_display_2d`, `smoke_test_display_methoden_existieren`,
`smoke_test_terrain_remesh`, `smoke_test_adaptive_mesh_vectorized`.

---

## Noch offen aus den 20

**Nicht angefasst, weil deutlich groesser als "einfach":**

* **8.1** Logzeilen/Fortschrittstexte von `lod` befreien (~373 Stellen) -
  mechanisch, aber jede Stelle einzeln zu pruefen.
* **7.9** Geology ohne GPU-Anbindung - ein ganzer Generator-Port, kein
  Kleinkram.
* **9.2** Neun Niederschlagskappungen gegen Modellgrenzen messen - braucht
  Messlaeufe je Regler.
* **12.5** Orchestrator-Rundenbetrieb testen - der Eintrag sagt selbst, dass
  er mit 8.2 wegfaellt.
* **6.4/6.13** 3D-Ansicht testen - OpenGL ist headless nicht pruefbar, ein
  Test kann nur Struktur pruefen, nicht das Bild.
* **3b.3** Mesh-Schnitt auf Vektorhoehen - haengt an 3b.1 (visuelle
  Bestaetigung), sonst optimiert man ins Blaue.
* **1.8** Klimatabelle gegenpruefen - Recherche an externen Quellen.
* **7.4** siehe oben, Angaben fehlen.
* **3b.8** siehe oben, waere falsch solange 3.9 offen ist.

## Was der Nutzer pruefen muss

Steht in `docs/PRUEFLISTE_LIVE.md`.
