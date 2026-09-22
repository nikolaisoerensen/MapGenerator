# Alle offenen Punkte — gepflegt, Stand 2026-08-12

**Diese Datei ist die einzige Aufgabenliste des Projekts.** Sie hat am
2026-08-12 `docs/TODO.md` vollstaendig aufgenommen (jene Datei ist damit
geloescht) und fuehrt ausserdem zusammen, was aus `docs/TESTBERICHT.md`,
`docs/SIEDLUNGEN_ENTWURF.md`, `docs/BIOME_MATRIX.md`, `docs/KLIMA_UND_SEE.md`
und `docs/archiv/2026-08-04_INTEGRATIONSPLAN.md` an Aufgaben stammte.

    [x]  erledigt und gemessen
    [ ]  offen
    [~]  teilweise / mit Vorbehalt — die Einschraenkung steht jeweils im Text

Aufwand in halben Tagen. **Diese Datei wird bei jedem Fortschritt gepflegt.**


## Bilanz

| | Punkte |
|---|---:|
| **[x] erledigt und gemessen** | **83** |
| [~] teilweise, Einschraenkung benannt | 21 |
| [ ] offen | 43 |
| **Summe** | **147** |

*(Zuletzt am 2026-08-16 MASCHINELL nachgezaehlt, nicht geschaetzt - Zaehlung
per Grep ueber die `[x]`/`[~]`/`[ ]`-Marker, nicht die Vorgaenger-Zahl von
2026-08-13 fortgeschrieben. Die Summe ist gegenueber dem 2026-08-13-Stand
(137) gewachsen, weil seither echte neue Punkte dazukamen (Siedlungs-Umbau
5.16-5.22, Remesh-Recherche 6.30-6.33, Mesh-Werkstatt-Sitzung), nicht weil
neu gezaehlt wurde. Am 2026-08-16 die komplette Sektion 10 (Altlasten,
10.1-10.5) geschlossen - zwei davon (10.1, 10.4) waren bei Pruefung KEIN
Bug, sondern bereits korrekt und dokumentiert gehandhabtes Verhalten, nur
der Backlog-Eintrag war veraltet.)*

Die `[~]`-Punkte sind der eigentliche Ruecklauf: dort ist gebaut und gemessen,
aber etwas Benanntes fehlt noch. **Die Sichtpruefung am 2026-08-13 hat gezeigt,
dass sich das lohnt** — sie schloss 3.8, 3.10, 3.11, 3.13 und 6.18 in einem
einzigen Durchgang. Was jetzt noch auf einen Bildschirmtermin wartet: 5.10,
6.1, 6.16, 6.21, 6.23.

Dass die Zahl der offenen Punkte trotzdem gestiegen ist, ist kein Rueckschritt:
die Sichtpruefung hat sechs neue benannt (3.14, 5.15, 6.24-6.26, 9.3), die
vorher unsichtbar waren — darunter mit 5.15 (Siedlungen Regional zeigt nichts)
ein echter Fehler, den kein headless-Test gefunden haette.


## OFFEN und spezifiziert: Overlays bekommen eine Naht (2026-09-14)

**Spezifikation: [`docs/spezifikation/15_ANZEIGE.md`](spezifikation/15_ANZEIGE.md)** — dort steht
Problem, Loesung, Nutzergeschichten, Bau- und Pruefentscheidungen
vollstaendig. Hier nur der Anlass und der eine Punkt, der SOFORT wirkt.

**Ein lebender Fehler, heute im Arbeitsverzeichnis.**
`BiomeTab.apply_overlays()` steigt in der ersten Zeile aus, wenn die Ansicht
nicht 2D ist (`gui/tabs/biome_tab.py:507`). Die 3D-Zweige darunter — am
2026-08-25 ausdruecklich als Behebung eingebaut, mit 25 Zeilen Begruendung —
**koennen nie ausgefuehrt werden.** Betroffen sind Siedlungen UND
Flussgenerationen, letztere obwohl die Methode auf beiden Anzeigen existiert.

Das ist **Vorfall 4** derselben Fehlerklasse (nach 2026-08-24
`overlay_river_generations`, 2026-08-25 `overlay_river_network`, 2026-08-25
`overlay_settlements`) und der dritte, der als behoben verbucht wurde, ohne
es zu sein.

`tests/smoke_test_display_methoden_existieren.py` meldet trotzdem gruen —
seine Ausnahme fuer `overlay_settlements` **begruendet sich mit genau der
unerreichbaren Zeile.** Der Test ist gruen wegen des toten Codes. Er prueft
Namensexistenz, nicht Erreichbarkeit; diese Luecke ist die eigentliche
Ursache dafuer, dass die Fehlerklasse viermal durchkam.

| | # | Sache | Aufwand | Issue |
|---|---|---|---|---|
| [x] | 14.1 | **Der vorzeitige Ausstieg in `biome_tab.py:507`.** Einzeiler, wirkt sofort, bringt Siedlungen und Flussnetz im 3D zurueck. Kann VOR dem Umbau passieren; der Kommentarblock darueber ist dabei richtigzustellen, er beschreibt eine Behebung, die nie gewirkt hat. | 0.5 | #5 |
| [x] | 14.2 | **Waechtertest auf Erreichbarkeit umstellen.** Heute prueft er, OB eine Anzeigemethode existiert. Er muss pruefen, ob der Zweig, der sie ruft, ueberhaupt laufen kann. Ohne diesen Punkt wiederholt sich der Fall — 14.1 allein verhindert nur den heutigen. | 1 | #7 |
| [x] | 14.3 | **`_push_overlays()` in `BaseMapTab`** nach dem Vorbild von `_push_data_to_current_display()`, dazu das Register beider Wege je Overlay. Bestehende Naht verbreitern, keine neue (Nutzerentscheidung 2026-09-14). | 3 | #8 |
| [x] | 14.4 | **Reiter umstellen, einzeln, mit Sichtpruefung dazwischen** — Biome zuerst (dort sitzt der Fehler, beide Overlays haben ihren Rasterweg), dann Siedlungen, Regional, Fluss. Regional ist der unangenehmste: dort verschluckt ein `except Exception: logger.debug` um den ganzen Block jeden Fehler, und alle fuenf Weichen greifen in 3D nicht. | 3 | #9 #10 #11 |
| [x] | 14.5 | **`gui/tabs/overview_tab.py` ist vollstaendig tot** *(Nebenbefund)*. Der Reiter ruft vier Methoden auf, die es nirgends im Programm gibt. Kein Teil der Overlay-Arbeit, aber ein Waechtertest nach 14.2 wird darueber stolpern — also vorher entscheiden: loeschen oder bauen. | 1 | #6 |
| [x] | 14.6 | **Rasterfunktionen in ein eigenes Modul** *(Vorarbeit, streichbar)*. Die fuenf `rasterize_*_rgba()` liegen heute in der 2D-Anzeige, und drei Reiter importieren sie von dort **im 3D-Pfad** - die 3D-Ansicht haengt am 2D-Modul, obwohl sie es sonst nicht braucht. Aendert kein Verhalten; macht 14.3 ordentlicher, weil das Register dann auf ein neutrales Modul zeigt. | 0.5 | #4 |
| [~] | 14.7 | **Aufraeumen nach der Umstellung.** Reste der Weichenlogik weg (Ausgangslage 43 `hasattr` in `gui/`), die drei Namensraeume fuer denselben Layer ins Register, und die heute registerlosen 3D-Layernamen (`"uebersicht"`, `"wegbaender"`) dazu - ein Tippfehler dort erzeugt kein `KeyError`, sondern ein Nichts. Danach zeigt jede Waechtertest-Begruendung aufs Register statt auf eine Reiterzeile. **Teilweise (Nachtlauf 2026-09-16, #12, Commit e5b78f1):** der Tippfehler-Teil ist behoben — `MapDisplay3D.set_layer_visibility()`/`update_overlay_data()` melden einen unbekannten `layer_name` jetzt laut ueber `rendering_error` statt lautlos entweder gar nichts zu tun oder einen nie gelesenen neuen Dict-Schluessel anzulegen; alle bestehenden Aufrufstellen wurden dabei gegen `overlay_data`/`layer_visibility` geprueft, keine Abweichung gefunden. **Offen geblieben, bewusst:** die drei Namensraeume (`_LAYER_NAME_MAP_3D`/`_LAYER_SELECTION_KEYS_3D` in `base_tab.py`, `overlay_data`/`layer_visibility` in `map_display_3d.py`) tatsaechlich zu EINEM Register zusammenzufalten - das ist ein groesserer Umbau an einer Stelle, die in praktisch jeden Tab-Push verdrahtet ist, und wurde als zu riskant fuer einen unbeaufsichtigten Nachtlauf eingestuft (CLAUDE.md/Erfahrung: vor grossen riskanten Umbauten nachfragen). Empfehlung: eigenes Ticket fuer einen Tagdurchlauf mit Live-Pruefung. | 2 | #12 |
**Die stehende Regel und warum sie nicht reicht.** `CLAUDE.md` haelt seit dem
2026-08-25 fest, dass jede 2D-Anzeige in derselben Aenderung auch in 3D
gebaut wird. Die Regel ist richtig und wurde trotzdem gebrochen — weil sie
eine Gedaechtnisleistung verlangt, wo 43 `hasattr`-Weichen in `gui/` jedem
Reiter erlauben, sie stillschweigend zu verfehlen. Der Umbau macht aus der
Regel eine Eigenschaft eines Moduls.


## Kuesten-Kalibrierung nach dem Verteilungsfix (2026-08-24)

Der Verteilungsfix (`SAAT_KOHAERENZ_STATIONEN`, siehe SITZUNGSLOG) hat
alle 27 Archetypen auf die Karte gebracht. Zwei Kennzahlen sind dadurch
schlechter geworden - beide sind KALIBRIERUNGSfragen, keine Fehler:

* [x] **ERLEDIGT 2026-08-24.** Skerrheim-Fairness 0.72 -> 0.64 -> **0.80** (`smoke_test_regionen_fairness`,
  Ziel 1.00). Es gibt jetzt echte Fjordwaende, und die sind steil - 52 %
  der Region. Der Kuestenbonus steht bei 1.08 und gleicht das nicht aus.
  **Zu entscheiden:** entweder der Zielwert fuer eine Fjordlandschaft ist
  zu hoch (der Nutzer hat fuer das Nevadin selbst 0.80 gesetzt - eine
  Fjordlandschaft ist strukturell aehnlich), oder der Kuestenbonus ist zu
  schwach. Fjorde haben real enorm viel Kuestenlinie je Flaeche; ein
  Bonus von nur 8 % bildet das nicht ab. **Das ist eine Nutzerentscheidung
  ueber das Zielbild, keine Rechenfrage.**

* [ ] **Naht 1.474 -> 1.540** (`smoke_test_regionen_welt`, Grenze 1.25).
  Der Test war vorher schon rot, mit denselben Befunden, und lag vor
  dieser Sitzung bei exakt 1.25 - also ohne jede Reserve. Die
  Kuestenformen praegen jetzt staerker. **Die Metrik selbst ist der
  Verdaechtige:** sie mischt Kuestenformen mit Regionsnaehten und kann
  deshalb nicht sagen, welches von beidem zu steil ist. Vor einer
  Aenderung am Gelaende waere zu klaeren, ob die Naht ueberhaupt das
  misst, was sie messen soll.

* [ ] **Algarve-Klippen** fallen bei 8 Saatstationen gelegentlich ganz
  aus. Unterhalb der Testschwelle (10) und damit geduldet - aber wenn ein
  Archetyp so selten gesaet wird, ist die Frage, ob sein `max_anteil` in
  der Samarcia richtig steht.

* [ ] **`_segmente_schliessen()` frisst kurze Archetypen — der Rest des
  "8 von 27 fehlen"-Befunds (gemessen 2026-08-25).** Segmente unter
  `MIN_SEGMENT_M` (750 m) werden in den laengeren Nachbarn eingeschmolzen.
  Ein Archetyp mit vielen einzeln verstreuten Stationen verliert dadurch
  seine ganze Kuestenlaenge: **Schaerenkueste bekommt 16 Stationen
  zugeteilt und behaelt 0.0 % Laenge**, Algarve-Klippen 8 Stationen und
  ebenfalls 0.0 %. Das gilt unabhaengig davon, wie die Quote gerechnet
  wird - beide Zaehlweisen wurden gemessen. **Hier, nicht an der Quote,
  muesste eine Korrektur ansetzen.**

* [ ] **Kola-Steilkueste schiesst auf 5.53x ueber** (`smoke_test_archetyp_verteilung`,
  Rahmen 0.30x-5.0x), und die Morobora steht in `smoke_test_regionen_welt` bei
  Hang 10.4 statt 7.5. Beides seit der Umstellung der Quote auf
  Kuestenlaenge am 2026-08-25 - der Preis dafuer, dass die flachen
  Archetypen jetzt ihr Profil treffen. Die Morobora hat nur ~33 Stationen in
  der ganzen Region, da schlaegt jede Umverteilung durch. **Der Handel ist
  im Code an `ziel_meter` in `_saat_setzen()` vollstaendig dokumentiert,
  samt der gemessenen Gegenprobe - wer ihn anders gewichtet, findet dort
  beide Fassungen.**


## Was zuerst — Stand 2026-08-12

Die frueher hier stehende Reihenfolge ist abgearbeitet und wurde ersetzt.

*Am 2026-08-13 nach der Sichtpruefung neu geordnet. Die Sichtpruefung selbst
ist abgearbeitet (sie schloss fuenf Punkte und foerderte sechs neue zutage) -
der Nutzer hat dabei die Reihenfolge fuer das Weitere selbst vorgegeben:
Siedlungen komplett durchgehen, danach Erosion, danach Wasser.*

| | Was | Warum jetzt | Aufwand |
|---|---|---|---|
| **1** | **5.15 — Siedlungen Regional zeigt gar nichts an** | **Echter Fehler, vom Nutzer zweimal gemeldet.** Konkreter Verdacht steht im Eintrag (stiller `None`-Rueckfall bei Groessenunterschied `region_map`/`heightmap`) - erst pruefen, dann umbauen. Danach der neue Zuschnitt nach Land-Bounding-Box. | 4 |
| **1b** | **Siedlungen insgesamt durchgehen** | Nutzer-Vorgabe 2026-08-13: *"Settlements muessen wir bald komplett neu durchgehen. alles reparieren."* 5.15 ist der Anlass, aber vermutlich nicht der einzige Fund. | 4 |
| 3 | **Erosion, DANN Wasser** (Abschnitt 4, dann 9.1/9.3/12.6) | Reihenfolge vom Nutzer vorgegeben (siehe 9.3): die Wasserwege entstehen auf der erodierten Form, andersherum waere die Arbeit doppelt. | 5 |
| 4 | 1.5 / 1.6 Atmosphaere | Letzter grosser Rechenposten. Braucht eine eigene Sitzung (siehe Text). | 3 |
| 5 | 10.x Altlasten | Kleinteilig, kein Risiko, macht den Kopf frei. | 3 |


## Aufraeum- und Effizienzfunde aus dem Code-Review 2026-09-22 (2026-09-22)

Herkunft: Code-Review des Nachtbranches `nacht/2026-09-19` gegen `main` (16
Commits, 101 Dateien, ~7000 Zeilen) beim Abschluss dieses Branches. Acht
Pruefwinkel liefen parallel; von zehn Funden war einer ein echter, wenn auch
noch folgenloser Fehler und wurde direkt behoben (`gui/tabs/overview_tab.py`,
`welt_laden()`, Commit `2ed2bd6` — siehe Morgenbericht). Die folgenden neun
sind Aufraeum- und Effizienzhinweise, keiner davon ein akuter Fehler, und
wurden bewusst NICHT in der Nachtsitzung selbst behoben (zu gross fuer
"klein und sicher" in einem unbeaufsichtigten Lauf). Kein Issue existiert
bisher zu diesen Punkten.

| | # | Sache | Aufwand | Issue |
|---|---|---|---|---|
| [x] | 15.1 | **`core/fluss_export.py`: Gaettungs- und Breitenformel doppelt geschrieben.** Zwei Codestellen berechnen dieselbe Fluss-Glaettung bzw. dieselbe Breitenformel unabhaengig voneinander statt ueber einen gemeinsamen Helfer. Aendert sich eine Formel, muss man beide Stellen finden und synchron halten — sonst laufen sie unbemerkt auseinander. Erledigt 2026-09-23 (Commit `d0344b1`): gemeinsamer Helfer `gebiet_je_knoten()`/`talbreite_anteil()` in `core/terrain_weltfluesse.py`. | 0.5 | #86 |
| [x] | 15.2 | **`core/settlement_generator.py`: `BIOME_SIEDLUNGSEIGNUNG` von Hand abgeschrieben.** Die Tabelle dupliziert Werte, die eigentlich aus der Biom-Matrix stammen, statt von dort abgeleitet zu werden. Aendert sich die Biom-Matrix, faellt `BIOME_SIEDLUNGSEIGNUNG` stillschweigend zurueck und zeigt veraltete Werte. Erledigt 2026-09-23 (Commit `4a13298`): Praemisse geprueft und widerlegt (core/biome_generator.py und docs/BIOME_MATRIX.md kennen keinen Siedlungseignungswert) — Tabelle als gepruefte Primaerquelle dokumentiert statt umgebaut. | 1 | #87 |
| [x] | 15.3 | **`core/settlement_generator.py`: `evaluate_biome_suitability()` iteriert 27x biomweise statt Lookup-Tabelle.** Bei jedem Aufruf wird ueber alle 27 Biome per Schleife gegangen, obwohl das Ergebnis ein reiner Tabellenwert ist. Effizienzfund, kein Fehler — ein Dictionary-Lookup ersetzt die Schleife. Erledigt 2026-09-23 (Commit `f387187`): Lookup-Array per Fancy-Indexing ersetzt die Schleife, Verhalten inkl. Logzeile fuer unbekannte IDs unveraendert. | 0.5 | #88 |
| [x] | 15.4 | **`gui/utils/map_export.py`: Normalisierung dreifach dupliziert.** Dieselbe Normalisierungsrechnung (Werte auf einen festen Bereich skalieren) steht an drei Stellen der Datei separat, statt einmal als Funktion. Erledigt 2026-09-23 (Commit `219f0bc`): `_normalize_to_unit()`/`_quantize_uint8()` als gemeinsame Helfer fuer rgb_anteil, scalar_8bit, wassertiefe und daempfungsmaske. | 0.5 | #89 |
| [ ] | 15.5 | **`core/erosion_generator.py`: CPU-Groessen-Waechter doppelt vorhanden.** Die Pruefung, ob eine Kartengroesse noch auf der CPU gerechnet werden darf (Performance-Grenze), steht zweimal im selben Modul statt einmal. | 0.5 | – |
| [ ] | 15.6 | **`core/settlement_generator.py`: `platziere_bruecken()` mit dreifach verschachtelter Schleife.** Fuer jede Bruecke wird ueber drei ineinander verschachtelte Schleifen gesucht, wo eine raeumliche Vorfilterung (z.B. ueber ein Gitter) die meisten Kombinationen von vornherein ausschliessen wuerde. Effizienzfund bei groesseren Kartengroessen relevant. | 1 | – |
| [ ] | 15.7 | **`gui/tabs/overview_tab.py`: `welt_backen()` macht Datei-Ein-/Ausgabe sequenziell blockierend.** Mehrere unabhaengige Schreib-/Lesevorgaenge laufen nacheinander statt parallel bzw. asynchron, was die Wartezeit beim Speichern einer Welt unnoetig verlaengert. | 1 | – |
| [ ] | 15.8 | **`core/settlement_generator.py`: `apply_spline_smoothing()` als eigenstaendige Funktion dupliziert.** Es gibt bereits eine vergleichbare Gaettungsfunktion an anderer Stelle im Code (vermutlich in der Fluss- oder Wegeverarbeitung); diese Funktion baut dieselbe Logik erneut nach, statt sie wiederzuverwenden. | 0.5 | – |
| [ ] | 15.9 | **`gui/tabs/overview_tab.py`: `welt_laden()` behandelt nur 2 von 7 Generatoren ueber einen eigenen Setter-Pfad.** Terrain und Geologie laufen (auch nach dem Fix in Commit `2ed2bd6`) ueber eine eigene, benannte Setter-Pruefung, die uebrigen fuenf Generatoren (Weather, Erosion, Water, Biome, Settlement) generisch. Das ist eine strukturelle Unwucht, keine falsche Berechnung — der tiefere Fix waere EIN gemeinsamer Ruecksschreibe-Mechanismus fuer alle sieben Generatoren statt zwei Sonderfaellen plus einem generischen Pfad. | 1 | – |

**Bilanz-Hinweis:** diese neun Punkte sind in der Zaehlung oben (Abschnitt
"Bilanz") noch nicht mitgezaehlt — die naechste Nachzaehlung erfasst sie.
| 6 | 8.x LOD entfernen | Grosser Umbau, hohes Risiko — bewusst zuletzt. | 7 |

---

# 1 — Wetter und Klima

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **1.1** | **Temperaturmodell als Festlegung.** `T = sockel(x,y) + amplitude(x,y) · jahresgang(t)`. Raum einmal gerechnet, Zeit als skalare Funktion. Alle neun Regionen treffen ihr Klimaziel **auf 1 K**. | 2 |
| **[x]** | **1.2** | **Expositionsnormierung** je Region auf Mittel 0.5, weich ueber die Grenzen geglaettet. Ohne sie laege jede Region mehrere Kelvin daneben (rohe Exposition mittelt auf 0.68, nicht 0.5). | 1 |
| **[x]** | **1.3** | **Niederschlag als Festlegung.** Regionswert × Luv-Faktor × Regenschatten, direkt auf `NIEDERSCHLAG_ZIEL` normiert. Luv gegen Lee **3.6×**, Jahressummen auf 5–11 % genau. Kein Kreislauf: je Pixel EINMAL eine feste Strecke gegen den Wind, ein Integral statt eines Einschwingens. | 2 |
| **[x]** | **1.3b** | **Jahresgang des Niederschlags**, aus der Kontinentalitaet abgeleitet (steckt schon in `temp_spanne`). Morobora Jul/Jan **1.96** (Sommerregen), Clonagh **0.64** (Winterregen), See **0.72**. | 0.5 |
| **[x]** | **1.3c** | **Seeniederschlag gedaempft** auf 80 % des Landwerts. Vorher bekam die See 1.37× so viel wie das Land (Nevadin 3.38×) — der Nutzer sah "knallgruen ueber dem meer". | 0.5 |
| **[x]** | **1.4** | **Ein Sonnensatz statt sechs.** Der Schattenwurf laeuft einmal statt sechsmal; die Jahreszeit steckt in der Temperaturkurve. | 1 |
| **[x]** | **1.9** | **Klimaschaerfe** (`KLIMA_SCHAERFE = 2.5`). Die Klimafelder mischen schaerfer als die Gelaendefelder — vorher ueberdeckte das Klimagefaelle innerhalb einer Region (bis 12 K) die Hoehenabnahme vollstaendig. Skerrheim jetzt **−0.59 K/100 m** statt +0.10. | 1 |
| **[x]** | **1.12** | **Zeitachse auf Monatsbasis.** Der Niederschlag wird als MONATSRATE gefuehrt und erst beim Ablegen mit `MONATE_JE_TICK` multipliziert. Vorher standen dort Zweimonatssummen, ohne dass es irgendwo stand. Auf 12 Monatsticks umgestellt liefert dieselbe Jahressumme, bis auf 0.0 % in allen neun Regionen. | 0.5 |
| [ ] | 1.5 | **Eine Atmosphaerenebene statt drei.** Spart rund Faktor 3 auf dem letzten grossen Posten. **Untersucht, aber NICHT umgesetzt** (2026-08-11): MID/HIGH-Layer sind bestaetigt von nichts ausserhalb weather_generator.py konsumiert (nur GROUND verlaesst die Datei), aber die drei Schichten sind in der Simulation eng verzahnt (Terrain-Ablenkung/-Speedup, Vertikalaustausch, LOD-Vererbung, Randpuffer) - ein sicherer Umbau braucht eine eigene, fokussierte Sitzung mit einem staerkeren Wind-Sicherheitsnetz als heute vorhanden, siehe 1.6. | 1 |
| **[~]** | **1.6** | **Atmosphaerensimulation zurueckschneiden.** Die urspruengliche Sorge ("Schrittzahl haengt an der Aufloesung, driftet") ist bereits durch frueheres, hier nicht dokumentiertes Arbeiten geloest: das Wetter rechnet seit `WETTER_GITTER` immer auf festen 256px, die Schrittzahl folgt der Gitterkante statt dem LOD. Neu 2026-08-11: docs/spezifikation/10_REGIONEN.md B.5 legt Windziele je Region fest (Referenzorte wie bei Temperatur/Niederschlag), `_wind_regional_faktor()` normiert das simulierte Regionsmittel direkt darauf (alle neun Regionen innerhalb 0.5 m/s, "direkt auf den Zielwert normieren" wie bei 1.3). Ein erster Versuch, den Regler raeumlich in die Druckgradient-Antriebskraft einzuspeisen, zeigte praktisch keinen Effekt (Faktor 0.16 gegen 0.71: 1.871 gegen 1.872 m/s) - der Druckgradient ist nur einer von mehreren additiven Antrieben. Luv/Lee-Kontrast (Ziel 1.5-2x) ist eingebaut, aber NICHT verlaesslich: die Simulation hat selbst schon eine terraingetriebene Windstruktur, die mit dem einfachen Hangneigungsansatz antikorreliert (-0.53 gemessen im Nevadin) statt neutral zu sein. Gesichert in `tests/smoke_test_weather_wind_regions.py`. **Die eigentliche Schrittzahl-Reduktion selbst (weniger als 25 Schritte/Monat) ist NICHT versucht worden** - das waere der naechste Schritt innerhalb dieses Punkts. | 2 |
| [ ] | 1.7 | **Windverwischung** als Abschluss: gerichtete Faltung entlang der Windrichtung. Offene Frage: wird das Meer mitverwischt? | 1 |
| [ ] | 1.8 | Klimatabelle **an einer Quelle gegenpruefen** — meine Zahlen sind aus dem Gedaechtnis, auf 1–2 K genau. | 0.5 |
| **[x]** | **1.10** | **Temperatureingaben auf Direktnormierung umgestellt.** `_temperatur_raummuster()` normiert `t_mittel`/`t_spanne` jetzt ueber `_je_region_auf_mittel()` direkt auf `KLIMA_ZIEL` (dieselbe Funktion, die schon Niederschlag traf, 1.3) - "direkt auf den Zielwert normieren" statt vorkompensierter Eingabe. `core/terrain_weltkarte.py` REGIONEN.temp_mittel_m0/temp_spanne sind seither die LESBAREN KLIMA_ZIEL-Werte selbst ("Morobora 3.8" statt "Morobora 1.8, damit 3.8 ankommen"). | 1 |
| **[x]** | **1.11** | **Morobora-Jahresspanne trifft jetzt PER KONSTRUKTION, nicht per Eichung.** Ueber vier Seeds gemessen (bewusst NICHT nur die drei alten Eichseeds): alle neun Regionen innerhalb 0.7 K (Jahresmittel) bzw. bis 1.7 K (Jahresspanne, schlechtester Fall Morobora/Seed 4242 statt vorher bis 1.9 K auf einem ungeeichten Seed) - und das OHNE jede Handkalibrierung, haelt also fuer JEDEN Seed, nicht nur die drei, auf die frueher geeicht wurde. Gesichert in `tests/smoke_test_weather_temperature_direktnormierung.py`. | 0.5 |

**Messwerte zum Wetter:** `weather.temperature` 170.7 s → **14.4 s** bei 512 px
(Faktor 11.9). Bei 256 px praktisch unveraendert — dort dominiert die
Atmosphaerensimulation, also Punkt 1.6.

# 2 — Biome

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **2.1** | **15 Grundbiome eingetragen**, 5 tropische raus, 3 zu grobe aufgeteilt. **Jedes kommt vor**, keines ist tot. | 1 |
| **[x]** | **2.3** | **Regionsaffinitaet** statt neun Matrizen: ein Eignungsbonus von 0.35 je regionstypischem Biom. 28 von 30 Hauptbiomen erwartungsgemaess. | 2 |
| **[x]** | **2.4** | **Weich mitgeloest**: der Bonus wird als Feld geglaettet, es gibt keine harte Kante an den Voronoi-Zellen. Neun getrennte Matrizen mit argmax haetten genau die erzeugt. | — |
| **[x]** | **2.5** | Biomfarben und Namen auf die neuen fuenfzehn. | 0.5 |
| **[x]** | **2.6** | **Niederschlagsskala an die Ticklaenge gebunden.** Sie stand als feste Zahl da; eine Umstellung auf Monatsticks haette **jedes Biom um Faktor zwei verschoben**, ohne dass jemand etwas geaendert haette. | — |
| **[~]** | **2.2** | **9 Superbiome.** Alpin und Firn auf TEMPERATURregel umgestellt (Juli unter 10 bzw. 0 Grad) - die alte Hoehenregel loeste bei 2750 m nie aus und lieferte stillschweigend Null. **Loest jetzt aus** (siehe 2.7 - unabhaengig von diesem Punkt durch die Klimakalibrierung geloest). Cliff/Beach existieren bereits als eigene Kategorien in `apply_super_biome_overrides`. Fels, Duenen, Aue bleiben offen. | 1.5 |
| **[x]** | **2.8** | **Superbiome sassen an der falschen Stelle** (Nutzerbild 2026-08-10: diagonale Baender aus Fluss-/Bachfarben quer ueber Land UND offenes Meer). Ursache lag NICHT in Aufloesung/Upsampling, sondern im Teilpixel-Zufall des 2×2-Supersamplings: `x*4 + y*4` haengt nur von der Koordinaten**summe** ab, jede Diagonale bekam denselben Wert. Betraf CPU-Pfad UND den GPU-Shader gleichermassen. Ersetzt durch eine echte Durchmischung (Hash aus drei Primzahlen je Achse), dabei die Schleifenfassung zugleich vektorisiert. Gesichert in `tests/smoke_test_biome_supersampling.py`. | 1 |
| **[x]** | **2.7** | **DAS GEBIRGE IST ZU NIEDRIG FUER EINE BAUMGRENZE** - Befund vom 2026-08-07 (kaelteste Julitemperatur an Land 10.4 Grad, BAUMGRENZE_JULI_C=10.0 loeste nirgends aus). **Nachgemessen 2026-08-11: Befund gilt nicht mehr**, aber in zwei Schritten korrigiert. Erstmessung (mit `weather.temperature`s `temp_map`) zeigte faelschlich ~8-9% Landflaeche - das Feld war das JAHRESMITTEL, nicht Juli (siehe 9.1, derselbe Bug betraf auch die Basis-Biom-Klassifikation). Mit dem in 9.1 gefixten `temp_map_juli` **neu gemessen: alpine_level 0.89%, snow_level ~0%** - viel kleiner, aber genau deshalb plausibler fuer eine echte, seltene Alpinzone statt einer flaechendeckenden. Auslöser bleibt derselbe: die Temperatur-Direktnormierung (1.10/1.11) hat die Julitemperatur-Verteilung so verschoben, dass die BESTEHENDE Schwelle (BAUMGRENZE_JULI_C=10.0, unveraendert) jetzt an echten Berggipfeln greift. KEIN Schwellenwert wurde geaendert. Stale-Code-Kommentar in `core/biome_generator.py` (bei `BAUMGRENZE_JULI_C`) korrigiert. | — |

**Voraussetzung erfuellt:** beide Achsen sind belastbar — Temperatur auf 1 K,
Niederschlag auf ~5 %, beide mit Jahresgang.

# 3 — See und Kuesten

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **3.4** | **Kuestenform je Region** (`kuestenform`). Potenzkurve auf die Hoehe, nicht Ortsband — zwei Entwuerfe davor waren falsch. Nebelrode 268 → **177 m** bei 100 m Kuestenabstand (Klippe weg), Skerrheim 53 → **112 m** ueber und −31 → **−76 m** unter Wasser. | 1.5 |
| **[x]** | **3.5** | **Estrande +40 %**, aus dem Meer allokiert: Kontinent 164 → **172 km²**. `flaeche_soll` allein haette nur umverteilt. | 0.5 |
| **[x]** | **3.1** | **Seegrad ueber den Voronoi-Zellgraphen** (`seegliederung()`, core/terrain_weltkarte.py). EIGENE Punktmenge (nicht die der Kultur-Regionen), Land 200/See 400 Zellen ueber die jeweilige Flaeche skaliert. Grad = Breitensuche ab allen Land-beruehrenden Zellen. **Bauen und angesehen** (gerendert, per Read-Tool geprueft) — keine sichtbare Kachelung; Zellkantenlaenge See ~0.90 km gegen Land ~0.81 km, vergleichbare Groessenordnung. | 1 |
| **[x]** | **3.2** | **Schelf durch die Gradtabelle ersetzt** (0/−40/−90/−150/−200 m, `TIEFE_JE_SEEGRAD`). Ersetzt die alte `-t*(1-exp(-d/L))`-Formel (zwei freie Konstanten, an der Aufloesung der Distanztransformation haengend) durch eine echte Festlegung ueber den Seegrad, ueber die Zellgrenzen geglaettet. Gemessen: Zieltiefe je Grad innerhalb 5 m der Tabelle (im Zellkern, fern jeder Gradgrenze). `schelf_tiefe_m`/`schelf_laenge_m`-Parameter aus `weltfeld()` entfernt (nirgends mit Nicht-Standardwert aufgerufen). | 1 |
| **[x]** | **3.3** | **Seegrad als Grundlage fuer Seewege** ("ab Grad 1" statt "ab 10 m Tiefe"). `bau_seekostenfeld()`/`_seeweg_anteil_tief()` nehmen jetzt optional `seegrad` entgegen und pruefen `seegrad >= 1` statt der Hoehenschwelle; ohne `seegrad` (alter Nicht-Weltkarten-Pfad) bleibt die alte Hoehenschwelle SEEWEG_TIEFE_ZIEL_M als Rueckfall erhalten. | 0.5 |
| **[x]** | **3.6** | **Seezellen mit Regionszuordnung** *(Nutzerwunsch 2026-08-07, umgesetzt 2026-08-11)*: `ufer_region_a`/`ufer_region_b` als eigene Felder (statt eines dicts je Zelle - konsistent mit `region_map`/`seegrad`, ein Wert je Pixel), 91.4 % Uebereinstimmung mit einer unabhaengigen Kreuzpruefung. Die drei genannten Regeln jetzt auch gebaut: `SEETYP_TIEFENTABELLE` gibt Skerrheim (Grad 1 bereits −90 m statt −40 m) und Clonagh (Grad 1 nur −17 m, erst ab Grad 2 die eigentliche Vertiefung) je eine eigene Zieltiefen-Tabelle statt der Standardtabelle; `see_eis` ist ein neues bool-Feld, True auf jeder Seezelle, deren naechstes Ufer die Morobora ist. Zwei Nachbesserungen 2026-08-11: erst "seewege freihalten, nur grade 1 und 2" (harter Schnitt), dann "grade 0 ist 100% eis grade 1 75% chance ... grade 2 ... 50% ... grade 3 ist 25%" (WAHRSCHEINLICHKEIT statt hartem Schnitt, `EISWAHRSCHEINLICHKEIT_JE_SEEGRAD`, Wuerfel je ZELLE nicht je Pixel — zusammenhaengende Schollen statt Rauschen). Grad 4+ bleibt 0 %, Seewege bleiben frei. Gemessen ueber 7 Karten gepoolt (einzelne Grade haben oft nur eine Handvoll Zellen, ein Einzelseed ist nicht aussagekraeftig): 97/65/50/21 % gegen Vorgabe 100/75/50/25 %, alle innerhalb 15 Prozentpunkten. Jahreszeitliche Beschraenkung ("nur im Winter") vom Nutzer selbst vertagt — diese Karte bleibt ein statischer Schnappschuss ohne Zeitachse. **Nachbesserung 2026-08-11 (Live-App-Befund):** Seeeis war bis dahin unsichtbar (kein Display-Konsument) UND die Ozean-Anzeige zeigte stattdessen vereinzelte weisse/graue Punkte - siehe 2.9. `see_eis` bekam jetzt eine eigene Biom-Kategorie ("Sea Ice", `#c8e8f5`, Index 26). Zusaetzlich `TIEFE_JE_SEEGRAD[0]`/`SEETYP_TIEFENTABELLE[*][0]` von 0.0 auf −3.0 m: "das meer ist nicht vertieft ... land um die regionen herum (wahrscheinlich auf 0m) ... soll an der kueste immer auf -3m seetiefe abfallen" - ohne erzwungene Mindesttiefe blieb ein breiter Saum um 0 m dem Rauschen ueberlassen, was als unentschiedenes Gruen/Blau erschien und den ganzen kuestennahen Teil der See flach aussehen liess. | 2 |
| **[x]** | **3.8** | **Kuesten-Archetypen** *(am laufenden Programm bestaetigt 2026-08-13: "Kuestenarchetypen wechseln sich ab ... gut genug" - mit benanntem Vorbehalt, siehe 3.14)* (Nutzer-Vorgabe 2026-08-12, ausfuehrlich diskutiert und Tabelle gemeinsam abgestimmt): jede Region bekommt 3 an realen Kuesten orientierte Auspraegungen (`KUESTEN_ARCHETYPEN`, `core/terrain_weltkarte.py`), die deterministisch laengs der Kuestenlinie wechseln - z.B. Nebelrode: Ruegen-Kreidekueste/Ostsee-Flachkueste (bewusst 50% Max-Anteil, "soll meistens zum Meer abfallen")/Foerdenkueste; volle Tabelle in der Session-Historie. Neue Funktion `_kuesten_umformen()`: (1) Kuesten-Saatpunkte je Region ausduennen, (2) Archetyp-Zuteilung per Quote aus `max_anteil` (garantiertes Vorkommen) + Praeferenz fuer zum vorhandenen Rohgelaende passende Punkte + Seed-Jitter (Einzigartigkeit je Karte), (3) Naechster-Saatpunkt-Verteilung aufs Kuestenband (gleiches Muster wie `seegliederung()`), (4) Zielhoehe aus Archetyp-Profil (`hoehe_faktor`/`winkel_grad`/`kantig` fuer facettierte Fjord-/Schaerenkueste/`strand_anteil` fuer Luecken in der Klippenwand), (5) weiches Blenden zur bestehenden Hoehe. Laeuft NACH `kuestenform()` (Regions-Grundform bleibt), VOR der Seegrad-Tiefe (behaelt das letzte Wort ueber die Meerestiefe). **Echter Bug gefunden+behoben:** die Zonengrenzen-Glaettung lief ueber die Kuestenlinie hinweg und konnte Seepixel ueber 0 anheben ("Seeeis liegt auch auf Land", `smoke_test_seegliederung.py`) - See/Land-Vorzeichen wird jetzt nach dem Blenden erzwungen erhalten (0.5m Sicherheitsabstand). Determinismus verifiziert (gleicher Seed -> bitidentisch), regressionsfrei (`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`, `smoke_test_terrain_scale_coupling.py`, `smoke_test_terrain_erosion_filter.py`). **Nachbesserung 2026-08-12:** (a) 2D-Anzeige - neuer Radio-Knopf "Kuestentypen" im Terrain-Reiter (`_render_kuesten_archetypen()`): Regionsfarbe wie beim "Regionen"-Modus, Helligkeit global (nicht je Region) nach `hoehe_faktor` (flach=hell, steil=dunkel), Deckkraft nach `kuesten_staerke` (die "Strahlungstiefe") - Kernzonen kraeftig, Rand blendet weich aus. Dafuer `kuesten_archetyp`/`kuesten_staerke` neu durch die ganze Terrain-Datenkette durchgereicht (TerrainData/assemble/set_terrain_data_complete_lod, gleiches Muster wie `seegrad`). (b) Performance - die Strand-/Facetten-Rauschfelder rechneten bisher auf der GANZEN Karte statt nur im Ausschnitt der jeweiligen Zone (oft < 1% der Kartenflaeche); neue `_kuesten_rauschen_lokal()` beschraenkt Zufallsziehung+Gauss-Filter auf den Begrenzungsrahmen der Zone. Gemessen: 512px von 6-8s auf ~4s zusaetzlich. **[~] statt [x]:** immer noch NICHT visuell im Programm bestaetigt (weder Gelaende noch die neue 2D-Anzeige) - reine geometrische/statistische Pruefung. Ein Compute-Shader fuer die Naechster-Punkt-Zuordnung waere der naechste Optimierungsschritt, falls 4s bei 512px noch zu viel sind - bewusst noch nicht angegangen (Shader-Arbeit hat in diesem Projekt eine Vorgeschichte). | 3 |
| **[x]** | **3.7** | **Rohes Rauschen vor der Seegrad-Tiefe auf −10 m angehoben** (Nutzer-Vorgabe 2026-08-12: "bevor das geschieht sollte die Tiefe aber auf min. -10m angehoben werden", Ziel: weniger 3D-Mesh-Dreiecke bei dicht beieinanderliegenden Inseln, Beispiel Thalassia). `H = np.where(H<=0, np.minimum(H, felder["seegrad_tiefe"]), H)` liess tiefere ROHE Rauschwerte unangetastet ("minimum kann nur vertiefen") - lokale Rauschausreisser blieben dadurch tiefer als ihr Seegrad verlangte, sichtbar als kleinraeumige, unregelmaessige Vertiefungen quer durch die sonst geglaettete Zonierung. Eine Zeile davor ergaenzt: `H = np.where(H<=0, np.maximum(H,-10.0), H)` - danach zieht `minimum(H, seegrad_tiefe)` praktisch immer den glatten Seegrad-Wert. Regressionsfrei (`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`, `smoke_test_terrain_scale_coupling.py`). | 0.5 |
| **[x]** | **2.9** | **Landmarks/Superbiome-Schnee auf offenem Meer** *(Live-App-Befund 2026-08-11: "warum sind landmarks im meer", "vereinzelte weisse Punkte im Meer")*. ZWEI unabhaengige Bugs mit demselben Muster - eine Bedingung ohne Landfilter, die auf der Weltkarte (negative Hoehen = Meer) auch von kalten/flachen Meerespixeln erfuellt wird: (1) `calculate_landmarks()`s "gipfel"/"abgelegen"-Kategorien pruefen nur `norm_height`, nicht `heightmap>0` - ein Meerespixel mit civ_map~0 (weit draussen ist niemand) und flacher "Hangneigung" (offene See ist flach) erfuellte klaglos "abgelegen". (2) `_calculate_snow_level_probabilities`/`_calculate_alpine_level_probabilities` haengen nur an der Julitemperatur (2.2-Umbau) - ein hinreichend kaltes Meeresstueck (z.B. See Nord vor der Morobora) erfuellt rechnerisch dieselbe Bedingung wie ein Gipfel, und `_apply_supersampling_cpu()` ueberschreibt dann einzelne Sub-Pixel des korrekt erkannten `ocean_mask` stochastisch mit "Snow Level"/"Alpine Level" - die gemeldeten weissen/grauen Punkte. Beide mit `& land` bzw. `np.where(ocean_mask, 0.0, ...)` behoben. Gesichert in `tests/smoke_test_settlement_region_grid.py` (Landmarks) und manueller Pruefung (Snow-Level-im-Meer 0, Alpine-Level 3 von 262144 - eine isolierte, nicht flood-fill-verbundene Senke, kein systematischer Rest). | 1 |

| [ ] | 3.9 | **Der Kuestenpass (3.8) hat die Regions-Hangeichung verschoben — und wurde nie dagegen geprueft** *(gefunden 2026-08-12 beim vollstaendigen Testlauf)*. `smoke_test_regionen_welt.py` schlaegt fehl: alle neun Regionen zu steil, dazu "Ordnung verrutscht um mehr als zwei Plaetze: Estrande, Samarcia". **Der Anteil, den 3.8 daran hat, ist gemessen** (384 px, Seed 20260804, roher `weltfeld`-Median-Hang, Kuestenpass ein/aus): Samarcia **10.6° mit / 6.6° ohne** (Ziel 6.5), Morobora **9.1 / 5.9** (Ziel 7.5), Skerrheim **21.7 / 19.5** (Ziel 16.0). Bei Samarcia und Morobora stammt die gesamte Abweichung aus dem Kuestenpass — ohne ihn treffen beide ihr Ziel fast exakt. **Aber er ist nicht die einzige Ursache:** auch ohne ihn verfehlen Estrande (15.4 statt 8.5), Thalassia (18.0 statt 11.5) und Nevadin (24.6 statt 19.5) ihr Ziel deutlich — das ist eine aeltere, unabhaengige Verstimmung. **Zwei getrennte Aufgaben also:** (a) entscheiden, ob die Kuestenformung ueberhaupt in die Regions-Hangeichung eingehen soll — sie formt bewusst Klippen, ein hoeherer Kuestenhang ist teilweise gewollt, dann muessen die ZIELWERTE nachgezogen werden statt der Formung; (b) die davon unabhaengige Verstimmung bei Estrande/Griechischen Inseln/Nevadin getrennt untersuchen. **Lehre:** 3.8 wurde gegen vier Terrain-Tests geprueft, `smoke_test_regionen_welt.py` war nicht darunter — die Regionseichung ist aber genau das, was eine Gelaendeformung als erstes verstimmt. Bei jeder kuenftigen Aenderung an `weltfeld()` gehoert dieser Test in die Pruefliste. | 2 |

| **[x]** | **3.11** | *(am laufenden Programm bestaetigt 2026-08-13: "Klippen sehen aufs erste ganz ok aus ... ich wuerde es als 85% bewerten und damit gut genug fuer unseren stand")* **Kuestenprofil an lokale Terrainhoehe koppeln + Reichweite an effektive Hoehe koppeln** *(Nutzer-Idee 2026-08-13, ENTWURF vor Umsetzung dokumentiert)*. Ausgangsbild: Cliffs of Moher stehen im Bild auf **flachem** Rohgelaende - die Zielhoehe kommt starr aus der Tabelle (`KUESTENHOEHE_M * hoehe_faktor`), unabhaengig davon, was um die Zone herum tatsaechlich an Land ist. Ein weicherer UEBERGANG (Maske/Strahlkraft) loest das nicht, weil das Problem nicht die Kante ist, sondern die absolute Zielhoehe selbst - eine 630-m-Klippe neben 20 m hohem Land bleibt eine 630-m-Klippe, nur mit laengerem Auslauf. **Nutzer-Vorgabe, in drei Teilen:**
1. **Zielhoehe wird ein gewichteter Mittelwert aus Tabellenwert und lokaler Terrainhoehe**, nicht mehr die Tabelle pur. `lokale_hoehe` (5x5-Mittel um jeden Saatpunkt, JETZT SCHON fuer die Archetyp-Zuteilung berechnet, siehe oben im Code) wird ueber die bereits vorhandene naechster-Saatpunkt-Zuordnung (`naechster`, cKDTree) auf jeden Pixel der Zone projiziert - ergibt ein stueckweise-konstantes Feld, das die bestehende Zonengrenzen-Glaettung am Ende der Funktion ohnehin schon einebnet, also KEIN neuer Glaettungsschritt noetig. Ein Gewicht (`KUESTEN_LOKALER_EINFLUSS`, Vorschlag 0.35) mischt Tabellenwert und lokale Hoehe.
2. **Reichweite waechst mit der EFFEKTIVEN (gemischten) Hoehe, nicht mehr nur der Tabellenwert.** Bereits vorhanden: `skala_m = Zielhoehe / tan(winkel)` fuer das INNERE Anstiegsprofil (aus 6.17/6.19) - wird jetzt aus der neuen effektiven, pixelweise variierenden Zielhoehe berechnet statt der festen Tabellenzahl, ist also automatisch schon ortsabhaengig. Die AEUSSERE Blendmaske (`reichweite_karte`, treibt `kuesten_staerke`) wird an dasselbe `skala_m` gekoppelt (`reichweite = max(Tabellenwert, FAKTOR * skala_m)`, Vorschlag FAKTOR=3.5, deckt ~97% des Anstiegs ab) - eine hohe effektive Klippe strahlt dadurch automatisch weiter ins Land, eine niedrige bleibt kurz. Genau die Vorgabe "wenn die klippen sehr hoch sind, dann strahlen diese auch etwas weiter rein, damit es realistischer aussieht (nicht so steil ueberall)".
3. **Facetten-/Strand-Rauschamplitude bleibt an der TABELLEN-Reichweite haengen, nicht an der jetzt moeglicherweise laengeren effektiven** - Felsstruktur ist eine Eigenschaft des Archetyps (Kantigkeit/Strandhaeufigkeit aus der Tabelle), keine Funktion der lokalen Hoehe. Getrennter Fund beim Nachsehen der 3.10-Ergebnisse (siehe unten): die Amplitude war bisher ABSOLUT in Metern fest, wurde aber nie an die durch 3.10 verkuerzte Reichweite angepasst - dieselbe absolute Unruhe macht in einem 0.7-km-Band viel mehr aus als im vorherigen 1.8-km-Band, sichtbar als zellige/blobartige Kuestenlinie statt einer glatten Form. Wird mit umgesetzt: Amplitude proportional zur Tabellen-`reichweite_km` statt absolut.
Aufwand insgesamt gering - wiederverwendet ausschliesslich bereits vorhandene Felder/Formeln, keine neue Rechnung, keine neue Glaettung. **UMGESETZT UND GEMESSEN 2026-08-13.** Alle drei Teile gebaut: (1) `lokale_hoehe_karte` je Pixel aus der bereits vorhandenen Saatpunkt-Referenz, `KUESTEN_LOKALER_EINFLUSS=0.35` mischt sie in die Zielhoehe: `ziel_hoehe_effektiv = tabelle*(1-0.35) + lokale_hoehe*0.35`. (2) `skala_m` (Anstiegsstrecke) UND die aeussere Blendmaske (`reichweite_karte = max(Tabellenreichweite, 3.5*skala_m)`) haengen jetzt an dieser effektiven, pixelweise variierenden Hoehe - auch die Unterwasser-Eintiefung folgt ihr, statt weiterhin an der reinen Tabelle zu haengen (ein lokal gedaempfter Klippenfuss soll nicht trotzdem einen tiefen Graben hinterlassen). (3) **Praezisere Ursache der 'zelligen' Kueste gefunden als zunaechst vermutet:** nicht die absolute Rauschamplitude allein, sondern dass das Facettenrauschen mit VOLLER Staerke exakt an der Wasserlinie ansetzte, wo das Basisprofil bewusst gegen 0 geht - eine Wackelamplitude von mehreren Metern kippt dort das Vorzeichen zufaellig. Jetzt faehrt die Rauschstaerke von 0 an der Wasserlinie auf voll hoch, sobald der Anstieg abgeschlossen ist (`np.clip(d_land/skala_m,0,1)`). **Sichtgeprueft** (Kuestentypen-Rendering vor/nach, siehe Bildpaar dieser Sitzung): die Kuestenlinie ist danach sichtbar glatter, keine Blasen/Zellen mehr, ein zusammenhaengender Saum. **Zwei ehrliche Zielkonflikte, gemessen (384px, Seed 20260804):** Winkel an der Wasserlinie im Mittel weiter von `winkel_grad` entfernt als vorher (13.0 -> 17.5 Grad) - folgerichtig, weil eine durch flaches Umland gedaempfte Zielhoehe zusammen mit dem `2*mpp`-Mindestwert fuer `skala_m` die tatsaechlich erreichte Steigung senkt, das ist GENAU die gewuenschte Daempfung, nur eben auf Kosten der Winkeltreue. Archetyp-Flaeche auf Land gewachsen (33.3% -> 48.2%) durch den reichweite-waechst-mit-hoehe-Mechanismus (Punkt 2) - bleibt weit unter dem urspruenglichen 96%-Befund aus 3.10, aber eine reale Verschiebung, kein Fehler. Falls diese Kompromisse zu stark sind: `KUESTEN_LOKALER_EINFLUSS` (0.35) ist der eine Regler dafuer, kleiner = naeher an der reinen Tabelle. Regressionsfrei (`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`, `smoke_test_terrain_scale_coupling.py`, `smoke_test_terrain_erosion_filter.py`, `smoke_test_river_reaches_sea.py`, `smoke_test_adaptive_terrain_mesh.py`). **Noch offen:** Live-Bestaetigung im Programm (dieser Befund stuetzt sich auf ein 2D-Rendering ausserhalb der App). | 1 |

| **[x]** | **3.10** | *(am laufenden Programm bestaetigt 2026-08-13, zusammen mit 3.11)* **Kuestenpass wirkte auf 96 % der Landflaeche - UMGEBAUT 2026-08-13, mit einem offenen Nebeneffekt.** Ausgangsbefund (Nutzer: "es sieht auf der karte aus als wuerden diese kuestenbereiche das gesamte land erstrecken"): von 76321 Landpixeln trugen 73283 einen Archetyp, `kuesten_staerke` hatte auf dem Land den Median 0.473. Ursache war eine einzige globale Bandbreite (1.2 km, im Bandtest 1.8 km) - auf einer 21.3-km-Karte mit gegliederter Kueste liegt fast jeder Landpunkt naeher am Wasser. **Umgesetzt:** (a) jeder der 27 Archetypen traegt jetzt eine eigene `reichweite_km` (Fjordwand 0.70 km, Ostsee-Flachkueste 0.18 km) - Nutzer-Vorgabe "die strahlwirkung soll je nach kuestentyp unterschiedlich stark strahlen": (b) der Abfall ist quadratisch statt linear (kraeftig an der Wasserlinie, dann zuegig verblassend): (c) `kuesten_archetyp` wird zurueckgesetzt, wo die Staerke unter 2 % faellt - das Feld meldet keinen Kuestentyp mehr fuer Land, an dem nichts geformt wurde (die 2D-Anzeige liest dasselbe Feld): (d) **Klippenfuss sticht unter Wasser weiter** (Nutzer-Vorgabe "bei hohen klippen auch tiefer ins wasser stechen damit wir spaeter keine probleme mit dem verfahren bekommen", Vorsorge fuer 6.20) - Tiefe linear aus `hoehe_faktor`, 12 m bei Flachkueste bis 177 m bei der Fjordwand: (e) die -10-m-Klemme in `weltfeld()` nimmt den geformten Kuestensaum jetzt aus, sonst haette sie den tiefen Fuss sofort wieder hochgezogen. **Ein Fehler dabei gefunden und behoben:** die Zonenglaettung lief mit EINEM gaussian_filter quer ueber die Kuestenlinie, wo Landwerte von mehreren hundert Metern auf Seewerte treffen - das Mittel landete nahe null und der Klippenfuss wurde wieder hochgezogen. Gemessen war die Seetiefe an der Kueste dadurch **-1.7 m, also FLACHER als ganz ohne den Pass (-8.1 m)** - das genaue Gegenteil des Gewollten. Jetzt maskiertes Glaetten je Seite (normalisierte Faltung), Kante an der Wasserlinie bleibt erhalten. **Gemessen nachher (384 px, Seed 20260804):** Archetyp auf Land **96.0 % -> 30.3 %**, Seetiefe an der Kueste **-8.1 -> -18.1 m** (tiefste Stelle -95 m, 25 % der Kuestensee tiefer als -30 m). **[~] wegen zwei offener Punkte:** (1) die Regions-Hangabweichung ist mit **4.22 Grad schlechter als ohne den Pass (3.38)** - ein schmaleres Band drueckt dieselbe Hoehe auf weniger Pixel, also steiler: das ist der Zielkonflikt aus 6.19 und braucht die Gestaltungsentscheidung des Nutzers, keine weitere Rechnung. (2) **`smoke_test_river_reaches_sea.py` schlaegt seither fehl** - bei 384 px 5 von 3087 bzw. 8 von 3328 Flussknoten enden nicht mehr im Meer (0.2 %), bei 256 px keiner. Ursache plausibel, aber NICHT verifiziert: die konzentrierten, hoeheren Kuestenwaende koennen eine Flussmuendung verriegeln. Sauberer Fix waere, Muendungen von der Klippenformung auszunehmen - dafuer muesste `_kuesten_umformen()` die Flusslagen kennen, die aber erst spaeter in `_calc_redistribution` entstehen. Zu entscheiden: Reihenfolge aendern, oder 0.2 % Sackgassen hinnehmen und die Zusicherung lockern. | 2 | **NACHTRAG 2026-08-13, Nutzereinwand "die klippen haben bereits winkel die festgelegt wurden, wo ist denn bitte die frage - das ist entweder noch nicht richtig umgesetzt oder ist buggy": der Einwand war berechtigt, es waren ZWEI Fehler, einer im Code und einer in der Messung.** (1) **Code:** die Zonenglaettung `sigma_zone` hing an der Bandbreite und lag bei 384 px bei rund 111 m - GENAU auf der Anstiegsstrecke der Klippen (Moher 111 m, Bretagne 132 m). Sie hat das Profil eingeebnet, statt nur die seitlichen Zonennaehte zu entschaerfen. Gegenprobe mit sigma 0.5 px: steile Typen sprangen von 34.8 auf 46.5 Grad. Jetzt auf 0.8 px gedeckelt - Naehte bleiben weich, Profil bleibt stehen. (2) **Messung:** der Vergleich lief ueber den Median der GANZEN Zone, also inklusive Klippenplateau oberhalb der Wand - dort ist der Hang naturgemaess flach. Der vorgegebene `winkel_grad` gilt aber an der WASSERLINIE. **Richtig gemessen (Abstand < 1.5 px vom Ufer, 384 px):** Santorini 75.1 bei Soll 84, Moher 74.7 bei 82, Fjordwand 73.1 bei 78: steile Typen im Mittel 62.5 bei Soll 74, flache 28.5 bei 35, **mittlere Abweichung 13.0 Grad statt der zuvor berichteten 29.1**. **Der Rest ist Aufloesung, kein Fehler:** `skala_m = max(2*mpp, L)` deckelt bei 384 px auf 111 m, was Moher rechnerisch auf 80 und Santorini auf 82 Grad begrenzt - gemessen 74.7/75.1, also nahe am Machbaren. Bei 1024 px ist der Deckel 41.6 m und damit kaum noch bindend: dort sollte der Sollwinkel besser getroffen werden (NICHT nachgemessen, da ein 1024er-Lauf mehrere Minuten kostet). **Damit ist die frueher hier vermerkte "Gestaltungsentscheidung" zum groessten Teil hinfaellig** - es war ueberwiegend ein Fehler.

| **[x]** | **3.12** | **Kuesten-Archetypen wirkten auch an Binnenseen** (Nutzer-Vorgabe 2026-08-13: "an einem binnensee soll keine klippe entstehen ... wenn es kontakt zum hauptmeer hat dann ist das ok"). Die Distanz-zur-Kueste-Berechnung in `_kuesten_umformen()` mass bisher den Abstand zum naechsten Gewaesser UEBERHAUPT - ein Seeufer zaehlte damit genauso als Kueste wie das offene Meer. Neue `_hauptmeer_maske(H)`: Zusammenhangskomponente von `H<=0`, die den Kartenrand beruehrt (in diesem Weltmodell zuverlaessig - der Kontinent liegt immer zentral, umgeben von Ozean bis zum Rand). `dist_land` misst jetzt Abstand zum Hauptmeer statt zu irgendeinem Wasser; Binnensee-Pixel selbst werden zusaetzlich explizit auf `-inf` gesetzt, damit ihr eigenes Ufer nie als Kuestenlinie durchrutscht. **Gemessen (512px, Seed 20260804):** 711 Binnensee-Pixel, keines mit Archetyp. Land am Binnenseeufer behaelt seinen Archetyp NUR dort, wo es zusaetzlich wirklich nah am Hauptmeer liegt (gemessen: median 208m Abstand bei den betroffenen 255 von 425 Uferpixeln, innerhalb der ueblichen Reichweiten 180-700m) - korrekt, kein Fehler. Sichtgeprueft, regressionsfrei (`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`, `smoke_test_terrain_scale_coupling.py`, `smoke_test_terrain_erosion_filter.py`, `smoke_test_river_reaches_sea.py`, `smoke_test_adaptive_terrain_mesh.py`, `smoke_test_settlement_sites.py`). | 1 |
| **[x]** | **3.13** | **Meerestiefe monoton zum Hauptmeer hin** (Nutzer-Vorgabe 2026-08-13: "keine stufe nach oben meer wenn davor eine vertiefung durch die klippen stattfand ... auch mit meeresarmen die ins land ragen"). Der tief eingestochene Klippenfuss (3.10/3.11) konnte lokal tiefer sein als die etwas weiter draussen geltende Seegrad-Tiefe - eine Stufe, die beim Wegfahren von der Kueste ploetzlich WIEDER flacher wird. Erste Fassung `_meerestiefe_monoton()`: ringweise Ausbreitung durchs Wasser per `grey_dilation`, `tiefe = min(eigener Wert, tiefster bereits erreichter Nachbar)`. **Verifiziert war nur MONOTONIE** (unabhaengiger Ring-Nachbau, 0 Verstoesse unter 69908 Nachbarpaaren) - **siehe Nachtrag: diese Fassung wurde inzwischen komplett verworfen und ersetzt**, weil eine andere Eigenschaft, Gleichmaessigkeit, nie geprueft wurde. | 1 | **NACHTRAG 2026-08-13 (zweite Runde), Nutzerbefund am gerenderten PNG: "da sind diagonale linien und stufen im wasser, das gruen ist total inhomogen. ich verstehe nicht wie du das darstellst und du es dann nicht siehst."** Der Befund war berechtigt und traf mein EIGENES `_meerestiefe_monoton()` von oben, nicht einen Altlast-Bug. Ring-/`grey_dilation`-Ausbreitung benutzt implizit eine Schachbrett-/Chebyshev-Metrik (ein 3x3-Strukturelement wiederholt angewandt) - diagonale Nachbarn werden dadurch systematisch anders behandelt als orthogonale, sichtbar als Strahlen/Stufen im offenen Meer. Gemessen (Kreuz- vs. Diagonal-Nachbar-Asymmetrie): Median 0.12 m, aber **p95 6.90 m, Maximum 69.4 m, 7.0 % der Seepixel ueber 5 m Abweichung** - klein im Mittel, aber genau die Ausreisser, die als sichtbare Linien auffallen. Die 0-Verstoesse-Verifikation der ersten Runde hatte nur MONOTONIE (nie eine Stufe nach oben) geprueft, nicht GLEICHMAESSIGKEIT (keine Richtungsabhaengigkeit) - beides war noetig, nur ersteres war getestet. **Nutzer-Vorgabe fuer den Neuentwurf:** "ALLE Meerestiefe wird jetzt einfach nurnoch ueber die Voronoi-Seetiefen Prozesse erzeugt... Wir speichern jetzt eine Stufe darunter, welcher Kuestentyp es ist. Dann kann die Seetiefe die negative Amplitude der Klippenhoehen sein (min. -10 m) und einem Seegradfaktor... Ganz einfach alles in einer Funktion." Umgesetzt als `_seetiefe_aus_archetyp(H, felder)`, die `_meerestiefe_monoton()` komplett ersetzt (Funktion entfernt) und gleichzeitig `_kuesten_umformen()` auf reine Landformung zurueckstutzt (`land_in_zone = d>=0.0` - das gesamte Unterwasser-Eintiefen aus 3.10/3.11 samt `UNTERWASSER_BASIS_M`/`UNTERWASSER_SPANNE_M` entfaellt ersatzlos, ersetzt durch `SEETIEFE_MINDEST_M=10.0`). Ablauf der neuen Funktion: (1) fuer jeden Hauptmeer-Pixel per `scipy.ndimage.distance_transform_edt(..., return_indices=True)` die AMPLITUDE des naechstgelegenen Kuesten-Archetyps holen (exakte euklidische Naechster-Punkt-Zuordnung, keine Ring-Iteration mehr noetig), (2) dieses Amplitudenfeld gaussgeglaettet (`sigma_px = max(2.0, 1200/mpp)`) gegen harte Spruenge an Archetyp-Zonengrenzen, (3) `tiefe = -max(SEETIEFE_MINDEST_M, amplitude) * seegrad_faktor`, wobei `seegrad_faktor` aus dem bereits vorhandenen, bereits korrekten `seegliederung()`-Feld `seegrad_tiefe` kommt (das die Verbindung durchs Wasser schon richtig behandelt - dafuer war keine eigene Geodaesie mehr noetig). Boden bei -10 m. **Ein Zwischenstand war beim ersten Versuch (Amplitudenfeld OHNE Glaettung) messbar SCHLECHTER als die alte Ring-Loesung** - p95 9.28 m, Maximum 342.1 m - vor dem Melden nachgemessen und dadurch aufgefangen, statt vorschnell als erledigt gemeldet. Mit der Gaussglaettung ergaenzt, erneut gemessen: **p95 0.40 m, Maximum 93.6 m, nur noch 1.37 % der Seepixel ueber 5 m** (vorher 7.0 %). Zusaetzlich diesmal AM BILD gegengeprueft (`meerestiefe_redesign.png`, direkt per Read-Tool angesehen, nicht nur die Kennzahl vertraut) - keine diagonalen Linien/Stufen mehr sichtbar, durchgehend glatter Tiefenverlauf inklusive an einem Meeresarm. Regressionsfrei (`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`, `smoke_test_terrain_scale_coupling.py`, `smoke_test_terrain_erosion_filter.py`, `smoke_test_river_reaches_sea.py`, `smoke_test_adaptive_terrain_mesh.py`). **LIVE BESTAETIGT 2026-08-13:** "Meeresboden sieht erstmal ok aus, moeglicherweise zu flach in vielen kuestenbereichen aber erstmal ausreichend bis alle meshfixes durch sind." - keine diagonalen Linien oder Stufen mehr gemeldet, der urspruengliche Befund ist damit weg. Der neue Vorbehalt (kuestennah zu flach) steht als eigener Punkt 3.14, damit er nicht in diesem langen Eintrag untergeht. |

| [ ] | 3.14 | **Zwei benannte Vorbehalte aus der Sichtpruefung 2026-08-13 — beide vom Nutzer ausdruecklich als "erstmal gut genug" eingestuft, hier nur festgehalten, damit sie nicht verlorengehen.** (a) **Kuestendetail reicht fuers fertige Spiel noch nicht:** "Kuestenarchetypen wechseln sich ab, fuers spiel am ende reicht es bestimmt noch nicht, da wir zu wenig detaillierung haben. also wie kommt man zum meer etc." - gemeint ist die begehbare Feinstruktur (Zugang zum Wasser, Strandabschnitte, Anlegestellen), nicht die grossraeumige Form, die jetzt sitzt. Haengt sinnvoll mit 13.x (Godot-Export) zusammen, weil dort entschieden wird, welches Detail ueberhaupt in die Engine geht. (b) **Meeresboden kuestennah moeglicherweise zu flach:** "moeglicherweise zu flach in vielen kuestenbereichen aber erstmal ausreichend bis alle meshfixes durch sind." Die Stellschraube dafuer steht bereits (`SEETIEFE_MINDEST_M` = 10 m und der `seegrad_faktor` in `_seetiefe_aus_archetyp()`, siehe 3.13) - es ist eine Eichungsfrage, kein Fehler. **Erst nach den Mesh-Aenderungen anfassen**, wie vom Nutzer vorgegeben. | 2 |

# 3b — Vektor-Kueste: was nach dem Einbau in die Main offen ist

*Stand 2026-08-23. Die Kuestenformung laeuft seit heute ueber
`core/vektor_kueste.py` statt ueber `_kuesten_umformen()`; der Mesh-Schnitt
ist in `map_display_3d._generate_terrain_mesh()` eingehaengt. Beides ueber
Schalter in `value_default.py` abschaltbar (`VEKTOR_KUESTE_AKTIV`,
`KUESTEN_SCHNITT_AKTIV`). Modellbeschreibung: `docs/spezifikation/11_GELAENDE.md`.*

| | # | Sache | Aufwand |
|---|---|---|---|
| [ ] | 3b.1 | **AM BILDSCHIRM ANSEHEN - der Blocker.** Nichts davon ist visuell bestaetigt: weder die neuen Kuestenprofile noch die geschnittene Kuestenlinie in 3D. GL laesst sich headless nicht pruefen (CLAUDE.md). Alles andere auf dieser Liste ist zweitrangig, bis das passiert ist. | 0.5 |
| [ ] | 3b.2 | **Nahtpruefung gerissen.** `smoke_test_regionen_welt.py` Pruefung 3: Regionsgrenzen 1.296 gegen 1.251 vorher, erlaubt 1.25. Die Kuestenformung wirkt an Regionsgrenzen offenbar staerker als im Regionsinneren - vermutlich, weil dort zwei Archetypsaetze mit verschiedenen Reichweiten aufeinandertreffen. **Gegenprobe zuerst:** mit `VEKTOR_KUESTE_AKTIV = False` messen, dann ist klar, ob es wirklich daher kommt. | 1 |
| [ ] | 3b.3 | **Mesh-Schnitt benutzt Rasterhoehen, nicht die Vektorhoehen.** `baue_schnitt_mesh()` wird ohne `hoehen_fn` gerufen, tastet also bilinear aus der Heightmap ab. Die KuestenLINIE ist damit rasterfrei, die KlippenSTEILHEIT aber weiter rastergedeckelt (gemessen: an freien Punkten p99 29-79 m mehr Hoehendetail). Dafuer muesste die `VektorKueste` aus `felder["vektor_kueste"]` bis ins Display durchgereicht werden. | 2 |
| [ ] | 3b.4 | **Dreieckszahl 2.9x bis 4.1x hoeher.** Der Schnitt geht vom vollen Gitter aus (512 px: 534046 gegen 129187 des Quadtrees). Bei 1024 px waeren das grob 2.1 Mio. Falls das ruckelt: Quadtree im Landesinneren, Schnitt nur im Kuestenband - beide Verfahren erzeugen aber verschiedene Vertexdichten, die Naht dazwischen ist genau die T-Stueck-Falle aus 6.20/6.33. | 3 |
| [ ] | 3b.5 | **Nachgelagerte Verbraucher ungeprueft.** Hydrologie, Biome, Siedlungseignung und Export lesen jetzt eine andere Heightmap (Median-Landhang 16.3 -> 13.7 Grad). Deren Eichungen koennen verschoben sein. `smoke_test_pipeline_outputs.py` und die Biom-/Siedlungssuiten laufen lassen. | 2 |
| [ ] | 3b.6 | **Bandweises Einblenden (e) nicht beurteilbar.** `BAND_EXPONENTEN` ist gebaut, aber auf dem Testgelaende ist die Wirkung zu klein zum Sehen (Band 0 hat 42 m Amplitude, Band 3 nur 0.9 m). In der echten Karte mit Erosionsfilter und Flussnetz neu beurteilen - dort gibt es mehr Feinstruktur. | 0.5 |
| [ ] | 3b.7 | **Schmale Landzungen bleiben schwach.** Im Inseltest ist `arme_3km` der schlechteste Fall: bei 288 m groesstem Kuestenabstand ist jeder Punkt Kuestenband, das Basisgelaende kommt nie durch. Loesung waere eine Tiefenklasse im Katalog (docs/spezifikation/11_GELAENDE.md Abschnitt 5) - Abschnitte mit wenig Hinterland bekommen Typen mit kurzer Reichweite. | 2 |
| [ ] | 3b.8 | **`_kuesten_umformen()` ist totes Gewicht**, solange der Schalter an ist - rund 200 Zeilen. NICHT loeschen, bevor 3b.1 und 3b.2 durch sind: es ist der Rueckfallpfad, wenn sich der Vektorweg als schlechter erweist. Danach entscheiden. | 0.5 |
| **[x]** | **3b.9** | **GEMESSEN 2026-08-24 - der Faktor bleibt konstant bei ~2x, er explodiert NICHT mit der Aufloesung:** 256 px 1.41 s gegen 0.65 s (2.18x), 384 px 2.13/1.02 (2.09x), 512 px 3.19/1.61 (1.98x), **1024 px 10.67 s gegen 5.39 s (1.98x)**. Die Hochrechnung unten ging von einem wachsenden Faktor aus und lag damit zu hoch. Ursprungstext: **Rechenzeit bei 1024 px ungemessen.** Bei 384 px kostet der Vektorweg 3.26 s gegen 1.25 s des Rasterwegs. Hochgerechnet waeren das bei 1024 px rund 20 s zusaetzlich - gegenueber den ~33 s des Terrain-Knotens spuerbar. Messen, bevor optimiert wird. | 0.5 |
| [ ] | 3b.10 | **Formvariation entlang der Kueste fehlt (b).** Die Profilform ist innerhalb eines Segments identisch, nur skaliert. Nutzerbild war *"als wuerde man eine Form ueber Sand ziehen und der Sand variiert kontinuierlich"*. Naechster Schritt waere ein langsam wandernder Mischungsanteil zwischen zwei Formen derselben Region. | 2 |

**Was gemessen und gruen ist** (damit es nicht erneut geprueft wird):
`smoke_test_vektor_kueste.py` 6/6, `smoke_test_kuesten_schnitt.py` 6/6,
`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`,
`smoke_test_river_reaches_sea.py`, `smoke_test_terrain_scale_coupling.py`.
Regionseichung: fuenf statt sechs Regionen daneben, drei neu im gruenen
Bereich (Macchia, Thalassia, Estrande).

# 4 — Erosionsfilter *(Nutzerwunsch 2026-08-07)*

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **4.1** | **Filter wieder mitlaufen lassen** bei aktiver Weltkarte — nach dem Weltfeld, vor dem Flussnetz. Eigene Weiche ohne Hoehennormierung. | 1 |
| **[x]** | **4.2** | **Regionsgewichtung**: Filterstaerke ueber `felder["relief_m"]` skaliert. Gemessen formt er in den Bergen 64.5 m um, in den Niederungen 17.6 — Faktor 3.7. | 1 |
| **[x]** | **4.3** | Die `erosion_filter_*`-Regler entsperrt (bis auf `octaves`, Nyquist). | 0.5 |
| **[x]** | **4.4** | **Talbreite hing am falschen Massstab** (2026-08-10). Sie war `faktor · formgroesse_m`, also 1200–5500 m, waehrend die Laeufe 111 m auseinanderstehen — das Eingraben wirkte als flaechige Glaettung und nahm dem Land 4.9 Grad Hang (Mittelmeerkueste 8.9 von 14.8). Haengt jetzt an `river_spacing_m`. | 1 |
| **[x]** | **4.5** | **Der Regionentest mass das Rohfeld**, nicht das fertige Gelaende — die Eichung galt fuer eine Welt, die niemand zu sehen bekam. Geht jetzt ueber `_calc_redistribution`; die neun Regionen wurden darauf neu geeicht. | 1 |
| **[~]** | **4.6** | **Naht**: alte Messung 1.268 gegen 0.945 im Inneren (erlaubt +25 %). **Nachgemessen 2026-08-11** (echte `region_map`-Grenzen, nicht das 3x3-Anzeigegitter, 512px): 99.-Perzentil-Verhaeltnis Grenze/Innen jetzt **1.209** - innerhalb der erlaubten 1.25, aber knapp. Max-basiertes Verhaeltnis sogar 0.852 (Grenze niedriger als das Innen-Maximum) - je nach verwendeter Kennzahl liegt der Befund zwischen "in Ordnung" und "gerade noch in Ordnung". Nicht eindeutig als weiterhin fehlerhaft bestaetigt, daher [~] statt [x] oder unveraendert [ ] - kein Fix vorgenommen, da unklar ob noch etwas zu fixen ist. Falls wieder eine sichtbare Kante gemeldet wird: mit `region_map`-Grenzpixeln (nicht dem Anzeigegitter!) und p99 statt Max messen, das war robuster als der urspruengliche Max-Vergleich. | 0.5 |
| **[x]** | **4.7** | **Fluesse erreichten das Meer oft nicht** (Nutzermeldung 2026-08-10, Vorbild [redblobgames](https://www.redblobgames.com/x/1723-procedural-river-growing/): vom Meer nach innen wachsen macht den Meeranschluss zur Bau-Eigenschaft statt zur Pruefpflicht). Gemessen VORHER bei 384 px: 29–47 % aller Netzknoten hingen in einem RING — "Ketten erzwingen" schrieb Eltern der Vorstufe zurueck, ohne auf Ringe zu pruefen. Dazu ein zweiter, kleinerer Fund: Inselpunkte, deren saemtliche Delaunay-Kanten ueber tiefes Wasser fuehrten, verloren jede Verbindung. Beides behoben — 100 % ueber 8 Kombinationen aus Groesse und Seed. Gesichert in `tests/smoke_test_river_reaches_sea.py`. | 1.5 |

| **[x]** | **4.8** | **Tektonische Platten — GEBAUT, GEMESSEN, BEWUSST NICHT ANGESCHLOSSEN (Exkurs 2026-08-17)** *(Nutzer-Vorgabe: "wir spawnen tektonische seeds ... diese wachsen an bis alle voronois zugewiesen sind. dann bewegen wir die platten ineinander und erzeugen damit falten fuer gebirge")*. Gebaut als `core/tektonik.py` (Poisson-Zellen -> Saaten mit Mindestabstand -> Wachstum per Breitensuche mit KONKURRIERENDEN Fronten -> Geschwindigkeitsvektor je Platte -> Konvergenz als Skalarprodukt mit der Grenznormalen -> Hebungsfeld mit Faltenzuegen) plus `tools/tektonik_labor.py` (fuenf Ansichten, acht Regler). Zellmaschinerie bewusst dieselbe wie `seegliederung()`. **Gemessen:** 0.28 s bei 512 px, bitgleich reproduzierbar, Plattenanteile identisch ueber 256/384/512 px (§10 erfuellt), Hebung 326-858 m ueber 20 Seed-/Plattenzahl-Kombinationen. **Ein echter Fehler dabei gefunden und behoben:** bei 2 Platten lieferten zwei von fuenf Seeds `Hebung max = 0 m` - die Platten liefen auseinander oder schrammten aneinander vorbei, die Karte bekam KEINEN Berg. Physikalisch ein gueltiger Grenztyp, als Generator der entartete Fall und damit genau die stille-Ruecklauf-Falle aus CLAUDE.md; `_bewegung_mit_gebirge()` wuerfelt jetzt deterministisch nach und meldet `konvergenz_gefunden=False`, statt es zu verschlucken. Faltenwellenlaenge auf Nutzerwunsch von 900 auf 3600 m vervierfacht ("wir deuten das alles ja nur an, auf 21 km brauchen wir weniger viel von allem") - danach fuegt die Tektonik dem Land nur noch 1.4 Grad Median-Hang hinzu statt 4.9, bei praktisch gleicher Gipfelhoehe. **Nutzerentscheidung 2026-08-17: "interessanter exkurs, aber wir brauchen erstmal keine tektonik."** `weltfeld()` ist unangetastet, die Module stehen ungenutzt und lauffaehig da. **Wieder aufgreifen, wenn die Welt eine KUGEL wird** - dann aber nach Nutzer-Vorgabe mit **Dekagon-/Pentagon-Zerlegung statt einem Gitter** (Goldberg-Polyeder/geodaetisches Netz), weil ein quadratisches Raster auf der Kugel an den Polen entartet. Die Plattenlogik selbst (Saat, Wachstum ueber Nachbarschaft, Konvergenz an der Grenze) ist von der Zellform unabhaengig und waere uebertragbar; nur `platten_zerlegung()`s Poisson/cKDTree-Teil muesste durch die Polyeder-Nachbarschaft ersetzt werden. | — |

**Nachtrag 2026-08-13, zweite unabhaengige Ursache gefunden:** die Kuesten-Reichweitenreparatur (3.10) liess `smoke_test_river_reaches_sea.py` erneut aufleben - 5 bzw. 8 Sackgassen von rund 3000 Knoten, 0 Ringe. Nachgemessen: scipys `dijkstra(..., return_predecessors=True)` markiert einen von der Ueberquelle aus NIE erreichten Knoten mit Sentinel `-9999`, nicht mit dem eigenen `-1` fuer den echten Meeresausgang. `eltern[eltern==n]=-1` fing nur den echten Fall ab - `-9999` rutschte seit dem Bau dieser Datei ungefiltert durch jede `eltern<0`-Pruefung (Kettenerzwingung, `netz['auslaesse']`, `taeler_eingraben`) und wurde als gueltiger Meeresausgang gezaehlt, obwohl der Knoten mitten im Land stand (gemessen: 63-306 m Hoehe). Ursache der Trennung: `_kanten_und_kosten()`s Bruecken-Reparatur haelt fuer einen isolierten Punkt nur EINE, die kuerzeste Kante - zwei Punkte, die einander GEGENSEITIG als kuerzeste Kante halten, bilden so eine eigene, vom Rest abgeschnittene Mini-Komponente. Behoben nach demselben Muster wie die bestehende Punkt-Bruecke, nur auf ganze Komponenten angewendet: jeder unerreichte Knoten wird an den naechstgelegenen ERREICHTEN Knoten gehaengt (echte Position, nicht ueber den Tiefwasser-gefilterten Kantensatz). **Ergebnis: 100% ueber alle 8 Kombinationen, 0 Sackgassen.** `core/terrain_weltfluesse.py`, `baue_stufe()`.

# 5 — Siedlungen und Wege *(Entwurf steht vollstaendig)*

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **5.1** | **Eignungsfeld aus fuenf Faktoren.** Vorher 3 Faktoren, zwei davon als Python-Doppelschleife. Jetzt Wasser (nach Typ: Grossfluss/See staerkstes Gewicht, dann Fluss, Bach, Meereskueste), ebener Grund, Ackerland im Umkreis (neu — traegt die GROESSE), Hoehe (echt daempfend statt Wohlfuehlzone in der lokalen Spanne), Erreichbarkeit (Platzhalter, haengt am noch fehlenden Wegenetz-Rueckschritt). Vollstaendig vektorisiert. Gesichert in `tests/smoke_test_settlement_placement.py`. | 2 |
| **[x]** | **5.2** | **Orte mit Kultur und Rang.** 2–5 je Kultur aus der Eignungssumme der Region (verglichen mit den anderen acht), garantiert eine Stadt je Kultur, Rang mit Rauschen statt starrer Schwelle, Haeuserzahl/Radius folgen dem Rang. Fund unterwegs: ein globaler statt kulturweiser Mindestabstand liess eine kleine Region ihr Ziel verfehlen (1 statt 2–5) — behoben. | 2 |
| **[x]** | **5.2b** | **Siedlungen klumpten** (Nutzer-Vorgabe 2026-08-10: "gleichmäßiger verteilen, etwas weniger geklumpt"). Ursache: die Eignungs-Absenkung nach jeder Platzierung wirkte nur innerhalb der harten Mindestabstands-Ausschlusszone — bei einer Region mit einem einzelnen Eignungshuegel blieb direkt daneben reichlich hohe Eignung uebrig, der naechste Ort setzte sich an dessen Rand statt in einen anderen Teil der Region ("Perlenkette" statt Streuung). Gemessen VORHER: mittleres Nachbarabstand-Verhaeltnis 0.76 (5 von 9 Kulturen unter 0.65). Die weiche Absenkung wirkt jetzt ueber das 2.5-fache des Mindestabstands, die harte Ausschlusszone selbst blieb unveraendert (Trefferquote 2-5 unveraendert getestet). Verhaeltnis danach 1.37. Gesichert in `tests/smoke_test_settlement_clustering.py`. | 1 |
| **[x]** | **5.3** | **Kostenfeld.** Ebener Grund 1.0, Hangkosten quadratisch (vorher linear), Wasser in drei Stufen (8×/25×/gesperrt unter -10 m), bestehender Weg 0.4× — "Wege buendeln sich". Einmal vektorisiert gebaut statt je A*-Schritt neu aus der Slope gerechnet. | 1 |
| **[x]** | **5.4** | **Gabriel-Graph + Bereitschaftstest + Kulturzusammenhang.** Ersetzt das bisherige Minimum-Spanning-Tree. Bereitschaft = Rang(A)·Rang(B)·(gleiche Kultur?1.0:0.45) gegen Wegkosten (Pfadkosten/Luftlinie) — zwei Staedte verbinden sich ueber einfaches Gelaende, zwei Doerfer verschiedener Kultur durch eine Wassersperre nicht; **dieselbe** Sperre zwischen zwei Orten DERSELBEN Kultur wird trotzdem ueberbrueckt ("egal was sie kostet"). Gesichert in `tests/smoke_test_settlement_roads.py`. | 2 |
| **[x]** | **5.11** | `settlement.outer_roads` entfernt — Knoten, Datenfelder, Graph-Eintrag, Zaehl-Assertion (39→<!-- PRUEFBAR: ausdruck=len(managers.calculator_graph.CALCULATOR_GRAPH) erwartet=38 -->38 Knoten). | 0.5 |
| **[x]** | **5.13** | `city_cost_map` benutzte `np.inf` als Sentinel — der gemeinsame Anzeige-Validator (`_validate_input_data`) haette JEDE kuenftige Darstellung dieses Feldes verweigert, ohne dass das beim Schreiben aufgefallen waere. Jetzt `-1.0`, spiegelbildlich zur bestehenden `city_mask`-Konvention. | 0.5 |
| **[x]** | **5.5** | **Seewege.** Eigenes Kostenfeld (Land gesperrt, Flachwasser teuer, tiefes Wasser billig) fuer Kulturpaare ohne endlichen Landweg. Auflage gesichert: mindestens die Haelfte der Laenge muss in echtem tiefen Wasser liegen, sonst waere es "kein Seeweg, sondern ein schlechter Landweg". Zwei Fehler unterwegs gefunden und behoben: (1) der A*-Fallback (Luftlinie bei nicht erreichtem Ziel) taeuschte guenstige Kosten vor, weil die Kostensumme nur Start/Ziel zaehlte statt der eigentlich unpassierbaren Zwischenzellen — dadurch wurden Landstrassen quer durchs gesperrte Wasser gebaut; (2) das Seekostenfeld sperrte auch die Siedlung selbst (steht ja auf Land), A* hatte dort nie einen Nachbarn. `sea_roads` als eigener Output (`settlement.pathfinding`); Anzeige folgte mit 5.8 (gestrichelt/blau im Global-Reiter). Gesichert in `tests/smoke_test_settlement_sites.py`. | 2 |
| **[x]** | **5.6** | **Kreuzungen + Roadsites (45 Arten).** Kreuzungen rein geometrisch aus dem gebauten Netz (Pixel, die zwei verschiedene Wege beruehren, Siedlungspunkte ausgenommen). Der komplette 9×5-Katalog aus `docs/spezifikation/14_SIEDLUNGEN.md` Abschnitt 9 eingetragen, jede Art einer Platzierungskategorie zugeordnet (furt/pass/kreuzung/strecke, aus dem Namen abgeleitet — im Entwurf selbst nicht ausgeschrieben). Platzierung bevorzugt Kreuzungen, dann Furten/Paesse, dann lange Zwischenstuecke; Typ kommt aus dem Katalog der naechstgelegenen Kultur. | 2 |
| **[x]** | **5.7** | **Landmarks (45 Arten).** Gleicher Katalogaufbau, vier Kategorien (gipfel/kueste/quelle/abgelegen). Fund unterwegs: die alte Fassung hatte eine pauschale Hoehen-Obergrenze (unterste 70 %), die GIPFEL-Arten ("Trutzburg auf dem Felskopf") von genau den Stellen ausschloss, die ihr eigener Name verlangt — jetzt vier verschiedene Feinauswahlen auf einer gemeinsamen Wildnis-Basis. | 2 |
| **[x]** | **5.8** | **Global-Reiter zeigt nur noch die weltkartentaugliche Uebersicht** (Nutzer-Vorgabe: "im globalen bereich ja nur städte, verbindungsstraßen und landmarks, roadsites, aber nur als punkte"). City Boundary und das PlotPhysicsSystem-Feingewebe (Kerne/Nodes/Kanten/Wildnisgrenze, 2D UND 3D-Skin) sind nach Regional gewandert. `roads`/`sea_roads` bekommen jetzt eine eigene Checkbox und werden als Linien gezeichnet (Landweg durchgezogen orange, Seeweg gestrichelt blau, 14_SIEDLUNGEN.md 5.3) — vorher unsichtbar. Fund unterwegs: `SettlementRegionalTab._overlays_zeichnen()` rief `overlay_plot_boundaries(kanten)` mit `plot_edges` im `plot_nodes`-Parameter auf — ein stets abstürzender Aufruf, lautlos von einem `except` verschluckt; das Plot-Feingewebe hat auf dem Regional-Reiter deshalb NIE gezeichnet. Behoben. | 2 |
| **[x]** | **5.9** | **Regional-Reiter auf 3×3-Box, mit ueberlappendem Rand** (Nutzer-Vorgabe 2026-08-10: "eine box um die insel und teilen das in maps auf und können diese durchwählen"). Der Reiter hatte bereits einen 9er-Regionswaehler, zoomte aber auf die BBox der weichen, verzogenen `region_map`-Zugehoerigkeit - organisch verformt, je Region unterschiedlich gross, mit nur `size/40` px Rand. Jetzt ein FESTER Kasten (`terrain_weltkarte.regionsbox_px()`, dieselbe Mathematik wie das gelbe Gitter aus 6.2) mit **25 % ueberlappendem Rand je Seite** (1.5× Kantenlaenge) - neun gleich grosse, durchblaetterbare Karten, ein Ort auf der Grenze erscheint mit Umland in beiden Nachbarkarten. | 1 |
| **[~]** | **5.10** | **Plot-Physik vereinfacht** (Nutzer-Vorgabe 2026-08-11: "sobald die Physik losgeht geht alles kaputt" - zwei Federn (core<->plotnode, plotnode<->plotnode) plus die Verkehrs-Kontraktion darauf ("dumme Idee, sehr komplex") und zu viele Kraefte insgesamt). `enable_core_plotnode_spring`/`enable_plotnode_plotnode_spring`/`enable_pressure` per Default auf `False` (`core/settlement_generator.py` `_load_default_parameters()` + `__init__`-Fallback) - die Verkehrs-Kontraktion (`_rest_length_plotnode_plotnode_batch()`) haengt ausschliesslich an der jetzt abgeschalteten plotnode<->plotnode-Feder und ist damit ebenfalls weg, ohne eigenen Codeeingriff. `enable_plot_node_repulsion` bleibt an ("einfach zurueck zur Abstossung der Nodes untereinander reicht"), ebenso Wildnis-/Stadtgrenzen-Eindaemmung und die Feldkraft. Bestehende Settlement-Testsuite regressionsfrei. **[~] statt [x]:** headless nur Kennzahlen (Naechster-Nachbar-Abstand, Iterationszahl) gepruefte, nicht das VISUELLE Ergebnis - braucht Bestaetigung im laufenden Programm. | 1 |
| **[x]** | **5.12** | **`roadsite_list` fiel bei 512 px aus, bei 128 px gefuellt.** Nach dem kompletten Roadsite-Umbau (5.6) nachgemessen: der urspruengliche Befund gilt nicht mehr (3 Roadsites bei 128 UND bei 512 px). Dabei aber ein reales PERFORMANCE-Problem in `_a_stern` gefunden: kein Closed-Set (abgelaufene Heap-Eintraege wurden voll neu expandiert) und `calculate_movement_cost`/`_heuristic` als Methodenaufrufe (10.48 Mio. / 1.39 Mio. Aufrufe, per cProfile gemessen). Ein zuerst versuchter zweistufiger Such-Ansatz (kleines Budget, nur bei Fehlschlag das volle) half NICHT (120.5 s statt 116.7 s — kein Unterschied). Die eigentliche Behebung war die Ueberarbeitung von `_a_stern` selbst: Closed-Set + Inlining beider Berechnungen. Isoliertes `settlement.pathfinding` bei 512 px: 39.8 s -> 9.7 s (4.1x). Pfade selbst unveraendert (`smoke_test_settlement_valley_routing.py`: identische Laengen/Hoehen vorher/nachher — A*s Optimalitaet haengt nicht vom Budget ab). Die zunaechst gemessenen "Gesamtketten"-Zeiten (116.7 s / 91.0 s bei 512 px) waren irrefuehrend: sie enthielten die GESAMTE Vorkette (Terrain/Wetter/Wasser/Biome), nicht nur `settlement.*`. Per-Schritt nachgemessen: alle `settlement.*`-Knoten zusammen nur 10.76 s bei 512 px (suitability 0.12, settlements 3.65, city_boundary 0.80, pathfinding 5.75, roadsites 0.44). Die restlichen ~77 s stammen aus `water.flow_network` (49.6 s), `weather.temperature` (12.3 s) und `terrain.redistribution` (11.8 s) — bereits vor diesem Block bestehend, ausserhalb des Siedlungs-Scopes, nicht weiter verfolgt. | 1 |
| **[x]** | **5.14** | **Orte weg von der 3×3-Kastengrenze — weiche Strafe, kein Verbot** (docs/TODO.md §D2 Vorschlag 2). Faktor 0.5 direkt auf der Gitterlinie, linear auf 1.0 ab 200 m Abstand (`_randfaktor()`, multipliziert auf `combined_suitability_map`) - fuer Siedlungen. Roadsites/Landmarks haben keine Eignungswerte, dort stattdessen eine stabile Randabstand-Umsortierung der Kandidaten (`_fern_zuerst()`): randferne zuerst, randnahe bleiben nachrangig verfuegbar statt ausgeschlossen - "entscheidet nur bei sonst gleichwertigen Plaetzen", genau wie im Entwurf gefordert. | 0.5 |
| **[x]** | **5.15** | **`roads` hat keinen eigenen Anzeigeweg mehr** — geloest mit 5.8: eigene "Roads"-Checkbox im Global-Reiter, Landweg durchgezogen orange, Seeweg gestrichelt blau. | 1 |
| **[x]** | **5.16** | **`settlement.settlements` skalierte 17.7x statt ~4x bei 256→512px** (Pipeline-Audit 2026-08-11, Nutzerfrage nach "stack deformation"-Performance). `_find_best_settlement_positions` baute bei JEDEM Platzierungsversuch die Sperrflaeche aus ALLEN bisherigen Siedlungen neu auf — je Siedlung ein volles (H,W)-Array. Jetzt inkrementell: `_markiere_gesperrt()` traegt nur eine lokal begrenzte Kreisscheibe je Siedlung ein (wie `_reduce_suitability_around_point`), die Sperrmaske wird nur bei Aenderung von `min_distance` (3x je Kultur) neu aufgebaut. Gemessen: 512px 3.37s → 0.12s (28x), 256px 0.23s → 0.04s (5.5x); Skalierung 256→512 jetzt 2.8x statt 17.5x. Ergebnisse bitidentisch (gleiche Kreisscheiben-Formel, nur lokal statt global berechnet) — komplette Settlement-Testsuite regressionsfrei. | 0.5 |

**Nebenbefund veraltet, siehe 5.4/5.8:** `pathfinding` liefert seit dem
Wegenetz-Umbau kein Bootstrap mehr, sondern das echte Gabriel-Graph/
Kostenfeld/Bereitschafts-Netz, und es ist seit 5.8 auch sichtbar. Der
PlotPhysicsSystem-Feinmesh (`plot_nodes`) bleibt ein DAVON UNABHAENGIGES,
intra-staedtisches System (siehe 5.10) - kein Ersatz fuer das inter-
Siedlungs-Wegenetz, wie der alte Nebenbefund unterstellte.

| **[~]** | **5.15** | **Siedlungen Regional zeigte gar nichts an — URSACHE GEFUNDEN UND BEHOBEN 2026-08-13; der Zuschnitt der neun Karten bleibt offen** *(Nutzerbefund: "Settlement Regional ist weiterhin nicht funktional. die Map zeigt nichts an.")*.

**Mein erster Verdacht war FALSCH und wurde verworfen, statt ihn zu bauen.** Vermutet hatte ich einen LOD-Groessenunterschied zwischen `region_map` und `heightmap`, der `_regionsgewichte()` still `None` zurueckgeben laesst. Nachgesehen: beide Karten stammen aus DEMSELBEN Calculator-Knoten (`terrain.redistribution`) und liegen damit immer auf demselben LOD - der Zweig kann gar nicht zuschlagen. Gut, dass das vor dem Umbau geprueft wurde.

**Die tatsaechliche Ursache, verifiziert:** `update_display_mode()` zeichnete die Basiskarte ueber eine eigene hasattr-Weiche

    if hasattr(ziel, "update_map_data"):     ziel.update_map_data(...)
    elif hasattr(ziel, "update_heightmap"):  ziel.update_heightmap(...)

`ziel` ist in der 2D-Ansicht ein `MapDisplay2D` - und das hat **weder das eine noch das andere**. Die einzige Zeichenmethode dort heisst `update_display(data, layer_type)`. `update_map_data` existiert im ganzen Projekt nirgends, `update_heightmap` nur auf `MapDisplay3D`. In 2D traf also KEINE der beiden Weichen zu, die Basiskarte wurde nie gezeichnet - und weil jede Overlay-Methode mit `if self.current_data is None: return` beginnt, kehrten danach auch samtliche Overlays sofort zurueck. Ergebnis: ein vollstaendig leerer Reiter, **ohne Absturz, ohne Logzeile, ohne fehlschlagenden Test**. In 3D lief es dagegen, weil `MapDisplay3D.update_heightmap` existiert - der Fehler war deshalb nur in einer der beiden Ansichten sichtbar. Maschinell nachgewiesen (`hasattr`-Ergebnis beider Klassen fuer alle drei Namen ausgegeben), nicht erschlossen.

**Behoben:** die eigene Weiche entfaellt, der Reiter benutzt jetzt `_push_data_to_current_display()` wie JEDER andere Reiter - der kennt beide Anzeigearten, geht ueber den `DisplayWrapper` (der die Fallback-Kette korrekt hat) und setzt nebenbei Schattenkarte, Sonnenstand und 3D-Layer-Sichtbarkeit. Die eigene Kopie hier war von Anfang an eine zweite, schlechter gepflegte Variante desselben Codes.

**Neuer Test gegen die ganze Fehlerklasse:** `tests/smoke_test_display_methoden_existieren.py` durchsucht alle `gui/tabs/*.py` nach `hasattr(<anzeige>, "name")` und prueft, dass jeder so gepruefte Name auf mindestens einer der Anzeigeklassen (`MapDisplay2D`/`MapDisplay3D`/`MapDisplay3DWidget`/`DisplayWrapper`) existiert. Ein Name, den keine kennt, ist immer ein Fehler - die Weiche trifft nie zu und der Code tut still gar nichts. **Gegenprobe gemacht:** den alten Aufruf testweise wieder eingebaut, der Test schlaegt zuverlaessig fehl und benennt Datei und Zeile - ohne diese Probe waere unklar, ob er ueberhaupt etwas kann. Beim ersten Lauf meldete er zwei weitere Stellen (`set_active` in base_tab.py); das war ein **Fehlalarm meines Tests**, weil ich den `DisplayWrapper` nicht zu den bekannten Klassen gezaehlt hatte - korrigiert, danach 34 gepruefte Aufrufe alle sauber.

**[~] statt [x]:** der leere Reiter ist behoben, aber der Zuschnitt der neun Karten ist eine getrennte, noch offene Vorgabe (siehe unten). Ausserdem ist die Behebung headless verifiziert, am Bildschirm noch nicht bestaetigt.

**NOCH OFFEN - VORGABE ZUM ZUSCHNITT (Nutzer 2026-08-13, mit Skizze):** *"nimm die noerdlichste Landerhebung, dann die suedlichste und dann die westlichste und die oestlichste. dann werden 9 karten aus dem bereich dazwischen gebildet (mit etwas meer drum herum)."* Also: Bounding-Box aller Landpixel (`H > 0`) plus Meeressaum, in 3x3 geteilt.

### Gemessene Bewertung (512 px, vier Seeds) — und eine Korrektur an mir selbst

**Ich hatte hier zuerst geschrieben, die aeusseren der neun Kaesten zeigten "ueberwiegend leeres Wasser". Das ist FALSCH und war ungemessen.** Nachgemessen zeigt jeder der neun heutigen Kaesten zwischen 34 % und 98 % Land. Der Grund: `gitter_kante_px()` liefert bei 512 px nur 96 px Zellkante, drei Zellen decken also 288 von 512 px ab — **das 3x3-Gitter liegt zentriert und umfasst nur 31.9 % der Kartenflaeche**, nicht die ganze Karte. Deshalb ist dort ueberall Kontinent.

**Der ECHTE Fehler ist ein anderer, und er ist gravierender:**

| | heutiges 3x3 | Land-Bounding-Box |
|---|---:|---:|
| abgedeckte Kartenflaeche | 31.9 % | 64.9 % (ohne Saum) |
| **Anteil des Landes, der in IRGENDEINER der 9 Karten liegt** | **71.1 %** | **100 %** |
| Land, das in KEINER Karte vorkommt | **28.9 % = 38.2 km²** | 0 |

Ueber vier Seeds stabil: 69–72 % Landabdeckung heute. **Rund ein Drittel des Kontinents ist heute in keiner einzigen Regionalkarte zu sehen** — inklusive der dort platzierten Siedlungen, Wege und Landmarken. Das ist der eigentliche Grund, die Vorgabe des Nutzers umzusetzen, und er ist staerker als der urspruenglich genannte.

### Vorteile der Bounding-Box

1. **Vollstaendigkeit** — alles Land ist in genau einer Karte (gemessen: 100 %, alle vier Seeds). Kein Dorf faellt mehr durchs Raster.
2. **Groessere Karten bei gleicher Anzahl** — Zelle waechst von 4.0 km auf 6.2–6.9 km Kante, also rund 2.7-fache Flaeche je Karte. Fuer eine Spielkarte ist das naeher an einer brauchbaren Groesse.
3. **Passt sich der Welt an** — ein kompakter Kontinent ergibt kleinere, ein zerfaserter groessere Karten, ohne dass jemand nachstellt.
4. **Sehr kleiner Eingriff** — `regionsbox_px()`/`gitter_kante_px()` sind bereits die EINE Quelle fuer Anzeige, Zuschnitt und Platzierungs-Randstrafe (5.9/5.14). Es genuegt, sie aus der Landmaske statt aus `size` abzuleiten.

### Nachteile, ehrlich benannt

1. **Manche Karten werden fast leer.** Gemessen bei quadratischer Box mit 1 km Saum: je Seed **1–2 Karten unter 10 % Land, bei zwei Seeds sogar exakt 0 %** (reine Meer-Ecken). Das heutige Gitter hat dieses Problem nicht (Minimum 33.9 %) — es erkauft sich das aber damit, dass es ein Drittel des Landes weglaesst. **Der Nachteil ist also nicht neu entstanden, er wird nur sichtbar**: die leeren Ecken sind genau die Stellen, die heute stillschweigend weggelassen werden.
2. **Die Box ist nicht quadratisch** (gemessen 15.8 x 18.7 km, Verhaeltnis 0.844). Bei direkter Drittelung waeren die Karten rechteckig (5.26 x 6.23 km) — fuer die 2D-Anzeige egal, fuer Terrain3D-Regionen (quadratisches Raster) unguenstig. **Gegenmittel:** die laengere Seite nehmen und quadrieren, um den Mittelpunkt zentriert. Kostet etwas mehr Wasser, erhaelt aber quadratische Karten. Gemessen: 18.7–20.7 km Seite, Zelle 6.24–6.89 km.
3. **Die Box haengt am Seed** — jede Welt bekommt eine andere Kartengroesse. Fuer den Editor unproblematisch, fuer einen Export mit fester Aufloesung heisst es: entweder variable m/px oder variable Pixelzahl. Siehe Godot unten.
4. **Ein einzelner Felsen im Ozean zieht die Box auf.** `H > 0` nimmt jeden einzelnen Pixel mit. Sinnvoller ist ein Perzentil oder eine Mindest-Inselgroesse (z. B. nur Zusammenhangskomponenten ueber ~50 Pixel), sonst bestimmt eine 3-Pixel-Klippe die Groesse aller neun Karten.

### Wie es fuer Godot funktionieren wuerde — und die eine offene Grundsatzfrage

**Terrain3D verwaltet eine Welt als Raster gleich grosser, quadratischer Regionen** (Standard 1024 Vertices Kante). Daraus folgt eine Frage, die den ganzen Zuschnitt entscheidet und die ich nicht selbst beantworten kann:

* **(A) Die neun Karten sind neun Kacheln EINER zusammenhaengenden Welt.** Dann ist der 3x3-Schnitt eine reine **Editor-Ansicht** und gar keine Exportgrenze: exportiert wird die ganze Welt als 3x3 Terrain3D-Regionen, die Uebergaenge sind nahtlos, der Spieler laeuft einfach hinueber. Der Zuschnitt muss dann nur "schoen anzusehen" sein, nicht spielmechanisch sauber — und die leeren Meer-Ecken sind voellig unkritisch, weil man dort ohnehin ins Wasser laeuft.
* **(B) Die neun Karten sind neun GETRENNTE Spielkarten** (Ladebildschirm dazwischen, jede eine eigene Welt). Dann ist der Schnitt echt, jede Karte braucht einen sinnvollen Rand (Meer oder Gebirge als natuerliche Begrenzung), eine fast leere Meer-Ecke ist eine **unbrauchbare Spielkarte**, und genau dafuer ist die Vielecke-Idee des Nutzers (unten) die richtige Antwort.

**Rechnerisch passt es in beiden Faellen gut:** bei Zellkante 6.9 km und 1024 px je Karte ergaeben sich **6.7 m/px** — feiner als die in 13.1 beschlossenen 10.4 m/px fuer die ganze Welt, und 1024 ist genau die Terrain3D-Standardregionsgroesse. Der Zuschnitt verbessert die Exportaufloesung also nebenbei, statt sie zu kosten.

### Mein Vorschlag zur Umsetzung

**Schritt 1 (klein, sofort, unabhaengig von A/B):** `gitter_kante_px()`/`regionsbox_px()` aus der Landmaske ableiten statt aus `size`. Konkret eine neue Funktion `landbox_px(H, saum_km, min_insel_px)` in `terrain_weltkarte.py`, die (a) Zusammenhangskomponenten unter `min_insel_px` ignoriert (Nachteil 4), (b) die Bounding-Box bildet, (c) auf die laengere Seite quadriert (Nachteil 2), (d) den Saum addiert und auf die Karte klemmt. Die beiden vorhandenen Funktionen lesen daraus. Weil sie bereits die einzige Quelle fuer Anzeige, Zuschnitt UND Platzierungs-Randstrafe sind, wirkt das ueberall zugleich — genau ein Eingriffspunkt. Absicherung: `smoke_test_settlement_region_grid.py` erweitern um "100 % des Landes liegt in mindestens einer der neun Boxen", gemessen ueber mehrere Seeds, sowie "keine Siedlung ausserhalb aller Boxen".

**Schritt 2 (nur wenn Fall B):** die Vielecke-Idee. Erst dann lohnt der Aufwand.

**Was ich NICHT tun wuerde:** die leeren Meer-Ecken durch Verschieben einzelner Karten "auffuellen". Das bricht die Gleichmaessigkeit des Rasters, macht die Zuordnung Region->Karte mehrdeutig und ist genau der Sonderfall-Pfad, der spaeter niemandem mehr erklaerbar ist. Entweder gleichmaessiges Raster (Schritt 1) oder ehrliche Vielecke (Schritt 2).

### Nutzerentscheidung 2026-08-13: Fall B, mit Vielecken — und die Performance-Rueckfrage

> **NACHTRAG, noch am selben Tag: die Grundsatzfrage ist inzwischen anders beantwortet — siehe 13.0.** Der Nutzer hat vorgeschlagen, in Godot IMMER dieselbe Terrain3D-Welt zu laden und nur die Details (Vegetation, Objekte) je aktiver Zone zu streamen, mit FoW ueber den Aussenbezirken. Damit sind die neun Vielecke keine Terrain-Schnittgrenzen mehr, sondern Streaming-/FoW-Zonen. Der Abschnitt hier bleibt stehen, weil die Abwaegung und die Zahlen weiterhin gelten - nur die Schlussfolgerung "neun getrennte Kartensaetze exportieren" ist ueberholt.

Wortlaut: *"grundsaetzlich erstmal nummer 2. 9 spielkarten, unterschiedlicher form (wir haben zwar eine rechteckige karte aber die eigentliche spielwelt ist ein vieleck, bis zu 6 ecken, und die fehlenden flaechen sind schwarz und vernebelt. sogesehen ist die flaeche jeweils etwa 4x4 km aber so ausgewaehlt das die welt nur dort zerschnitten wird, wo es sinn macht (berggrad, buchten, steile kuesten wo wenig siedlung und landmarks etc passieren."* Rueckfrage des Nutzers: *"eine nahtlose weltkarte waere natuerlich schoen, aber die frage ist: bekommen wir das performant hin?"*

**Die 4x4 km sind rechnerisch genau richtig** (gemessen): Welt 454 km², davon Land 132 km². Neun Karten a 16 km² = 144 km², decken das Land also 1.1-fach ab. Die HEUTIGE Zellkante ist bereits exakt 4.0 km — die Vorstellung des Nutzers und der bestehende Rasterabstand stimmen ueberein, es fehlt nur die richtige LAGE der Kaesten (siehe die 28.9 % nicht abgedeckten Landes oben).

**Zur Performance-Frage, mit Zahlen, soweit ich sie seriös liefern kann:**

| Variante | Aufloesung | Datenmenge (Height+Control+Color) |
|---|---:|---:|
| ganze Welt nahtlos | 10.4 m/px (13.1) | 48 MB |
| ganze Welt nahtlos | 4 m/px | 325 MB |
| ganze Welt nahtlos | 2 m/px | 1.3 GB |
| 9 Karten a 4x4 km | 4 m/px | 103 MB (11 MB je Karte) |
| 9 Karten a 4x4 km | 2 m/px | 412 MB (46 MB je Karte) |

**Das staerkste Argument fuer die neun Karten ist gar nicht die Bildrate, sondern die Verschwendung:** die nahtlose Welt speichert 454 km², davon **322 km² reines Meer**, in voller Aufloesung. Die neun Karten speichern nur die 144 km², auf denen ueberhaupt gespielt wird. Bei 4 m/px sind das 325 MB gegen 103 MB — Faktor 3, ohne dass ein einziger Spielinhalt verlorengeht.

**Was ich zur Bildrate ehrlich sagen kann und was nicht:** unser Land (132 km²) liegt in der Groessenordnung von Witcher 3 Velen+Novigrad (~136 km²), die Welt gesamt (454 km²) deutlich darueber — aber der ueberwiegende Teil davon ist Wasser, das keine Geometrie braucht. Terrain3D bringt Clipmap-LOD und regionsweises Streaming mit, das Terrain selbst ist bei diesen Datenmengen erfahrungsgemaess **nicht** der Engpass; die Grenze setzen in offenen Welten fast immer Vegetation/Streuobjekte (siehe 13.4), Texturspeicher und Simulation. **Belegen kann ich das hier aber nicht** — das ist eine Godot-Frage, und dieses Projekt hat keine Godot-Seite, an der ich messen koennte. **Die Frage ist in etwa einer Stunde definitiv beantwortbar:** eine flache 21-km-Testheightmap in Terrain3D laden, mit der Kamera durchfliegen, Bildrate und Speicher ablesen. Ohne diesen Versuch waere jede Zusage von mir geraten.

**Wichtigster Architekturhinweis: die Entscheidung muss beim DATENEXPORT gar nicht fallen.** Vieleck-Zerschneidung und Nahtlosigkeit schliessen sich nicht aus:
* Die Hoehen-/Farb-/Controldaten koennen **nahtlos** ueber die ganze Welt exportiert werden (ein Koordinatensystem, keine Naht, keine doppelten Randpixel).
* Die neun Vielecke kommen als **eigene Maskendatei** dazu (Polygon-Punktlisten, analog zu 13.5).
* In Godot entscheidet dann die Spiellogik, ob die Vieleckgrenze **hart** ist (Ladebildschirm, Rest schwarz+vernebelt wie vom Nutzer beschrieben) oder **weich** (man laeuft hinueber). Beides liest dieselben Daten.

Damit ist die Performance-Frage **entkoppelt**: sie muss beantwortet werden, bevor das SPIEL gebaut wird, aber nicht, bevor der Export gebaut wird. Falls der Godot-Versuch zeigt, dass nahtlos gut laeuft, kostet der Wechsel dann nichts weiter als das Weglassen der harten Grenze.

**Wie die "sinnvollen" Schnittlinien technisch entstehen wuerden** (Nutzer: *"nur dort zerschnitten wird, wo es sinn macht (berggrad, buchten, steile kuesten wo wenig siedlung und landmarks etc passieren"*) — dafuer hat das Projekt die Maschinerie bereits: **es ist dasselbe Kostenfeld-plus-guenstigster-Weg-Verfahren wie beim Wegenetz (§5.4), nur mit umgekehrtem Vorzeichen.** Wo eine Strasse teuer ist, ist ein Schnitt billig:
* billig zu schneiden: Meer, Seegrad hoch, steile Hangneigung, Berggrat (`ridge_map` liegt vor), Buchten
* teuer zu schneiden: Naehe zu Siedlungen/Landmarks/Roadsites, bestehende Wege, flaches bewohnbares Land, Fluesse laengs
Aus neun Startpunkten (den Regionsschwerpunkten) ein Voronoi ueber diese Kosten — also eine **kostengewichtete Zerlegung** statt eines geraden Rasters — ergibt genau die Vielecke der Skizze: Grenzen, die Meeresarmen und Graten folgen. Die Begrenzung auf "bis zu 6 Ecken" waere danach eine Vereinfachung des Polygonzugs (Douglas-Peucker o. ae.). **Aufwand dafuer ist die eigentliche Arbeit an diesem Punkt**, nicht die Bounding-Box.

**Reihenfolge daher unveraendert:** Schritt 1 (Bounding-Box, klein, sofort, in jedem Fall richtig), danach die Vielecke als eigener Schritt. Der Godot-Bildratenversuch kann parallel und unabhaengig laufen.

### UMGESETZT 2026-08-13: `core/spielkarten.py` — die Vielecke stehen (Bounding-Box uebersprungen)

Der Nutzer hat direkt die Vieleck-Zerlegung beauftragt, mit diesen Vorgaben: *"erstmal in quasi voronoi-vieleck-Karten mit jeweils gleich viel Landmasse (der erste seegrad an kueste zaehlt auch als 50% landmasse, da hier viel passiert). ich moechte das die karten optimiert werden um auf einem quadrat gezeigt werden zu koennen, also nicht unnoetig lang, und dennoch logische zusammenhaenge enthalten ... auch regionen sollten wenn moeglich zusammenbleiben ... die schnittlinien sollten irgendwo zwischen den staedten verlaufen. die karten koennen bis zu 50% unterschiedlich gross sein."* Schritt 1 (reine Bounding-Box) entfaellt damit — die Zerlegung deckt ohnehin die gesamte Landmasse ab.

**Gewaehltes Verfahren: Power-Diagramm (gewichtetes Voronoi) mit Kapazitaetsausgleich.** Die Begruendung steht ausfuehrlich im Modul-Docstring; der Kern: die Vorgaben "nicht unnoetig lang" und "logische Zusammenhaenge" koennen sich widersprechen. Ein kostenbasiertes Verfahren (guenstigste Wege durch ein Gelaendekostenfeld, wie beim Wegenetz) erfuellt die Zusammenhaenge gut, kann aber beliebig krumme Baender erzeugen. **Ein Power-Diagramm garantiert dagegen konvexe Zellen mit geraden Kanten** — also automatisch kompakt, quadratisch darstellbar und ein Vieleck mit wenigen Ecken. Die Zusammenhaenge kommen stattdessen ueber die LAGE DER SAATPUNKTE: liegen sie auf den Siedlungsschwerpunkten, verlaeuft jede Grenze per Voronoi-Definition mittig zwischen zwei Siedlungsclustern — genau die Vorgabe "Schnittlinien zwischen den Staedten". Der Kapazitaetsausgleich laeuft ueber additive Gewichte `lambda` und NICHT ueber Verschieben der Saatpunkte (Lloyd), weil ein Verschieben die Saatpunkte von den Staedten wegziehen und damit die Schnittlinien-Vorgabe wieder aufgeben wuerde.

**Drei Fehler beim Bauen gefunden, alle durch Messen und nicht durch Hinsehen:**

1. **Die Massenbilanz sah ausgeglichen aus, waehrend das echte Land um Faktor 2 auseinanderlag.** Erste Messung: Massenspanne 1.35 (im Ziel), aber Landspanne **2.09** — eine Karte hatte 9.0 km² Land gegen 18.8 km² der groessten. Ursache: der k-Means fuer die Saatpunkte lief ueber das GESAMTE Gewichtsfeld inklusive der halb zaehlenden Kuestensee, ein Saatpunkt landete dadurch draussen zwischen den Inseln. Die Kuestensee ist auf dieser Welt fast halb so gross wie das Land (61.9 gegen 132.1 km²) und kann eine Karte muehelos "auffuellen". Behoben: Saatpunkte werden nur ueber Vollgewicht-Pixel (echtes Land) gesetzt. **Seither wird die Landspanne als eigene Kennzahl neben der Massenspanne gefuehrt** — ohne sie war der Fehler unsichtbar.
2. **Die lambda-Iteration divergierte bei einem Seed.** Seed 12345 schaukelte sich auf eine Spanne von **19.1** auf, also weit schlechter als der Startzustand — und genau dieser Ausreisser wurde zurueckgegeben, weil die erste Fassung das LETZTE statt des besten Ergebnisses nahm. Drei Aenderungen: Lernrate faellt mit der Rundenzahl, `lambda` wird zentriert (nur die Differenzen bestimmen die Grenzen, ein gemeinsamer Offset laesst die Werte nur davondriften), und zurueckgegeben wird das beste je gesehene Ergebnis.
3. **Der Ausgleich hatte keinen Grund mehr, das Land anzufassen.** Nach Fix 1+2 zeigte Seed 777001 eine Massenspanne von **1.00** (perfekt) bei einer Landspanne von **2.11** — sobald die Masse ausgeglichen war, verschwand der Gradient. Behoben, indem die Fehlerfunktion Masse UND Land mittelt statt nur die Masse.

**Ergebnis, gemessen ueber 8 Seeds (512 px):** alle acht innerhalb der Vorgabe, Massenspanne 1.09–1.49, Landspanne 1.29–1.48 (Ziel jeweils <= 1.5, entspricht dem "bis zu 50% unterschiedlich gross"). Groesstes Seitenverhaeltnis einer Karten-Bounding-Box 2.11, Median um 1.4 — also kompakt genug fuer eine quadratische Darstellung. Laufzeit **0.4–1.2 s bei 512 px, 1.0 s bei 1024 px** (vor Fix 2 waren es 8–12 s, weil die Iteration bis zum Rundenlimit lief). Regionen je Karte im Mittel 3.6–3.9 von 9.

**Sichtgeprueft** (drei Seeds nebeneinander gerendert und angesehen, nicht nur Kennzahlen): Halbinseln bleiben als Einheit zusammen — beim Beispielseed liegen die Nord-, West- und Suedhalbinsel jeweils vollstaendig in einer Karte, genau wie in der Nutzerskizze. Inselgruppen (Thalassia) bleiben naturgemaess zerstueckelt; das ist die Welt, nicht der Schnitt.

**Ehrlich offen:**
* **Der Kern-Anteil** (groesstes zusammenhaengendes Stueck je Karte) liegt zwischen 0.36 und 0.76. Karten mit vielen kleinen Inseln zerfallen also in mehrere Teile. Das laesst sich mit konvexen Zellen nicht beheben — es waere ein Argument fuer ein kostenbasiertes Verfahren, mit dem oben benannten Preis (krumme, langgezogene Formen).
* **Die Saatpunkte laufen bisher immer ueber den Rueckfall** (k-Means ueber die Landmasse), weil die Zerlegung noch nicht an die Siedlungsdaten angeschlossen ist. Der Siedlungspfad ist gebaut und getestet, aber im Programm noch nicht verdrahtet — **damit ist die Vorgabe "Schnittlinien zwischen den Staedten" noch NICHT erfuellt**, nur vorbereitet. Der Rueckfall meldet sich ueber `aus_siedlungen=False` statt still zu greifen (CLAUDE.md).
* **Keine GUI-Anbindung.** `regionsbox_px()`/`gitterlinien_px()` und der Regional-Reiter nutzen weiterhin das starre 3x3-Raster; die Zerlegung ist bisher nur als Bibliothek samt Test vorhanden.

Gesichert in `tests/smoke_test_spielkarten.py` (Gewichtsfeld inkl. der 50-%-Regel, Saatpunkt-Herkunft samt meldendem Rueckfall, Determinismus, alle vier Vorgaben ueber vier Seeds).

### IN DIE PIPELINE EINGEBRACHT 2026-08-13 (Nutzer: "kannst du es in die main einbringen und die regionen ansicht?")

`spielkarte` ist jetzt ein regulaerer Terrain-Output und laeuft denselben Weg wie `region_map`/`seegrad`/`kuesten_archetyp`:

* **Berechnung** in `_weltkarte_spielkarten()` (`core/terrain_generator.py`), aufgerufen aus `_weltkarte_heightmap()`. Bewusst DORT und nicht als eigener Calculator-Knoten: die Zerlegung braucht genau die zwei Felder, die an dieser Stelle frisch vorliegen (fertige Heightmap und `seegrad`) - ein eigener Knoten muesste beide erneut anfordern. Kosten **0.4 s bei 512 px / rund 1 s bei 1024 px**, gegenueber den ~33 s dieses Knotens (siehe 7.12) vernachlaessigbar.
* **Durchgereicht** ueber `TerrainData.spielkarte`, die Schluesselliste in `set_terrain_data_complete_lod()` und `CalculatorSpec("terrain.redistribution").output_keys` - alle drei Stellen, die auch die uebrigen Weltkarten-Felder auffuehren.
* **Fehlschlag meldet sich laut**: schlaegt die Zerlegung fehl, gibt es eine WARNING mit Grund und die Regionalansicht faellt sichtbar auf das alte 3x3-Raster zurueck. Ein stiller Rueckfall waere von Erfolg nicht zu unterscheiden (CLAUDE.md) - headless gegengeprueft, indem eine landlose Karte hineingegeben wurde.

**Anzeige:**
* **Terrain-Reiter, neuer Radioknopf "Spielkarten"** - neun Farben (bewusst `tab10` und NICHT die Regionsfarben: Spielkarten und Kulturregionen sind beide neun an der Zahl, decken sich aber nicht, gemessen 3.6 Regionen je Karte - gleiche Farben wuerden nahelegen, Karte 3 sei Region 3), weisse Grenzen, Kartennummer in der Mitte. Meer nur schwach getoent, damit die Kuestenlinie sichtbar bleibt.
* **Siedlungen (Regional) zoomt jetzt auf das Vieleck** statt auf den starren Kasten (`_spielkarten_kasten()`), quadratisch aufgezogen mit 8 % Rand - quadratisch, weil der Anzeigebereich es ist und der Massstab auf allen neun Karten gleich sein soll. Die Radioknoepfe heissen deshalb jetzt **"Karte 1..9" statt der Regionsnamen** - die alten Namen waeren schlicht falsch beschriftet gewesen. Welche Regionen tatsaechlich auf einer Karte liegen, sagt jetzt die Statistik daneben (neue `_dominante_regionen()`, nennt die groessten drei mit Flaechenanteil). Auch das ausgewertete Gebiet der Statistik folgt jetzt der Spielkarte statt der Kulturregion.

**Ein Fehler dabei gefunden und behoben:** `MapDisplay2D.update_display()` packt dict-Payloads mit `data["regionen"]` aus - fuer den neuen `spielkarte`-Payload warf das einen **KeyError**, der Reiter waere sofort gescheitert. Jetzt werden `regionen`/`spielkarte`/`heightmap` der Reihe nach probiert. Gefunden, weil der Renderer nach dem Bauen tatsaechlich einmal durchlaufen wurde statt nur der Code gelesen.

**Verifiziert:** Renderer zeichnet 3 Bildebenen + 9 Kartennummern; fehlende Zerlegung faellt sauber auf die Heightmap zurueck ohne Absturz; `region_map` unveraendert; die volle Durchreichkette maschinell geprueft (TerrainData-Feld, Manager-Liste, `output_keys`). Regressionsfrei (`smoke_test_display_2d.py` alle 30 Darstellungen, `smoke_test_layer_2d_3d_parity.py`, `smoke_test_spielkarten.py`, `smoke_test_display_methoden_existieren.py`, `smoke_test_settlement_sites.py`). Sichtgeprueft am gerenderten Bild.

**Nachgezogen 2026-08-13, weil sonst zwei Ansichten widersprochen haetten:** das gelbe Gitter (`overlay_region_grid()`) zeichnete weiterhin das gerade 3x3-Raster, waehrend der Regionalreiter bereits auf die Vielecke zoomte - **zwei verschiedene Neunerteilungen nebeneinander**, also genau die Verwechslung, die der urspruengliche D2-Hinweis ("man liest zwei verschiedene Neunerteilungen als eine") vermeiden wollte. Es zeichnet jetzt die tatsaechlichen Vieleck-Grenzen, wenn die Zerlegung vorliegt, und faellt sonst auf das gerade Raster zurueck (auch bei nicht passender Kartengroesse). Beide Aufrufstellen (Siedlungen Global und Regional) reichen `spielkarte` durch. Verifiziert: mit Zerlegung eine Bildebene und KEINE Geraden, ohne Zerlegung acht Geraden, bei falscher Groesse ebenfalls Rueckfall ohne Absturz.

**Bewusst NICHT gemacht (Nutzer: "die feinheiten machen wir spaeter"):**
* **Keine 3D-Darstellung der Spielkarten.** `_LAYER_NAME_MAP_3D` hat fuer `spielkarte` keinen Eintrag; im 3D-Modus zeigt dieser Radioknopf daher das nackte Gelaende. Der Weg dafuer steht (RGBA-Skin wie Regionen/Kuestentypen, siehe 6.1/6.23), es ist reine Fleissarbeit.
* **Die Siedlungsplatzierung kennt die Vielecke noch nicht.** `settlement_generator._randfaktor` und das gelbe Gitter (`overlay_region_grid`, `gitterlinien_px()`) haengen weiter am starren 3x3-Raster. Solange beides nebeneinander besteht, koennen Anzeige und Platzierung auseinanderlaufen - **das ist die wichtigste offene Stelle** und gehoert in den vom Nutzer angekuendigten Siedlungs-Umbau.
* **Schnittlinien zwischen den Staedten** (Nutzer: "ist mit dem settlement-update verwandt und da gehen wir spaeter drauf ein, denn ich habe schon ideen") - die Zerlegung laeuft weiter ueber den Rueckfall-Saatpunktweg.

**NEUE VORGABE ZUM ZUSCHNITT (Nutzer 2026-08-13, mit Skizze):** die neun Regionalkarten sollen nicht mehr aus dem starren 3x3-Gitter ueber die GANZE Karte kommen, sondern aus dem tatsaechlich bewohnbaren Bereich: *"nimm die noerdlichste Landerhebung, dann die suedlichste und dann die westlichste und die oestlichste. dann werden 9 karten aus dem bereich dazwischen gebildet (mit etwas meer drum herum)."* Also: Bounding-Box aller Landpixel (`H > 0`) plus ein Meeressaum, und DIESE Box wird in 3x3 geteilt - nicht die volle Weltkarte. Auf der aktuellen Karte liegt viel Ozean am Rand, die aeusseren der neun Kaesten zeigen deshalb heute ueberwiegend leeres Wasser.

**Und der Nutzer benennt selbst, dass das nur die Zwischenloesung ist:** *"Kann sein das dann die Karte etwas bloed zerschnitten wird, aber damit muessen wir leben bis wir eine idee haben wie wir das loesen."* Die Idee dazu steht schon (Skizze im Bild): *"zb koennten wir die welt in sinnvolle vielecke teilen, also bis zu 6 kanten oder so und dann die karten mit unterschiedlicher groesse generieren und die dreiecke die abgeschnitten sind werden einfach schwarz dargestellt."* In der Skizze verlaufen die Trennlinien NICHT gerade durchs Land, sondern folgen Meeresarmen und Taelern - eine Karte endet also dort, wo ohnehin eine natuerliche Grenze liegt, statt mitten durch eine Siedlungslandschaft zu schneiden. **Bewertung ohne Umsetzung:** das ist deutlich mehr Arbeit als die Bounding-Box (Zellen unterschiedlicher Groesse, nicht-rechteckige Ausschnitte, schwarze Fuellung ausserhalb des Vielecks, und fuer 3D ein entsprechend zugeschnittenes Mesh, siehe unten) - aber es ist die Loesung, die am Ende gebraucht wird, wenn die neun Karten wirklich als einzelne Spielkarten dienen sollen. **Vorschlag zur Reihenfolge:** erst die Bounding-Box (klein, sofort sichtbar besser), das Vieleck-Verfahren als eigener spaeterer Schritt.

**Fuer 3D zusaetzlich (Nutzer):** *"wir brauchen fuer den 3D teil spaeter ein zerschnittenes mesh, so dass wir nur das mesh sehen von der einzelnen karte."* Heute zoomt der Regionalreiter nur die ACHSEN, die Daten bleiben absichtlich ungeschnitten (siehe Modul-Docstring - ein Zuschnitt haette alle Koordinaten verschoben). Fuer 3D reicht Achsen-Zoomen aber nicht, dort muesste das Mesh selbst begrenzt werden. Machbar ueber den bestehenden adaptiven Mesh-Bau (Blaetter ausserhalb des Ausschnitts weglassen), aber ein eigener Schritt.

**Der Nutzer will Siedlungen ohnehin komplett neu durchgehen** (*"Aber Settlements muessen wir bald komplett neu durchgehen. alles reparieren."*) - dieser Punkt ist der erste Anlass dafuer, aber vermutlich nicht der einzige. | 4 |

| [ ] | 5.16 | **Siedlungs-Umbau: Stadttypen, Handelsgewichte, Reihenfolge — Nutzerentwurf 2026-08-13, HIER BEWERTET, noch nicht umgesetzt.** Ausgangslage vom Nutzer: *"Die Siedlungen sind ziemlich gut gesetzt worden bisher. Die Strassen und Roadsites und Landmarks sind schlecht."* Der volle Entwurf steht in der Sitzungshistorie; hier die Bewertung mit den gemessenen Zahlen, weil eine Praemisse darin nicht stimmt.

### Zuerst eine Korrektur: es gibt KEINE Voronois in der Wegfindung

Der Nutzer vermutete: *"ich weiss nicht wie die rechnung derzeit verlaeuft, aber 1000 voronois zur verbesserung der ladezeiten haben wir oder sowas? damit ein Dijkstra verfahren schneller geht."* **Nachgesehen: nein.** Der Ist-Zustand ist:

* **Wegfindung: A\* auf dem vollen Pixelgitter** (`PathfindingSystem`, Kostenfeld aus `bau_kostenfeld()`), nicht Dijkstra, nicht auf einem Graphen.
* **Voronois gibt es zwar** — aber ausschliesslich fuer die **Plot-Physik** (Stadtgrundrisse, `LandscapeVoronoiSystem`/`CityBlockSystem`) und, davon voellig getrennt, fuer die **Seegliederung** im Terrain. Mit dem Wegenetz hat beides nichts zu tun. Der einzige `dijkstra()`-Aufruf in `settlement_generator.py` gehoert zur Stadtblock-Zerlegung.
* **Kandidatenpaare** kommen aus einem Gabriel-Graphen ueber die Siedlungen, danach entscheidet ein Bereitschaftstest (`rang_a * rang_b * kulturfaktor > wegkosten/luftlinie`), und bereits gebaute Wege verbilligen spaetere Routen (`WEGERABATT`).

### Bewertung des Voronoi-Wegegraphen: davon wuerde ich ABRATEN

Gemessen (echtes Gelaende):

| | 512 px | 1024 px |
|---|---:|---:|
| A\*-Lauf je Wegpaar | 0.12 s | 0.47 s |
| hochgerechnet auf ~30 Kandidatenpaare | ~4 s | ~14 s |

Ein Voronoi-Graph mit 10 000 Zellen waere als Dijkstra praktisch umsonst (Millisekunden). **Aber er wuerde genau das verschlechtern, was der Nutzer als Problem benennt.** Groessenvergleich:

| Aufloesung | Zellgroesse |
|---|---:|
| 1 000 Voronoi-Zellen | ~674 m Kantenlaenge |
| 10 000 Voronoi-Zellen | ~213 m |
| 50 000 Voronoi-Zellen | ~95 m |
| Pixelgitter 1024 px | **20.8 m** |

Selbst 10 000 Zellen sind **rund zehnmal groeber als das heutige Pixelgitter**; ein Weg ueber Zellmittelpunkte bestuende aus ~213 m langen Geradenstuecken und koennte sich nicht mehr durch ein Tal schlaengeln. Der Nutzer sagt selbst *"bisher sieht das ziemlich gut aus, halte dich etwa daran"* — genau diese Qualitaet kaeme aus dem Gelaendedetail, das ein Zellgraph wegwirft. **Das Problem ist Qualitaet, nicht Tempo; der Voronoi-Graph loest Tempo und kostet Qualitaet.** 14 s bei 1024 px sind zudem kein Engpass neben den ~33 s von `terrain.redistribution` (7.12).

*(Zur Nebenfrage "Kosten aus den Hoehenabstaenden zwischen den Mittelpunkten - genau genug?": nein, und das ist unabhaengig vom Obigen ein Fehler, der sich lohnt zu kennen. Zwei Zellmittelpunkte auf gleicher Hoehe koennen durch eine Schlucht getrennt sein - die Differenz der Endpunkte sagt nichts ueber den Weg dazwischen. Man muesste mindestens die maximale Steigung ENTLANG der Zellgrenze nehmen, besser die tatsaechlichen Pfadkosten - womit man wieder beim Pixel-A\* waere.)*

### Wofuer Voronoi-Zellen dagegen SEHR sinnvoll sind: Platzierung

Derselbe Vorschlag ist an anderer Stelle richtig, und dort wuerde ich ihn uebernehmen: **Zellen als Auswahlraster fuer die Suitability**, nicht als Wegegraph. Der Nutzer beschreibt das Verfahren bereits vollstaendig: Suitability je Zelle, beste zuerst belegen, Nachbarzellen danach abwerten (verhindert Klumpen), weiter bis Zielzahl oder Mindestguete erreicht. Das ist ein Standardverfahren (Greedy mit Unterdrueckungsradius), passt zur Stadttypen-Idee (je Typ eine eigene Eignungskarte) und ist billig, weil auf Zellen statt Pixeln gerechnet wird. **Fuer Roadsites gilt dasselbe** — nur eingeschraenkt auf Zellen, durch die eine Strasse laeuft, wie vom Nutzer vorgeschlagen.

### Stadttypen und Handelsgewichte: gute Idee, ein offener Widerspruch

Die vier Typen (Bergdorf / Marktstadt / Agrarstadt / Sonstige) sind klar definierbar, und **die noetigen Daten liegen alle vor**: Hoehe und Hangneigung fuer Bergdorf, `heightmap`/Kuestennaehe plus Erreichbarkeit fuer Marktstadt, `biome_map` und Hangneigung fuer Agrarstadt. "Nur eine Marktstadt pro Region" ist als Nebenbedingung im Greedy-Verfahren oben leicht unterzubringen.

**Ein Widerspruch in der Handelsmatrix ist vor der Umsetzung zu klaeren:** die Gewichte sind je Typ EINSEITIG angegeben, ergeben aber paarweise zwei verschiedene Werte. Beispiel Bergdorf-Marktstadt derselben Fraktion: aus Sicht des Bergdorfs 5, aus Sicht der Marktstadt 3. Welcher Wert gilt fuer die Kante? Drei sinnvolle Auflösungen: **Maximum** (der Interessiertere setzt sich durch - passt zur Erzaehlung "das Bergdorf braucht den Markt dringender"), **Summe** (beide Interessen addieren sich), oder **Mittel**. Empfehlung: Maximum, weil es die Asymmetrie der Beschreibung am ehesten trifft und keine kuenstlich grossen Zahlen erzeugt.

### Was ich am bestehenden Weg BEHALTEN wuerde

* **Gabriel-Graph als Kandidatenfilter** — verhindert von vornherein Kanten quer durch andere Orte, ohne Sonderregeln.
* **Wegerabatt auf bereits gebauten Trassen** — genau der vom Nutzer gewuenschte Effekt "es werden bevorzugt die bereits gesetzten Strassen gewaehlt", und er wirkt schon heute auch auf die ENTSCHEIDUNG, nicht nur auf die Geometrie (siehe Kommentar im Code, das war ein bewusster frueherer Fix).
* **Bereitschaftstest gegen die Wegkosten** — die Handelsgewichte des neuen Entwurfs treten schlicht an die Stelle von `rang_a * rang_b`, der Mechanismus bleibt.

### Vorgeschlagene Reihenfolge (deckt sich weitgehend mit dem Nutzerentwurf)

1. Zellraster + Eignungskarten je Stadttyp, Staedte greedy setzen (mit Unterdrueckung und "max. eine Marktstadt je Region")
2. Handelsgewichte aus den Typen, Wegenetz wie heute (Gabriel + A\* + Rabatt), nur mit den neuen Gewichten
3. Landmarks (1-2 je Region, eigene Eignungskriterien je Typ aus dem vorhandenen 45er-Katalog)
4. Stichwege zu Landmarks (1-2 Anbindungen bis zur naechsten Strasse, kein Vollnetz)
5. Roadsites auf Strassen-Zellen, 2-4 je Region, mit Typkriterien (Taverne an Kreuzung ab 3 Wegen usw.)

**Aufwand gesamt gross** — das ist der vom Nutzer angekuendigte "Settlements komplett neu durchgehen". Sinnvoll in dieser Reihenfolge auch abzuarbeiten, weil jeder Schritt auf dem vorigen aufbaut.

### SCHRITT 1 UMGESETZT 2026-08-13: Stadttypen, Groessengrenzen, Handelsgewichte

Nutzerentscheidungen dazu: Kantengewicht = **Summe** beider einseitiger Interessen (statt Maximum/Mittel), und **Schritt 1 zuerst**.

**Die bestehende Platzierung wurde NICHT angefasst.** Der Nutzer hatte sie ausdruecklich gelobt ("Die Siedlungen sind ziemlich gut gesetzt worden bisher"); der Typ wird deshalb NACH der Platzierung aus der Lage der bereits gesetzten Orte abgeleitet, statt die Orte nach Typ neu zu verteilen. Damit entfaellt auch das urspruenglich erwogene Zellraster fuer die Platzierung - es haette genau das veraendert, was gut ist.

**Gebaut:**
* `STADTTYPEN` mit `rang_erlaubt` je Typ (Bergdorf dorf/siedlung, Marktstadt siedlung/stadt, Agrarstadt siedlung/stadt, sonstige alles) - setzt "klein bis mittel"/"mittel bis gross" durch.
* `TerrainSuitabilityAnalyzer.stadttyp_eignungen()` - eine Eignungskarte je Typ, **aufgebaut auf den bereits vorhandenen vier Teilfaktoren** (`calculate_water_proximity`/`analyze_slope_suitability`/`evaluate_elevation_fitness`/`evaluate_farmland_radius`). Keine zweite Gelaendeanalyse, keine zweite Wahrheit.
* `SettlementGenerator._typen_zuweisen()` - Marktstadt zuerst (hoechstens eine je Kultur, und nur ueber `TYP_GRUNDGUETE`: eine Region ohne brauchbare Hafenlage bekommt lieber keine), danach je Ort der beste der uebrigen Typen, zuletzt Rang gegen `rang_erlaubt` korrigiert. **Seit 5.17 zaehlt fuer die Marktstadt `max(Wasserlage, Zentralitaet)`** - das "oder" aus "liegt am Wasser oder kann viele Staedte gut erreichen" ist damit woertlich umgesetzt, statt die Erreichbarkeit nur ueber die Ebenheit zu naehern.
* `Location.settlement_type` als neues Feld.
* `handelsgewicht()` / `_handelsinteresse_einseitig()` - die Zahlen der Vorgabe, als Summe beider Sichten. Tritt im Bereitschaftstest des Wegenetzes an die Stelle von `rang_a * rang_b * kulturfaktor`; die Kultur steckt jetzt IM Handelsgewicht (ein zusaetzlicher Fremdkultur-Faktor waere doppelt gezaehlt). **Rueckfall auf die alte Formel**, wenn kein Ort einen Typ traegt (alter Nicht-Weltkarten-Pfad).

**Ein Fehler beim Bauen, gefunden durch Messen:** die Bergigkeit war zuerst als `1 - hoehe_suit` angesetzt. Das ist falsch - `evaluate_elevation_fitness()` liefert eine EIGNUNG (hoch = angenehme Wohnhoehe), kein Hoehenmass; ihr Median liegt auf der Testkarte bei 0.94, die Invertierung also bei 0.06. Gemessen kamen dadurch **nur 2.6 % der Landflaeche** ueberhaupt als Bergdorf in Frage, obwohl 34.6 % des Landes ueber 200 m liegen - praktisch waere nie ein Bergdorf entstanden. Ersetzt durch den Rang der ECHTEN Hoehe zwischen Median und 95. Perzentil des Landes (relativ zur jeweiligen Karte, damit auch eine flache Welt ihre "Berge" hat). Danach **13.0-13.4 % ueber drei Seeds**, stabil. Der Smoke-Test sichert genau diese Untergrenze, damit der Rueckfall nicht unbemerkt wiederkehrt.

**Gemessen (384 px, drei Seeds):** Typverteilung nach Flaeche im Wettbewerb ohne Marktstadt (die ist je Region auf eine beschraenkt): Bergdorf 13.0-13.2 %, Agrarstadt 40.1-52.1 %, sonstige 34.9-46.7 %. An der jeweils besten Position je Typ entsteht auch der erwartete Typ; zwoelf Orte an bester Wasserlage ergeben genau eine Marktstadt.

Gesichert in `tests/smoke_test_stadttypen.py` (Eignungskarten inkl. Bergdorf-Untergrenze, Zuweisung an gezielten Lagen, Marktstadt-Einzigkeit, Handelsgewichte inkl. Symmetrie und Kultur-/Groessenabhaengigkeit). Regressionsfrei (`smoke_test_settlement_sites.py`, `smoke_test_settlement_region_grid.py`, `smoke_test_display_2d.py`).

**Noch offen aus dem Entwurf:** Landmarks (Schritt 3), Stichwege zu Landmarks (4), Roadsites auf Strassen-Zellen (5) - und die Sichtpruefung, ob das Wegenetz mit den neuen Handelsgewichten tatsaechlich besser aussieht. **Die Handelsgewichte sind rechnerisch verifiziert, ihre Wirkung auf das Bild aber nicht** - dafuer braucht es einen Blick auf die erzeugte Karte. | 8 |

| **[~]** | **5.17** | **Wegfindung: drei algorithmische Fragen des Nutzers, beantwortet und gemessen (2026-08-13).** Frage 2 ist inzwischen auch UMGESETZT (siehe unten), Frage 3 bewusst zurueckgestellt - mit einer Korrektur an meiner eigenen frueheren Aussage in 5.16.

### Frage 1: "Ist A\* das beste Verfahren, um Wegkosten zwischen zwei Orten zu finden?"

**Fuer EIN Paar ja, fuer die Erreichbarkeitsfrage nein** - und dieser Unterschied ist der Kern der zweiten Frage. A\*s Vorteil gegenueber Dijkstra ist ausschliesslich die Zielheuristik: sie lenkt die Suche auf EIN Ziel zu. Sobald man Kosten zu VIELEN Zielen braucht, gibt es kein einzelnes Ziel mehr, die Heuristik ist wertlos, und A\* faellt auf Dijkstra zurueck - bei k Zielen aber k-mal statt einmal.

### Frage 2: "Wie findet man in jeder Region den Ort mit der besten Erreichbarkeit?"

**Nicht mit A\* je Ortspaar.** Das richtige Werkzeug ist ein **Multi-Source-Dijkstra**: eine einzige Wellenfront, die gleichzeitig von ALLEN Startpunkten laeuft und fuer JEDEN Pixel die Kosten zum naechsten Start liefert. `skimage.graph.MCP_Geometric.find_costs(starts)` macht genau das und ist im Projekt bereits im Einsatz (`terrain_river_network.py`, `water_generator.py`).

Gemessen mit dem echten Kostenfeld, 12 Staedte:

| | 512 px | 1024 px |
|---|---:|---:|
| **A) Ein Multi-Source-Lauf** (Kosten zum naechsten Ort, alle Pixel) | **0.06 s** | **0.35 s** |
| B) 12 Einzellaeufe (Summe aller Distanzen = echte Closeness) | 0.79 s | 4.07 s |
| C) A\* je Paar, hochgerechnet auf 66 Paare (heutiger Weg) | 4.19 s | 14.30 s |

**A ist rund 40x schneller als C** und liefert dabei MEHR: nicht nur Kosten zwischen den Orten, sondern ein vollstaendiges Erreichbarkeitsfeld ueber die ganze Karte, aus dem sich der beste Punkt je Region direkt als Minimum ablesen laesst. Fuer "erreicht viele Staedte gut" (Marktstadt-Kriterium) braucht man Variante B - eine Welle je Stadt, aufsummiert; auch die bleibt mit 4 s bei 1024 px deutlich unter dem heutigen paarweisen Verfahren.

**Empfehlung:** das Erreichbarkeitsfeld fuer die Marktstadt-Eignung (5.16) und fuer jede kuenftige "wo ist es gut angebunden"-Frage ueber `MCP_Geometric.find_costs()` rechnen, nicht ueber A\*-Paare. Der heutige A\*-Pfad bleibt richtig fuer das, wofuer er da ist: die tatsaechliche GEOMETRIE eines einzelnen Weges.

### Frage 3: grobes Gitter mit vorberechneten Feinwegen - und die "Stummel"

**Das ist ein etabliertes Verfahren (HPA\*, Hierarchical Pathfinding A\*) und es ist BESSER als das, wovon ich in 5.16 abgeraten habe.** Der Unterschied ist wesentlich und ich hatte ihn dort nicht sauber getrennt:

* **Voronoi-/Zellgraph mit Zellmittelpunkten** (wovon ich abriet): der WEG selbst laeuft von Mittelpunkt zu Mittelpunkt. Die Geometrie wird dadurch grob - bei 10 000 Zellen ~213 m lange Geradenstuecke.
* **HPA\*** (die Idee des Nutzers): der Graph dient nur der SUCHE. Jede Graphkante haelt einen vorberechneten Weg auf dem FEINEN Gitter. Der ausgegebene Weg ist die Aneinanderreihung dieser Feinwege - **die Geometrie bleibt pixelfein**. Verloren geht nur die Garantie strikter Optimalitaet (der Weg muss durch die gewaehlten Tore), typischerweise wenige Prozent Mehrkosten.

Also: Tempo wie ein Graphverfahren, Aussehen wie das heutige A\*. Genau das, was hier gebraucht wird.

**Zum Stummel-Problem** (Nutzer: *"eventuell aber stummel die an den punkten entstehen wenn der optimale weg leicht an dem punkt vorbeifuehrt und muessten diese punkte noch einmal verschieben"*): die Beobachtung stimmt, aber das Verschieben der Punkte ist die muehsamere der beiden Loesungen und bleibt ein Nachlauf gegen ein Symptom. **HPA\* loest es an der Wurzel, indem die Knoten gar nicht erst auf ein starres Raster gelegt werden:**

1. Karte in Cluster teilen (das koennen die Spielkarten-Vielecke aus 5.15 sein oder ein einfaches Kachelraster).
2. Knoten NICHT in die Clustermitte, sondern in die **Tore** - die Stellen, an denen zwei Cluster aneinandergrenzen. Wo eine Clustergrenze ueber einen Pass, eine Furt oder einen schmalen Kuestenstreifen laeuft, sitzt der Knoten genau dort, wo ein Weg ohnehin durchmuss. An breiten, gleichfoermigen Grenzen nimmt man den kostenguenstigsten Punkt des Grenzabschnitts, ggf. mehrere je Abschnitt.
3. Kanten: innerhalb eines Clusters von Tor zu Tor per feinem A\* vorberechnen (billig, weil der Cluster klein ist).
4. Einen Ort anbinden: temporaer als Knoten einfuegen und per feinem A\* mit den Toren SEINES Clusters verbinden.

Weil die Knoten auf den natuerlichen Engstellen sitzen statt auf Rasterpunkten, fuehrt der optimale Weg durch sie hindurch statt an ihnen vorbei - der Stummel entsteht gar nicht erst. Ein Nachlauf zum Verschieben entfaellt.

**Aufwand ehrlich:** das ist ein eigener, nicht kleiner Umbau (Cluster-Zerlegung, Torfindung, Kanten-Vorberechnung, Anbindungslogik, Cache-Invalidierung bei Gelaendeaenderung). **Und er loest ein Tempoproblem, das derzeit keines ist** - 14 s bei 1024 px neben den ~33 s von `terrain.redistribution` (7.12). Lohnt sich erst, wenn entweder die Kartengroesse deutlich waechst oder die Zahl der Wege (Landmark-Stichwege, Roadsite-Anbindungen aus 5.16) so steigt, dass die paarweisen A\*-Laeufe spuerbar werden. **Fuer die Erreichbarkeitsfrage aus Frage 2 wird es nicht gebraucht** - dort ist Multi-Source-Dijkstra einfacher und schneller.

### FRAGE 2 UMGESETZT 2026-08-13 (Nutzer: "gut dann los")

* **`erreichbarkeits_matrix(kostenfeld, positionen)`** - (n,n)-Wegkostenmatrix ueber je eine `MCP_Geometric`-Welle pro Startpunkt, statt n*(n-1)/2 A\*-Laeufe.
* **Gerechnet wird auf einem GROBEN Gitter** (`ERREICHBARKEIT_GITTER_PX = 128`). Begruendung und Messung: hier wird BEWERTET, nicht gezeichnet - es zaehlt die Rangfolge, nicht der Meterwert. Gemessen mit 30 Orten bei 1024 px: volle Aufloesung **7.8 s**, auf 128 px **0.08 s** (**92x schneller**) bei einer Rangkorrelation der Zentralitaet von **0.997**. Die tatsaechlichen WEGE entstehen unveraendert per feinem A\* in `calculate_road_network()` - dieses Feld ersetzt sie nicht, es bewertet nur Standorte.
* **`zentralitaet(matrix)`** - je Ort ein Wert 0..1 aus der Summe der Wegkosten zu allen anderen (Closeness). Unerreichbare Ziele zaehlen mit dem hoechsten endlichen Wert statt `inf`, sonst waere die Normierung NaN und eine abgeschnittene Insel kaeme als "gut angebunden" durch.
* **Marktstadt-Wahl nutzt jetzt beides**: `max(Wasserlage, Zentralitaet)`. Das **Maximum** und kein Produkt, weil die Vorgabe "liegt am Wasser ODER kann viele Staedte gut erreichen" ein echtes ODER ist - eine Binnenstadt am Wegeknoten ist ebenso Marktstadt wie ein abgelegener Hafen.
* Das Kostenfeld ist DASSELBE, das auch das Wegenetz benutzt (`bau_kostenfeld()`), durchgereicht von `_calc_settlements` - eine zweite Kostendefinition waere eine zweite Wahrheit.
* **Rueckfaelle melden sich**: fehlt das Kostenfeld oder scheitert die Rechnung, gibt es eine WARNING und es zaehlt nur die Wasserlage (schwaechere, aber nicht falsche Bewertung).

**Verifiziert:** Matrixform, Nulldiagonale, naeherungsweise Symmetrie (das grobe Gitter rundet Start-/Zielpixel, daher rtol 5 %), Normierung der Zentralitaet, und die Richtungsprobe "hoechste Zentralitaet = kleinste Kostensumme" (faengt ein verdrehtes Vorzeichen). Gemessen 0.04 s fuer 6 Orte bei 512 px. **Die Erreichbarkeit aendert die Marktstadt-Wahl auch tatsaechlich** - im Test faellt sie mit Kostenfeld auf den Ort mit Zentralitaet 1.00, ohne auf einen anderen; waere das nicht so, waere der ganze Aufwand wirkungslos und der Test wuerde es zeigen. Gesichert in `tests/smoke_test_stadttypen.py` (Gruppe `erreichbarkeit`). Regressionsfrei (`smoke_test_settlement_sites.py`, `smoke_test_settlement_region_grid.py`, `smoke_test_spielkarten.py`, `smoke_test_display_2d.py`, `smoke_test_display_methoden_existieren.py`).

**Frage 3 (HPA\*) bleibt bewusst offen** - siehe Begruendung oben: loest ein Tempoproblem, das derzeit keines ist. | 4 |

| **[~]** | **5.18** | **Landmarks: Auswahl nach EIGNUNG statt per Zufallsziehung (Schritt 3 des Siedlungs-Umbaus, 2026-08-13).** Nutzerbefund war "Die Strassen und Roadsites und Landmarks sind schlecht"; fuer die Landmarks liess sich die Ursache klar benennen.

**Was schlecht war:** die vier Kategorien (`gipfel`/`kueste`/`quelle`/`abgelegen`) waren **binaere Masken**, aus denen anschliessend **zufaellig** gezogen wurde. Ein "Gipfel"-Landmark landete damit auf irgendeinem Pixel oberhalb 60 % der Hoehenspanne - nicht auf dem Gipfel; eine "Kueste"-Landmark auf irgendeinem kuestennahen Pixel - nicht am markanten Kliff. Die Namen aus dem 45-Arten-Katalog versprachen also etwas, das die Lage nicht einloeste.

**Umgebaut:** neue `SettlementGenerator.landmark_eignungen()` liefert je Kategorie eine kontinuierliche Guete (0..1), und die Platzierung nimmt je Region schlicht das Beste (mit Mindestabstand und leichtem Rauschen, damit nicht jede Region demselben Muster folgt). Die alten Masken bleiben als harte Vorbedingung (Wildnis, kein Extremhang, Land) - die Eignung entscheidet innerhalb davon. Kriterien:

* **gipfel** - echtes lokales Hoehenmaximum ueber `maximum_filter`, nicht nur "hoch gelegen". Der Hoehenrang wird NUR ueber Land gebildet; ueber die ganze Karte gerechnet wuerde die Meerestiefe die Spanne dominieren (derselbe Fehler wie bei den Stadttypen, 5.16).
* **kueste** - nah am Meer UND markant: die Hangneigung geht ein, damit ein Kliff ein Sandufer sticht.
* **quelle** - nah an Suesswasser, aber hoch gelegen. Ohne den Hoehenanteil waere jede Muendung eine "Quelle".
* **abgelegen** - fern jeder Zivilisation, **gedeckelt auf `ABGELEGEN_DECKEL = 0.55`**.

**Zwei Fehler beim Bauen, beide durch Messen gefunden:**

1. **"abgelegen" haette alles verschluckt.** `civ_map` ist auf weiten Teilen der Karte schlicht 0, die Einsamkeit dort also 1.0 - gemessen mit leerem Zivilisationsfeld lag die Kategorie auf **100 % der Landflaeche** bei Maximalwert, waehrend ein echter Gipfel nur 0.32 % der Flaeche ueber 0.5 bringt. Ohne Deckel waere JEDES Landmark "abgelegen" geworden und die drei ortsgebundenen Kategorien haetten nie gezogen. Der Deckel macht sie zum Auffangtyp, genau wie `TYP_GRUNDGUETE` bei den Stadttypen.
2. **Eine Kategorie belegte trotzdem fast alles.** Auf einer 21-km-Insel ist die Kuesteneignung flaechendeckend hoch - gemessen **14 von 20 Landmarks "kueste"**, alle mit demselben Katalognamen. Behoben ueber `KATEGORIE_WIEDERHOLUNG = 0.45`: nach jeder Wahl wird die gewaehlte Kategorie in DIESER Region gedaempft. Kein hartes Verbot - eine Region ohne Berge soll kein Gipfel-Landmark erzwingen muessen.

**Gemessen nachher (384 px, zwei Seeds):** staerkste Kategorie **48-52 %** statt 70 %, Verteilung z.B. kueste 11 / abgelegen 8 / gipfel 4. "gipfel"-Landmarks liegen im Mittel im **96.8-97.2 %-Perzentil** der Landhoehen, "kueste"-Landmarks im Mittel **1.1-1.2 px** vom Meer - die Kategorienamen loesen ihr Versprechen jetzt ein. 21 verschiedene Katalognamen bei 23 Landmarks.

**EIN ECHTER REGRESSIONSFEHLER, DEN MEIN EIGENER TEST VERDECKT HATTE:** der neue Code rief `zufall.uniform(..., size=...)`. `_knoten_zufall()` liefert aber ein `random.Random`, kein numpy-`RandomState` - das kennt kein `size=`. Mein Landmark-Test blieb trotzdem gruen, weil ich `_knoten_zufall` mit einem numpy-Generator **gemockt** hatte; aufgefallen ist es erst, als `smoke_test_settlement_sites.py` mit `TypeError` umfiel. **Ein Mock, der mehr kann als das Original, prueft nichts** - der Test setzt jetzt `map_seed` und laesst die echte Methode laufen. Das Rauschfeld kommt aus einem aus demselben Generator geseedeten numpy-Generator, damit es weiterhin nur an Seed und Knotennamen haengt.

Gesichert in `tests/smoke_test_landmarks.py` (Eignungskarten inkl. Deckel-Wirkung, Platzierung mit Kategorie-Obergrenze und Lagequalitaet je Kategorie, Determinismus) - bewusst mit einem REALISTISCHEN Zivilisations- und Wasserfeld, weil leere Felder genau den ersten Fehler unsichtbar gemacht haetten. Regressionsfrei (`smoke_test_settlement_sites.py`, `smoke_test_settlement_region_grid.py`, `smoke_test_stadttypen.py`, `smoke_test_display_2d.py`).

**[~] statt [x]:** die Lagequalitaet ist gemessen, das BILD aber nicht beurteilt - ob die Landmarks jetzt auch gut aussehen, muss der Nutzer am Programm sehen. **Noch offen aus dem Entwurf:** Stichwege zu Landmarks (Schritt 4) und Roadsites auf Strassen-Zellen (Schritt 5). | 3 |

| **[~]** | **5.19** | **Steigungskosten waren als Strafe praktisch wirkungslos - jetzt exponentiell** *(Nutzerbefund 2026-08-13: "die hoehenkosten sind zu niedrig. es gibt strassen die ueber hohe berge gehen, dann sollte vorher lieber eine strasse gewaehlt werden die um die berge herum geht ... keiner wuerde eine strasse bauen die zB mehr als x Grad steigung hat. und 5 Grad weniger ist schon wesentlich besser quasi. also irgendwann wird es einfach unpassierbar.")*

**Der Befund stimmte und war drastischer als vermutet.** Die alte Formel `1 + ratio * hang^2` rechnete ueber den Gradientenbetrag; gemessen kostete damit

| Steigung | alte Kosten | neue Kosten |
|---:|---:|---:|
| 0 Grad | 1.00 | 1.0 |
| 10 Grad | 1.05 | 4.7 |
| 15 Grad | 1.11 | 9.3 |
| 20 Grad | 1.20 | 17.8 |
| 25 Grad | 1.33 | 33.6 |
| **30 Grad** | **1.50** | **500** |
| 40 Grad | 2.06 | 500 |

Ein 30-Grad-Hang kostete also nur das **1.5-fache** eines ebenen Pixels. Damit war jeder Umweg von mehr als 50 % Mehrlaenge teurer als die Direttissima ueber den Berg - der Wegfinder verhielt sich voellig folgerichtig, nur war die Vorgabe falsch parametriert.

**Neu:** `1 + ratio * (exp(winkel / STEIGUNG_SKALA_GRAD) - 1)`, gerechnet ueber den WINKEL in Grad statt ueber den Gradientenbetrag. `STEIGUNG_SKALA_GRAD = 8.0` trifft die Nutzer-Vorgabe "5 Grad weniger ist wesentlich besser" quantitativ: von 20 auf 15 Grad faellt der Preis um **Faktor 1.92**. Ab `MAX_WEG_STEIGUNG_GRAD = 30.0` gilt der Hang als unbrauchbar.

**Die Sperre ist BEWUSST ENDLICH** (`WEGEBAU_UNMOEGLICH = 500.0` statt `np.inf`): eine harte Sperre wuerde ganze Landesteile abschneiden, sobald ein Ort hinter einem durchgehend steilen Wall liegt - er waere dann gar nicht mehr ans Netz anzubinden, und das Ergebnis waere ein fehlender Weg statt eines teuren. Mit einem hohen endlichen Wert nimmt A\* einen solchen Uebergang nur, wenn es wirklich keine Alternative gibt.

**Gemessen an echten Wegen (384/512 px, acht bis zehn Ortspaare, identische Start-/Zielpunkte):**

| | alt | neu |
|---|---:|---:|
| Median-Steigung entlang der Wege | 10.3 Grad | **6.3 Grad** |
| p90 | 23.6 Grad | **17.9 Grad** |
| Anteil ueber 20 Grad | 16.8 % | **7.3 %** |
| Anteil ueber 30 Grad | 4.4 % | **1.2 %** |
| Gesamtlaenge aller Wege | 1792 px | 2334 px |

Die Wege sind also **laenger geworden und deutlich flacher** - genau das gewuenschte "lieber drumherum". **Und es bleibt alles erreichbar:** von einem beliebigen Landpunkt aus sind weiterhin **100 % der Landflaeche** erreichbar, kein Weg scheiterte.

**Nebenwirkung, bewusst in Kauf genommen:** das Kostenfeld speist auch die Erreichbarkeits-/Zentralitaetsrechnung der Marktstadt-Wahl (5.17). Deren Ergebnis aendert sich dadurch - was richtig ist: "erreicht viele Staedte gut" soll ja gerade heissen "ueber Wege, die man auch baut", nicht "Luftlinie ueber den Grat".

Gesichert in `tests/smoke_test_stadttypen.py` (Gruppe `steigungskosten`): Kurvenform und Monotonie, der 5-Grad-Faktor, die Unpassierbarkeitsschwelle, die **Endlichkeit** der Sperre, der Vorher/Nachher-Vergleich an echten Wegen und die vollstaendige Erreichbarkeit. Regressionsfrei (`smoke_test_settlement_sites.py`, `smoke_test_settlement_region_grid.py`, `smoke_test_landmarks.py`, `smoke_test_settlement_clustering.py`).

**[~] statt [x]:** die Zahlen sind eindeutig, das Bild ist es noch nicht - ob die Wege jetzt auch plausibel AUSSEHEN, muss der Nutzer am Programm beurteilen. **Nicht angefasst:** Serpentinen. Ein isotropes Kostenfeld je Pixel kann sie prinzipiell nicht erzeugen - es kennt nur "wie steil ist es hier", nicht "in welche Richtung laeuft der Weg". Ein Weg quer zum Hang haette geringe Steigung, das Pixel trotzdem hohe Neigung. Dafuer braeuchte es richtungsabhaengige Kanten statt eines Pixelfelds; das ist ein eigener Umbau und war nicht Teil der Vorgabe. | 1 |

| **[~]** | **5.20** | **Roadsites: echte Wegscheiden von blossen Beruehrungen unterscheiden (Schritt 5, 2026-08-13).** Nutzer-Vorgabe: *"roadsites haben auch bestimmte kriterien (zB taverne oder wie das heisst an einer kreuzung mit min. drei wegen etc)"*.

**Warum das bisher gar nicht pruefbar war:** `kreuzungen_finden()` meldet jede Stelle, an der sich **mindestens zwei** Wege beruehren, und gibt nur die ORTE zurueck - keine Information darueber, WIE VIELE Wege dort zusammenkommen. Der Roadsite-Katalog kennt zwar eine Kategorie "kreuzung" (7 Eintraege ueber alle Kulturen), aber jede zufaellige Beruehrung zweier Strecken zaehlte genauso viel wie eine echte Wegscheide. Eine "Taverne an der Kreuzung" konnte damit an einer Stelle landen, an der sich zwei Wege nur streifen.

**Neu:** `kreuzungsgrade(roads, sea_roads, kreuzungen, shape)` zaehlt je Kreuzung die verschiedenen Weg-IDs in einem kleinen Umkreis. Der Umkreis ist noetig, weil `kreuzungen_finden()` den SCHWERPUNKT einer Pixelgruppe zurueckgibt - der genaue Mittelpunkt muss selbst nicht auf jedem beteiligten Weg liegen. Kreuzungen ab `KREUZUNG_MIN_WEGE = 3` stehen bei der Roadsite-Auswahl VOR den uebrigen; die schwaecheren bleiben als nachrangige Kandidaten erhalten (eine Region ohne echte Wegscheide soll nicht leer ausgehen).

**Verifiziert an synthetischen Wegen** - bewusst synthetisch und nicht an einem generierten Netz: hier wird GEZAEHLT, und ob ein Stern vier und eine Beruehrung zwei Wege hat, muss exakt stimmen. An einem generierten Netz waere die erwartete Zahl selbst unbekannt, der Test koennte nur sich selbst bestaetigen. Geprueft: Stern (4 Wege) -> Grad 4, T-Kreuzung -> Grad 3, blosse Beruehrung -> Grad 2; die ersten beiden gelten als Wegscheide, die dritte nicht. Zusaetzlich, dass Grade und Kreuzungen deckungsgleich bleiben - der Grad wird VOR `_fern_zuerst()` bestimmt und muss beim Umsortieren mitwandern, sonst zeigte er anschliessend auf die falsche Kreuzung (beim Bauen mitgedacht, im Test abgesichert).

Gesichert in `tests/smoke_test_stadttypen.py` (Gruppe `kreuzungsgrade`). Regressionsfrei (`smoke_test_settlement_sites.py`, `smoke_test_settlement_region_grid.py`, `smoke_test_landmarks.py`, `smoke_test_settlement_clustering.py`).

**Ein Missverstaendnis beim Pruefen, hier festgehalten weil es Zeit gekostet hat - mit einer Korrektur am 2026-08-13 selbst:** ein Wegwerf-Skript meldete "0 Wege gebaut" und weckte den Verdacht, die neuen Steigungskosten (5.19) haetten die Wegfindung kaputtgemacht. Die erste Gegenprobe war richtig und entlastete die Aenderung: die ALTE Kostenformel lieferte im selben Skript ebenfalls 0/10. **Die dann notierte Erklaerung war aber falsch** - ich schrieb, der Pfad sei "auf 2 Punkte vereinfacht" worden. Der tatsaechliche Grund kam erst spaeter heraus: `PathfindingSystem.find_least_resistance_path()` gibt ein **Tupel `(pfad, erreicht)`** zurueck, nicht den Pfad. Der Filter `len(p) > 2` prueft damit die Laenge eines Zweier-Tupels und ist IMMER falsch, egal wie lang der Pfad ist. Der Fehler lag also beide Male im Pruefskript - aber aus einem anderen Grund als zuerst notiert. **Zwei Lehren:** (1) bevor man eine frische Aenderung verdaechtigt, denselben Test gegen den alten Stand laufen lassen; (2) eine plausible Erklaerung ist noch keine gepruefte - der Rueckgabetyp haette sich in einer Zeile nachsehen lassen.

**[~] statt [x]:** die Gradbestimmung ist exakt geprueft, die WIRKUNG auf das Kartenbild nicht - ob die Roadsites dadurch sichtbar besser sitzen, muss der Nutzer beurteilen. **Noch offen aus dem Entwurf:** die uebrigen Roadsite-Kriterien (der Nutzer nannte die Taverne nur als Beispiel, "die liste durchgehen und kriterien entwerfen" steht noch aus) und Schritt 4, die Stichwege zu den Landmarks - `calculate_landmark_roads()` bindet heute jedes Landmark an den naechsten Wegpunkt an, die Vorgabe "eher 1-2 strassen zu siedlungen in der naehe (bzw bis zur naechsten strasse)" ist damit erst halb erfuellt. | 2 |

| **[~]** | **5.21** | **UMGESETZT 2026-08-13. Wegenetz nach BEDARF statt nur paarweise** *(Nutzerfrage 2026-08-13 am Kartenbild: "der berg in der mitte und darueber zwei ortschaften die nicht verbunden sind ... die verbindung waere in echt sehr sehr hoch angesehen, da es fuer alle ortschaften auf der linken seite bedeutet nur unten lang zu koennen ... wir vergleichen ja nur dorf 1 will in dorf 2 handeln und sucht den kostenguenstigsten weg fuer sich. aber nicht das 10 doerfer daran interessiert sind eine verbindung oben zu haben")*

**Die Beobachtung trifft genau die Schwaeche des heutigen Verfahrens.** `calculate_road_network()` entscheidet je PAAR: lohnt sich fuer A und B eine direkte Strecke? Ein Pass ueber den Berg lohnt sich fuer kein einzelnes Paar - er ist fuer jedes fuer sich zu teuer. Dass er fuer ZEHN Paare zusammen der mit Abstand groesste Gewinn waere, sieht das Verfahren strukturell nicht. Das ist kein Fehler in der Umsetzung, sondern eine Luecke im Modell.

**Der Fachbegriff dafuer ist Grenznutzen (marginal utility) beim Netzausbau.** Vorgehen:

1. **Kostenmatrix** `C[i][j]` - Wegkosten zwischen allen Ortspaaren. **Liegt bereits vor** (`erreichbarkeits_matrix()`, 5.17).
2. **Bedarfsmatrix** `W[i][j]` - wie viel Handel zwischen i und j stattfinden WILL. **Liegt bereits vor** (`handelsgewicht()`, 5.16).
3. **Netzdistanz** `d_N(i,j)` - Kosten von i nach j ueber die BISHER GEBAUTEN Strecken (nicht Luftlinie, nicht Direktweg).
4. **Grenznutzen einer Kandidatenkante e:** `Nutzen(e) = SUMME ueber alle Paare von w_ij * (d_N(i,j) - d_{N+e}(i,j))` - also: um wie viel verkuerzt diese eine Strecke die Wege ALLER Handelspaare zusammen, gewichtet mit ihrem Handelsvolumen.
5. Greedy die Kante mit dem besten `Nutzen / Baukosten` bauen, dann `d_N` neu rechnen und wiederholen, bis kein Kandidat mehr eine Schwelle ueberschreitet.

**Der entscheidende Punkt fuer die Machbarkeit: Schritt 3-5 laufen auf dem GRAPHEN mit rund 30 Knoten, nicht auf der Pixelkarte.** Die teure Geometrie steckt komplett in `C` und ist einmalig bezahlt (gemessen 5.17: alle Wellen zusammen 4 s bei 1024 px). Ein Dijkstra ueber 30 Knoten kostet Mikrosekunden; selbst hunderte Kandidatenkanten durchzuprobieren bleibt im Millisekundenbereich. **Der Umbau ist also billig zu rechnen - der Aufwand steckt im Umstellen der Logik, nicht in der Laufzeit.**

**Billiger Vorab-Detektor, der dieselbe Situation ohne den vollen Umbau sichtbar macht:** der **Umwegfaktor** `d_N(i,j) / C[i][j]` - Netzdistanz geteilt durch die Kosten des direkten Weges. Wo der gross ist, fehlt eine Verbindung. Genau die Situation im Bild: die beiden Doerfer ueber dem Berg haben einen kleinen Direktwert (sie sind nah), aber eine grosse Netzdistanz (rundherum). Das laesst sich in wenigen Zeilen aus den vorhandenen Matrizen ablesen und waere ein guter erster Schritt - erst messen, wie oft und wie stark das auftritt, dann entscheiden, ob der volle Grenznutzen-Ausbau noetig ist.

**Wichtige Wechselwirkung mit 5.19:** die verschaerften Steigungskosten machen Paesse jetzt richtig teuer. Genau deshalb wird der Grenznutzen wichtiger - ohne ihn wird ein Pass kuenftig noch seltener gebaut als vorher, obwohl er als EINZIGE Nordverbindung fuer die halbe Karte wertvoll waere. **Die beiden Punkte gehoeren zusammen betrachtet.**

### UMGESETZT 2026-08-13 (Nutzer: "ja umsetzen")

Drei neue Funktionen in `core/settlement_generator.py`:

* **`netzdistanzen(kanten, anzahl)`** - kuerzeste Wege zwischen allen Orten UEBER DAS GEBAUTE NETZ (Dijkstra auf dem Ortsgraphen, nicht auf Pixeln).
* **`umwegfaktoren(netz_D, direkt_C)`** - der billige Detektor: Netzdistanz geteilt durch Direktkosten. Gross = "die beiden sind eigentlich nah, das Netz macht einen weiten Bogen".
* **`kanten_nach_bedarf(C, W, bestehende, kandidaten)`** - der eigentliche Ausbau, gierig nach `Nutzen / Baukosten`.

Verdrahtet als **Schritt 4** in `calculate_road_network()` (neue `_netz_nach_bedarf_ausbauen()`), NACH dem bisherigen paarweisen Bau - **ergaenzend, nicht ersetzend**, wie vom Nutzer vorgegeben ("halte dich etwa daran, wir wollen das nur etwas besser machen"). Zwei Sicherungen: `NETZAUSBAU_MINDESTNUTZEN = 3.0` (eine Strecke muss mindestens das Dreifache ihrer Baukosten an Gesamtnutzen bringen) und `NETZAUSBAU_MAX_KANTEN = 3` je Kultur, damit aus dem sparsamen Gabriel-Netz kein Vollgraph wird. Scheitert der Schritt, meldet er das per WARNING und das Netz bleibt beim paarweisen Ergebnis.

**Am Modellfall geprueft** (die Bergsituation des Nutzers exakt nachgebaut: zwei Ketten zu vier Orten, nur unten herum verbunden, oben ein kurzer Pass moeglich): der Umwegfaktor zeigt **4.0x** an, der Pass wird gebaut, die gewichteten Gesamtwegkosten fallen **4240 -> 2320 (45 %)**. Bewusst als kleines, exakt nachrechenbares Modell und nicht an einem generierten Netz - dort waere das erwartete Ergebnis selbst unbekannt und der Test koennte nur sich selbst bestaetigen.

**Am echten Gelaende gemessen** (384 px, 7 Orte einer Kultur, Gabriel-artiges Ausgangsnetz): 2 Kanten ergaenzt, groesster Umwegfaktor **1.63x -> 1.15x**, gewichtete Gesamtwegkosten **2.9 % kuerzer**, Laufzeit **2 ms**. **Der Gewinn ist hier ehrlicherweise klein** - das Ausgangsnetz war bereits gut (groesster Umweg nur 1.63x). Die 45 % aus dem Modellfall treten nur dort auf, wo tatsaechlich ein Bergruecken zwei Ortsgruppen trennt. Der Mechanismus greift, seine Wirkung haengt an der Karte.

**Gegenproben im Test** (sonst waere nicht belegt, dass er auch NICHTS baut, wenn nichts zu holen ist): ohne Handelsinteresse wird keine Kante gebaut; eine absurd teure Kante wird nicht gebaut; die Obergrenze wird eingehalten.

Gesichert in `tests/smoke_test_stadttypen.py` (Gruppe `netzausbau_nach_bedarf`). Regressionsfrei (`smoke_test_settlement_sites.py`, `smoke_test_settlement_region_grid.py`, `smoke_test_landmarks.py`, `smoke_test_settlement_clustering.py`).

**[~] statt [x]:** rechnerisch belegt, am Kartenbild noch nicht beurteilt - ob der Pass aus dem Ausgangsbild jetzt tatsaechlich erscheint, muss der Nutzer sehen. **Noch offen:** die Werte `NETZAUSBAU_MINDESTNUTZEN`/`NETZAUSBAU_MAX_KANTEN` sind gesetzt, aber nicht gegen ein Zielbild geeicht - dafuer braucht es den Blick auf mehrere erzeugte Karten. | 3 |

| **[~]** | **5.22** | **TEILWEISE UMGESETZT 2026-08-13. Seewege: Umsteigekosten statt Notloesung, plus Fischersiedlung** *(Nutzerfrage 2026-08-13: "verbindungen zu inseln, also seeverbindungen sind noch nicht wirklich moeglich, bzw werden nicht dargestellt")*

**Die Darstellung ist NICHT das Problem** - sie steht in beiden Ansichten (2D `overlay_roads(sea_roads, color='royalblue', linestyle='--')`, 3D seit 6.23 in `rasterize_settlements_rgba()`). Es entstehen schlicht kaum Seewege. Drei Bedingungen muessen gleichzeitig zutreffen (`calculate_road_network()`):

1. **Es darf KEINEN endlichen Landweg geben.** Nur bei einer echten Insel erfuellt - Wasser ist weiterhin `np.inf` (die endliche Sperre aus 5.19 betrifft nur STEILHANG, nicht Wasser; das ist richtig so).
2. **Beide Orte muessen zur selben Kultur gehoeren.** Die Schleife laeuft je Kultur. Eine Insel mit fremder Nachbarkultur bleibt unverbunden.
3. **Mindestens 50 % des Seewegs muessen in echt tiefem Wasser liegen** (`_seeweg_anteil_tief`), damit es "kein Seeweg, sondern ein schlechter Landweg" wird.

**Gemessen (512 px, Seed 20260804):** 66 getrennte Landmassen, davon nur 3 groesser als 0.3 km². Im Ausschnitt der Inselgruppe unten rechts: 52 % Wasserflaeche, davon **genau 50 % tief** - die Bedingung (3) liegt also exakt auf der Schwelle und kippt je nach Route. Zwischen dicht beieinanderliegenden Inseln ist das Wasser ueberwiegend Kuestensee (Seegrad 0), und genau dort faehrt ein Schiff in Wirklichkeit am ehesten.

**Drei moegliche Wege, in aufsteigendem Aufwand:**

* **(a) Bedingung (3) lockern** - die 50 %-Regel war gegen "Seewege, die an der Kueste entlangtasten" gedacht. Eine kurze Ueberfahrt zwischen zwei Inseln ist aber genau das, was sie sein soll. Statt eines festen Anteils waere eine Laengenschranke sinnvoll: bei kurzen Ueberfahrten (< X km) den Anteil gar nicht pruefen.
* **(b) Bedingung (2) lockern** - Seewege auch zwischen Kulturen zulassen. Handel ueber See zwischen Nachbarvoelkern ist historisch die Regel, nicht die Ausnahme.
* **(c) Bedingung (1) ersetzen** - **das ist zugleich die Loesung von 5.21**: ein Seeweg ist einfach eine Kante mit eigenen Kosten. Statt "nur wenn gar kein Landweg existiert" gehoert er als normaler Kandidat in den Grenznutzen-Ausbau. Dann entsteht eine Faehre automatisch dort, wo sie den groessten Gesamtnutzen bringt - auch wenn ein sehr langer Landweg drumherum theoretisch existiert.

**Empfehlung:** (a) ist ein Einzeiler mit sofortiger Wirkung und sollte zuerst gemessen werden; (c) ist die saubere Loesung und faellt mit 5.21 zusammen an.

### UMGESETZT 2026-08-13 (Nutzer-Vorgabe: fixe Umsteigekosten, geeicht auf ~35 % Seehandel)

**Messgrundlage zuerst** - ohne sie waere "35 %" eine unbelegbare Behauptung:
* **`kanten_traffic(kanten, W, n)`** - gewichtete Kanten-Betweenness: fuer jedes Ortspaar den Netzweg bestimmen und sein Handelsvolumen auf alle Kanten dieses Weges schlagen. **Zugleich die Grundlage der spaeteren Traffic-Anzeige aus 6.27** - vorher waere jede solche Zahl erfunden gewesen.
* **`seehandel_anteil(...)`** - daraus der Anteil, der ueber Seekanten laeuft. Bewusst nach VOLUMEN, nicht nach Anzahl der Seewege: zwei kaum genutzte Faehren sollen nicht so viel zaehlen wie eine Hauptroute.

**Umsteigekosten in KILOMETERN, nicht in Kostenpunkten** (`HAFEN_UMSTEIGEKOSTEN_KM`, `hafenkosten(mpp)`). **Das war ein Fehler im ersten Anlauf und fiel beim Eichen auf:** ein ebenes Pixel kostet 1.0, die Kosten eines Weges wachsen also mit seiner PIXELzahl - dieselbe absolute Zahl ergab bei 320 px und 384 px deutlich verschiedene Seehandelsanteile und waere bei 1024 px praktisch wirkungslos geworden. In Kilometern gerechnet bremst der Hafenwechsel bei jeder Kartengroesse gleich stark.

**Eichung, gemessen ueber vier Faelle (320/384 px, vier Seeds):**

| Umsteigekosten | 0 km | 22 | 30 | **40** | 45 | 70 |
|---|---:|---:|---:|---:|---:|---:|
| Seehandel (Mittel) | 68 % | 50 % | 43 % | **~35 %** | 33 % | 20 % |

Gesetzt auf **40 km**. **Ehrliche Einschraenkung, die im Code steht:** der Anteil streut je Karte stark (bei 45 km zwischen 0 % und 53 %). Das ist nicht behebbar und auch richtig - eine Karte ohne vorgelagerte Inseln hat keinen Seehandel, egal wie billig die Haefen sind. Der Regler stellt den DURCHSCHNITT ein, nicht die Einzelkarte; genau so hat der Nutzer es formuliert ("insgesamt, manche inselorte sind natuerlich bei 100% seehandel").

**Fischersiedlung als fuenfter Stadttyp** (Nutzervorschlag, von mir befuerwortet): sie fuellt eine echte Luecke, weil die Marktstadt per Definition mittel bis gross und hoechstens einmal je Region ist - eine Insel mit drei Haeusern kann keine sein, braucht aber einen Hafen. Deshalb **ausdruecklich klein** (nur dorf/siedlung, nie stadt), mit eigenen Handelswerten (zur Marktstadt 6/3, untereinander 3/2, ins Binnenland 1.5/1) und einer eigenen Eignungskarte: unmittelbar am Wasser, aber OHNE den Ebenheits-/Hinterlandanteil der Marktstadt - genau das ist der Unterschied. `FISCHER_DECKEL = 0.62` haelt sie unter der Marktstadt, wo beide moeglich waeren (gemessen: auf 85 % der Landflaeche gewinnt die Marktstadt).

**Nicht umgesetzt und weiterhin offen:** der eigentliche Punkt (c) - Seewege als gleichberechtigte Kandidaten IM Netzausbau. `_seehandel_messen()` rechnet die Umsteigekosten heute nur zur MESSUNG mit; `calculate_road_network()` baut Seewege weiterhin nur, wenn gar kein Landweg existiert. Der Regler ist damit geeicht und die Messung steht, aber die Bedingungen (1) und (2) aus der Analyse oben sind unveraendert. Das ist der naechste Schritt und gehoert mit 5.21 zusammen.

Gesichert in `tests/smoke_test_stadttypen.py` (Gruppe `seehandel_und_fischersiedlung`): Aufloesungsunabhaengigkeit der km-Rechnung, Monotonie des Reglers, Traffic-Zaehlung, Deckel und Rangbeschraenkung der Fischersiedlung. Regressionsfrei (vier Siedlungs-Suiten). | 2 |

# 6 — Anzeige

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **6.6** | **Regionen in 2D sichtbar** — `region_map` als Output, neun eingefaerbte Gebiete mit Beschriftung im Terrain-Reiter. | 1 |
| **[x]** | **6.7** | **Vier Ausgaben kamen nie an** (`region_map` und die drei Flussausgaben): `set_terrain_data_complete_lod` zaehlte nur heightmap/slopemap/shadowmap auf. Ohne Absturz, ohne Warnung. | 0.5 |
| **[x]** | **6.9** | **`matplotlib.cm.get_cmap` gibt es nicht mehr** (entfernt in 3.9) — der eine Aufruf in `_render_generic_map` legte **18 Darstellungen** lahm: Hardness, die fuenf Geologie-Diagnosen, Humidity, Flow, Bodenfeuchte, Verdunstung und den ganzen Erosionsreiter. Rock Outcrop und Cross-Section ueberlebten, weil sie eigene Zeichenwege haben. | 0.5 |
| **[x]** | **6.10** | **Slope 2D tot**: `self.current_colorbar = None`, in der naechsten Zeile `.set_label()` darauf. AttributeError bei jedem Zeichnen; 3D lief weiter, weil der Weg dort vorbeigeht. | 0.5 |
| **[x]** | **6.11** | **Fluss-Reiter zeigte nie etwas**, 2D wie 3D: `_show_data` fragte nach `self.current_display` — das Attribut gibt es in `BaseMapTab` nicht. Laeuft jetzt ueber `_push_data_to_current_display()`. | 0.5 |
| **[x]** | **6.12** | **Niemand prueft zwischen Daten und Bild.** `smoke_test_pipeline_outputs.py` sichert die Daten, danach war Lehrraum — alle drei Fehler oben blieben deshalb unbemerkt. Neu: `tests/smoke_test_display_2d.py` zeichnet alle 30 Darstellungen wirklich und prueft, dass danach etwas auf der Achse steht. | 1 |
| **[~]** | **6.1** | **Regionenansicht** (Nutzer-Vorgabe 2026-08-11: "Grenzen wie in medieval... mit farbigen Regionen, weisser Trennung und halt nur wenn man Regionen ausgewaehlt und in subtilen Toenen bei Settlement-Vorschau, wo Staedte/Strassen angezeigt werden"). **2D erledigt:** neue `rasterize_regions_rgba()` (`map_display_2d.py`) - Regionsfarben als Flaeche + WEISSE Grenzlinien, transparent ausserhalb Land, headless verifiziert (Kernfarbe trifft exakt, 977 Grenzpixel bei 256px). Terrain-Reiters bestehende exklusive Regionen-Ansicht (6.6) von dunklen auf weisse Grenzen umgestellt. NEUES zuschaltbares Overlay `overlay_regions()` - "Regionen"-Checkbox in Siedlungen Global, AUS per Default, subtiler Ton (alpha 0.25) damit Staedte/Strassen vorne bleiben. **Noch offen:** dieselbe Checkbox in Siedlungen Regional (dort weniger dringend, der Reiter zeigt ohnehin schon eine einzelne, ausgewaehlte Region) und die eigentliche 3D-Darstellung - dafuer aber der sicherste Weg schon gefunden: `map_display_3d.py`s bestehender RGBA-Textur-Overlay-Pfad (`_render_plot_boundaries()`, `useAlphaOverlay=1`) kann `rasterize_regions_rgba()`s Ausgabe genauso hochladen wie die Plot-Grenzen - KEIN neuer GLSL-Code noetig, nur ein neuer `overlay_data`-Schluessel + Render-Aufruf + Checkbox-Verdrahtung je Reiter. | 1 | **3D-Teil umgesetzt 2026-08-13** (Nutzer-Vorgabe: "die 3D darstellung ALLER 2D maps, aber vor allem der kuesten auf die 3D Terrains bekommen ... ich will das jetzt endlich in 3D vergleichen koennen"): genau der oben skizzierte Weg, ohne neuen GLSL-Code. `_render_plot_boundaries()` zu generischem `_render_rgba_overlay(rgba)` verallgemeinert (aufgerufen jetzt von drei Stellen: Plots, Regionen, Kuestentypen). Neue `_render_dict_rgba_overlay()`: rasterisiert beim Zeichnen aus demselben Rohdaten-Dict, das auch der 2D-Renderer bekommt (`rasterize_regions_rgba()`/neue `rasterize_kuesten_archetypen_rgba()`, EINE Farblogik fuer beide Ansichten statt einer zweiten, die auseinanderlaufen koennte). `_LAYER_NAME_MAP_3D`/`_LAYER_SELECTION_KEYS_3D` (`base_tab.py`) um `region_map`->`region_overlay`/`kuesten_archetyp`->`kuesten_overlay` ergaenzt - `terrain_tab.py` brauchte dafuer KEINE Aenderung, es schickte die richtigen Daten bereits ueber denselben Push-Mechanismus. **Wichtiger Fund dabei:** nur `renderMode==5` (getSettlementColor() in terrain.frag) behandelt den Alpha-Kanal der Overlay-Textur korrekt (`useAlphaOverlay`) - jeder andere renderMode-Zweig mischt nur mit fester `overlayStrength` und wuerde transparente Bereiche trotzdem einfaerben. Deshalb nutzt JEDER RGBA-Skin-Overlay renderMode=5, unabhaengig vom aufrufenden Tab - keine neue Shader-Logik noetig, aber die urspruengliche Annahme "eine neue renderMode-Nummer pro Overlay" war falsch. Headless verifiziert: beide Rasterisierer liefern korrektes (H,W,4) uint8 RGBA, `update_overlay_data()`s neue Dict-Validierung akzeptiert/verwirft korrekt, `_render_dict_rgba_overlay()`s Dispatch-Logik Ende-zu-Ende getestet (GL-Aufruf gemockt). Regressionsfrei inkl. `smoke_test_layer_2d_3d_parity.py` (dort `region_overlay`/`kuesten_overlay` in die bereits vorhandene Kategorisch-Ausnahmeliste aufgenommen - laufen nie durch die skalare Farbskalen-Logik). **Nicht pruefbar ohne die Live-App:** das tatsaechliche GL-Rendering selbst (CLAUDE.md) - ob es im 3D-Fenster gut aussieht, muss der Nutzer bestaetigen. Regionen in Siedlungen Regional bleibt offen (dort weniger dringend).
| [ ] | 6.13 | **3D ist weiter ungeprueft.** Der Prueftand oben deckt nur 2D ab; ein OpenGL-Widget laesst sich nicht kopflos zeichnen (CLAUDE.md). Fuer 3D bleibt es beim Nachsehen im laufenden Programm. | 2 |
| **[x]** | **6.2** | **Gelbes 3×3-Gitter** im Regional-Reiter und in Siedlungen Global — `overlay_region_grid()` in `map_display_2d.py`, gespeist aus `terrain_weltkarte.gitterlinien_px()` (derselben Rechnung wie 5.9/5.14). Bewusst NICHT im Terrain-Reiter (docs/TODO.md §D2: dort laege es neben den Kulturfarben). | 1 |
| **[x]** | **6.3** | Wetter-Reiter mischt Hoehenschichten mit Groessen in EINER Knopfreihe | 0.5 | **WAR BEREITS ERLEDIGT** (2026-08-24 nachgeprueft): `create_visualization_controls()` in `gui/tabs/weather_tab.py` baut seit laengerem zwei getrennte Zeilen und nennt 6.3 im Docstring. Nur der Haken fehlte.
| [ ] | 6.4 | 3D-Ansicht ist von keinem Test erfasst | 2 |
| [ ] | 6.5 | Terrain-Auswahlfeld je Region *(Nutzer: "auf lange sicht")* | 2 |

| **[x]** | **6.8** | **Geology brach bei "stack deformations" ein** (Nutzermeldung 2026-08-10). Zwei Glaettungen mit sigma als festem Bruchteil der Karte wuchsen KUBISCH (16.6 s bei 2048 px), und das Stoerungsfeld rechnete elf Kartenarrays je Segment (25.5 s). Jetzt Grobgitter bzw. Kachelvorauswahl: 18- bis 29-fach schneller, Stoerungen bitgleich. Gesichert in `tests/smoke_test_geology_speed.py`. | 1 |
| **[x]** | **6.15** | **Kartenkosmetik** (Nutzer-Vorgabe 2026-08-12): (a) Hoehenkonturen-Abstaende verdoppelt (25/50/100m -> 50/100/200m), die 0-Linie zeichnet jetzt dick+dunkel, alles unter 0 in Graustufen statt der bunten Landkontur-Palette (`_calculate_contour_levels()`/`_draw_contour_lines()`, `map_display_2d.py`). (b) `water_depth_map` (Erosion-Reiter, misst Oberflaechenwasser/Abfluss, nicht Ozeantiefe) zeigte offene See fast weiss statt blau, da dort kein Abflusswert existiert - jetzt eine deckende Meeresfarbe ueber jedes Pixel unter 0m gelegt (`_render_generic_map()`, nutzt die ohnehin vorhandene Referenz-Heightmap). Regressionsfrei (`smoke_test_display_2d.py`, alle 30 Darstellungen). | 0.5 |
| **[x]** | **6.14** | **Jeder Tab redraw'te bei JEDER Terrain/Geology/Erosion-Fertigstellung, nicht nur der sichtbare** (Pipeline-Log-Nutzerbefund 2026-08-11, siehe 8.4). `data_updated` feuert `on_data_updated()` in allen zehn Tabs synchron auf dem Hauptthread; `_TERRAIN_FORMING_GENERATORS` loeste dort JEDEN aus, nicht nur den sichtbaren - gemessen 187s allein bei Terrain (1024px). Fix: `on_data_updated()` (`base_tab.py`, `overview_tab.py`) prueft jetzt `self.viewport_widget.isVisible()`, bevor es neu zeichnet - `map_editor._on_tab_changed()` ruft `update_display_mode()` beim Tab-Wechsel ohnehin bereits auf, und `update_display_mode()` hat einen Dirty-Check, holt verpasste Updates also von selbst nach. **Gemessen (1024px, GPU-Pfad, sonst identischer Lauf): Gesamtzeit terrain.noise→settlement.[assemble] von >15 Min (mit GPU-Fallback-Kaskade, siehe 8.4) bzw. 4:19 Min (ohne) auf 2:41 Min** - `Ablegen`-Werte fielen von bis zu 187s auf 0-12s. | 1 |
| **[~]** | **6.16** | **3D-Mesh an Klippen "pixelig", zu viele Dreiecke insgesamt** (Nutzer-Vorgabe 2026-08-12: Remesh so, dass Dreiecke an Klippen/Detailbereichen kleiner sind, sonst groesser, ohne auf 5 Mio. Dreiecke zu laufen). Diskutiert: Nutzer wollte urspruenglich Kombination aus fehler-getriebener adaptiver Triangulierung UND entfernungsbasiertem LOD (~3 Stufen), aber explizit mit der Bitte, die Notwendigkeit von LOD-Stufen fuer diesen Zweck selbst zu beurteilen. Empfehlung (angenommen: "go ahead"): NUR adaptive Triangulierung, KEIN entfernungsbasiertes LOD - kleine (~21km), einzelne Vorschau mit Orbit-Kamera, kein offenes/stroemendes Terrain; adaptive Triangulierung platziert Detail ohnehin nach Gelaende-Komplexitaet, nicht nach Kameraabstand; Mehrstufen-LOD haette echtes Nahtstellen-Risiko fuer kaum Nutzen bei dieser Groessenordnung eingebracht. **Umgesetzt:** `gui/widgets/adaptive_terrain_mesh.py` (neu) - restricted Quadtree (bewusst NICHT RTIN/Martinis Bit-Indizierung, siehe Modul-Docstring: ohne Referenz-Implementierung zum Gegenpruefen zu fehleranfaellig fuer nur im laufenden 3D-Fenster sichtbare Risse), Hoehen-Fehler-getriebene Teilung, balanciert (kein Nachbar mehr als 1 Stufe feiner/groeber), Kanten-Faecher-Triangulierung gegen T-Junctions - reproduziert exakt Diagonale+Wicklung des bisherigen Gleichmaessig-Gitters (wichtig wegen aktivem Backface-Culling). `_generate_terrain_mesh()` (`map_display_3d.py`) nutzt das neue Mesh bei quadratischer Heightmap mit Zweierpotenz-Kantenlaenge, sonst automatischer Rueckfall auf das alte Gitter. **Headless verifiziert** (`tests/smoke_test_adaptive_terrain_mesh.py`, 10 Tests): Risslosigkeit (jede innere Kante von genau 2 Dreiecken geteilt) bei flacher Karte, scharfer Klippe, Zufallsrauschen (niedrige und hohe Toleranz), gemischtem Bild, realistischer Groesse mit geglaettetem Rauschen+Klippe; Determinismus; normierte Normalen. **VORBEDINGUNG WAR ZUERST FALSCH und der Fehler wurde NUR durch den Nutzerlauf sichtbar** (2026-08-12): `ist_fuer_adaptives_mesh_geeignet()` verlangte anfangs `Kantenlaenge-1 = Zweierpotenz` (die RTIN-uebliche "2^n+1"-Konvention), aber dieses Projekt nutzt map_size-Werte, die **selbst** Zweierpotenzen sind (1024, nicht 1025). Damit war die Bedingung fuer **jede reale Kartengroesse** falsch, das adaptive Mesh lief nie, und der Rueckfall auf das Gleichmaessig-Gitter griff still — im Log als `Adaptives Mesh nicht anwendbar (Heightmap-Groesse)` bei jedem Aufbau. **Die zehn gruenen Tests deckten das nicht auf, weil sie alle mit 2^n+1-Groessen (129/257/513) gebaut waren — sie testeten eine Groessenklasse, die im Programm nicht vorkommt.** Behoben: Bedingung auf `Kantenlaenge = Zweierpotenz` umgestellt, intern um genau eine Zeile/Spalte gepolstert (`_gepolsterte_hoehen()`, Kantenwert dupliziert), Tests auf die echten Groessen (128/256/512/1024) umgestellt. **Danach am echten 1024px-Lauf gemessen: 412855 statt 2093058 Dreiecke (19.7 %), 206527 statt 1048576 Vertices** — im Programm bestaetigt, nicht nur im Test. **Lehre, uebertragbar:** eine Vorbedingung, die im Fehlerfall still auf einen funktionierenden Pfad zurueckfaellt, ist von gruenen Tests nicht zu unterscheiden — sie braucht entweder eine laute Logzeile (hier vorhanden, deshalb gefunden) oder einen Test mit den ECHTEN Eingabegroessen. **Noch offen:** Sichtpruefung, ob das Gelaende unveraendert aussieht (Nutzer 2026-08-12: "sieht ok aus, also genau wie vorher" — das ist die erwartete und gewuenschte Antwort, aber vor dem Vorbedingungs-Fix gegeben, also am ALTEN Gitter; nach dem Fix noch nicht erneut beurteilt). Fehlertoleranz fest auf 6 m (`_adaptive_mesh_fehler_toleranz_m`, `map_display_3d.py`), kein Bedienelement — bewusst, da nicht angefragt. | 2 |

| **[x]** | **6.18** | **DIAGNOSTIZIERT, BEIDE URSACHEN BEHOBEN UND AM LAUFENDEN PROGRAMM BESTAETIGT 2026-08-13.** Der urspruengliche Befund lautete "jeder 3D-Ansichtswechsel dauert 15 s". **Das war nur zur Haelfte richtig, und die eingebaute Aenderungserkennung wirkt tatsaechlich:** ein Layer-Wechsel bei bereits gebautem Netz kostet gemessen **0.002-0.008 s** (`Rebuild uebersprungen`). Was 15 s kostet, ist der **ERSTE** Netzaufbau je Reiter (`mesh vorhanden=False`) - Terrain **15.7 s**, Geologie **33.3 s**. Heightmap-Beschaffung (0.009 s) und Vergleich (0.000 s) sind vernachlaessigbar, die gesamte Zeit steckt in `_generate_terrain_mesh()`. **Zwei getrennte Ursachen, beide behebbar:** (a) **Das identische Netz wird je Reiter neu gebaut.** Terrain und Geologie melden exakt dieselben Zahlen - `450193/2093058 Dreiecke, 194068 Blaetter, 225178 Vertices`. Bei zehn Reitern im 3D-Modus zehnmal dieselbe Rechnung. Ein Zwischenspeicher ueber den Heightmap-Inhalt spart neun davon. (b) **Der Erbauer selbst ist zu langsam.** Headless bei 1024 px gemessen: 2.4 s - im Programm 15.7 s. Der Unterschied ist die Gelaendebeschaffenheit: das Testgelaende (geglaettetes Rauschen) ergab **5065 Blaetter**, das echte **194068** - Faktor 38. `_blaetter_balancieren()` und `_dreiecke_aus_blaettern()` arbeiten je Blatt in Python-Schleifen und skalieren damit linear mit; `_besitzer_gitter()` baut ausserdem in JEDER Balancier-Runde ein volles 1024x1024-int64-Feld neu auf. Zu vektorisieren. **Nebenbefund:** die Dreiecksersparnis ist mit **21.5 %** viel geringer als die 1.5 % der Testmessung - echtes Gelaende ist ueberall detailliert, der Quadtree kann kaum zusammenfassen. Eine hoehere Fehlertoleranz als 6 m wuerde beides zugleich verbessern (Tempo und Dreieckszahl), kostet aber Detail - vor dem Drehen daran messen. | 2 | **TEIL (a) BEHOBEN 2026-08-13:** `build_adaptive_mesh()` speichert das Ergebnis jetzt ueber einen INHALTS-Schluessel der Heightmap zwischen (`_MESH_CACHE`, drei Eintraege, FIFO). Bewusst der Inhalt und nicht `id()`: `get_terrain_data_combined()` liefert bei jedem Aufruf ein frisches Array per `.copy()`, eine Identitaetspruefung ginge immer daneben. Der Schluessel enthaelt zusaetzlich beide Skalierungsfaktoren und die Fehlertoleranz. **Gemessen (1024 px): erster Aufbau 3.16 s, jeder weitere Reiter 0.005 s** - bitgleiches Ergebnis, und der Treffer funktioniert auch bei einer KOPIE der Karte. Die Konsole meldet jetzt `(neu gerechnet)` bzw. `(aus Zwischenspeicher)`. **TEIL (b) EBENFALLS BEHOBEN 2026-08-13** (Nutzerfreigabe "ja angehen"): alle vier Aufbaustufen vektorisiert, Ergebnis **bit-identisch** zur alten Fassung. (1) `_fehler_pyramide()` ersetzt die rekursive `_quad_fehler()` durch ebenenweise numpy-Slices bottom-up - die alte Fassung machte je Quadtree-Knoten einen Python-Funktionsaufruf samt dict-Zugriff, bei 1024 px rund 1.4 Mio. davon. (2) `_blaetter_sammeln_pyramide()` waehlt die Blaetter ebenenweise statt rekursiv. (3) `_blaetter_balancieren()` und (4) `_dreiecke_aus_blaettern()` arbeiten auf drei parallelen int64-Arrays statt auf einem dict, gruppiert nach Blattgroesse (Python-Schleife nur noch ueber die ~11 VORKOMMENDEN Groessen statt ueber 194068 Blaetter); die Faecher-Triangulierung nutzt zusaetzlich, dass es nur 16 moegliche Nachbar-Kombinationen gibt (welche der vier Kanten einen feineren Nachbarn hat), also 16 feste Polygon-Schablonen statt einer Fallunterscheidung je Blatt. Vertex-Zusammenfassung ueber ein einzelnes `numpy.unique()` statt ueber ein Python-dict.

**EIN ZWISCHENSTAND WAR MESSBAR LANGSAMER ALS DAS ORIGINAL und wurde verworfen:** der erste Anlauf holte die Nachbargroessen ueber `sliding_window_view(besitzer, c).min(axis=-1)`, also ein gleitendes Minimum ueber das GESAMTE Gitter - das skaliert mit `zellen**2`, unabhaengig davon, wie wenige Blaetter dieser Groesse es gibt. Gemessen bei 1024 px/3019 Blaettern: **1.63 s alt gegen 3.24 s neu, also halb so schnell.** Aufgefallen nur, weil der Vergleichstest die Zeiten BEIDER Fassungen nebeneinander ausgibt statt nur die Gleichheit zu pruefen - ein reiner Korrektheitstest haette die Verschlechterung durchgewinkt. Ersetzt durch gezieltes Gathern der `K*c` Randzellen (`_kanten_minima_je_seite()`), das mit der tatsaechlich beruehrten Kantenlaenge skaliert.

**Gemessen (`tests/smoke_test_adaptive_mesh_vectorized.py`, neu - vergleicht die neue gegen eine eingebettete Kopie der alten Fassung):** 8 Faelle auf ECHTEN Kartengroessen (128/256/512/1024 px, je zwei Toleranzen), **alle bit-identisch** in Blattmenge, Vertexmenge und Dreiecksgeometrie. Gesamtzeit **4.21 s -> 0.38 s (11.2x)**, je Fall 9.4x bis 13.7x. Der 1024px-Fall mit Toleranz 0.5 (16750 Blaetter) stellt bewusst das Szenario aus dem Nutzerlog nach - glattes Testgelaende allein ergibt nur rund 3000 Blaetter und haette genau den Fall verfehlt, der im Programm langsam war (dieselbe Falle wie bei 6.16, siehe CLAUDE.md). Bestehender Test `smoke_test_adaptive_terrain_mesh.py` weiter gruen mit identischen Zahlen (7740 Dreiecke/3298 Blaetter), seine Performance-Zeile faellt von 0.31 s auf 0.02 s. Regressionsfrei (`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`, `smoke_test_terrain_scale_coupling.py`, `smoke_test_river_reaches_sea.py`, `smoke_test_layer_2d_3d_parity.py`). **Noch offen:** die Zeit im laufenden Programm ist damit NICHT nachgemessen - die 15.7 s stammten aus einem Nutzerlauf bei echtem Weltgelaende, nicht aus dem Testgelaende hier. **LIVE BESTAETIGT (Nutzerprotokoll 2026-08-13, 1024 px, echtes Weltgelaende, 196339 Blaetter): erster Netzaufbau `1.795 s` statt der zuvor gemessenen 15.7 s (8.7x), jeder weitere Reiter `0.007 s` aus dem Zwischenspeicher.** Nutzer zum Bild: "auch keine risse vorhanden" - die Wasserdichtigkeit haelt also auch bei echtem Gelaende, nicht nur im Test. Damit sind Teil (a) und (b) beide erledigt und am Programm nachgewiesen.
| **[x]** | **6.17** | **Klippen sprangen senkrecht auf einem einzigen Pixel** *(Nutzerbefund 2026-08-12: gezackte "Mauer" ringsum jede Kueste in der 3D-Ansicht)*. Zunaechst als Mesh- oder Textur-Problem vermutet — **beides falsch, es war ein Geometriefehler in der Heightmap selbst**, also unabhaengig von 6.16. Gemessen an einer echten Karte: Land-Pixel direkt an der Kueste lagen im **Median 63 m, im Extrem 530 m** ueber ihrem tiefsten See-Nachbarn — bei ~42 m Pixelbreite (512px) eine echte Senkrechte. Ursache in `_kuesten_umformen()` (3.8): die Reichweite der Klippen-Anstiegskurve `L` kam aus einer Formel ohne Bezug zur Zielhoehe, mit fester Untergrenze **15 m — kleiner als ein einziger Kartenpixel**. Jede hohe, steile Klippe (bis ~800 m Zielhoehe) erreichte ihre volle Hoehe damit INNERHALB eines Pixels; Heightmap und Mesh koennen so etwas gar nicht abbilden. Behoben ueber den physikalischen Zusammenhang statt einer neuen Konstante: fuer `H0*(1-exp(-d/L))` ist die Anfangssteigung genau `H0/L`, also setzt **`L = Zielhoehe / tan(winkel)`** die tatsaechliche Steigung an der Kuestenlinie auf den im Archetyp gewuenschten Winkel — der `winkel_grad`-Eintrag der Tabelle bedeutet damit erstmals wirklich das, was sein Name sagt. Zusaetzliche Untergrenze bei 2 Pixeln. **Gemessen nachher: Median-Sprung 63 m → 20 m bei 1024px** (512px: 30 m), regressionsfrei (`smoke_test_seegliederung.py`, `smoke_test_terrain_river_network.py`, `smoke_test_terrain_scale_coupling.py`). **Ehrliche Einschraenkung:** vollstaendig verschwindet die Kantigkeit nicht und kann es nicht — mehrere Archetypen stehen bewusst auf 78–84°, und eine echte Steilwand SIEHT bei dieser Aufloesung kantig aus. Wer sie glatter will, muss die Winkel senken (Vorschlag: Obergrenze 55–60° statt 85°, `MAX_KLIPPENWINKEL_GRAD`) und dafuer Dramatik aufgeben — eine Gestaltungsentscheidung, keine Fehlerbehebung. | 1 |

| **[x]** | **6.19** | **Eine Heightmap kann keine senkrechte Wand darstellen — das ist die Grenze, nicht das Netz** *(Nutzerbefund 2026-08-13: "die kuesten sind immernoch 90 Grad ... die pixel sind immernoch da ... ich habe nicht das gefuehl dass das mesh ausser x und y auch z in betracht zieht")*. **Die Beobachtung ist richtig und die Erklaerung dafuer ist strukturell:** eine Heightmap speichert je (x,y) GENAU EINEN Hoehenwert. Zwei benachbarte Pixel mit 180 m und -3 m ergeben zwangslaeufig eine senkrechte Flaeche von einem Pixel Breite - und weil die Kuestenlinie dem Pixelraster folgt, entsteht daraus die sichtbare **Treppe**. **Das adaptive Netz (6.16) kann das nicht beheben und war nie dafuer gedacht:** es fasst Dreiecke zusammen, wo das Gelaende flach ist: es kann eine Stufe, die in den DATEN steht, nicht glaetten. Die Vermutung des Nutzers, das Netz beachte die z-Richtung nicht, trifft insofern zu, als die Unterteilung nach Hoehen-FEHLER geht und eine echte Stufe an jeder Aufloesung ein Fehler bleibt - sie wird bis zur feinsten Stufe geteilt und bleibt trotzdem senkrecht. **Drei ehrliche Wege, alle mit Preis:** (a) **Die Hoehen an der Kueste ueber mehrere Pixel verteilen** - dann sind die Klippen glatter, aber weniger dramatisch: das ist dieselbe Stellschraube wie 6.17 und haengt an `MAX_KLIPPENWINKEL_GRAD` und den `winkel_grad`-Werten der Archetypen. (b) **Aufloesung erhoehen** (2048 statt 1024) - halbiert die Stufenbreite, verdoppelt Speicher und Rechenzeit, beseitigt die Treppe aber nicht, sondern verkleinert sie nur. (c) **Das Heightmap-Modell an der Kueste verlassen** (echte Geometrie, Ueberhaenge) - das waere ein anderer Renderer und steht in keinem Verhaeltnis zum Nutzen. **Zu entscheiden ist (a) gegen (b)**: keine Rechnung, sondern eine Gestaltungsfrage. | 1 | **ERLEDIGT 2026-08-24 (Doku):** die Begruendung steht jetzt auch im Kopf von `gui/widgets/adaptive_terrain_mesh.py`, also dort, wo man sie sucht - samt Verweis auf `terrain_remesh.py` und den Vektorweg.

| **[x]** | **6.23** | **3D-Darstellung der globalen Siedlungsuebersicht** *(Nutzer-Vorgabe 2026-08-13 nach der Sichtpruefung: "3D Settlements global sollte jetzt umgesetzt werden")*. **Warum bisher nichts zu sehen war:** `_render_settlement_markers()` (`map_display_3d.py`) war seit jeher ein **leerer TODO-Stub** - die drei Layer `settlements`/`landmarks`/`roads` wurden zwar gesetzt und abgefragt, gezeichnet hat die Funktion nie etwas. Kein Fehler im Log, keine Ausnahme: sie holt die Daten und kehrt zurueck. **Umgesetzt** ueber denselben Alpha-Skin-Weg wie Regionen/Kuestentypen (6.1) statt ueber echte 3D-Marker-Geometrie: neue `rasterize_settlements_rgba()` (`map_display_2d.py`) zeichnet Staedte (rot/Kreis), Landmarken (gold/Dreieck), Roadsites (saddlebrown/Quadrat), Landwege (darkorange) und Seewege (royalblue gestrichelt) auf eine transparente RGBA-Textur - **exakt dieselben Farben und Marker wie die 2D-Seite** (`overlay_settlements()`/`overlay_roads()`), eine zweite Farbwahl waere eine zweite Wahrheit. Neuer 3D-Layer `settlement.uebersicht` mit eigenem `_render_settlement_uebersicht()`, gecacht ueber dieselbe Objekt-Identitaets-Logik wie 6.21 (kein Neurasterisieren je Frame). `SettlementTab.apply_3d_overlays()` fuellt ihn und folgt dabei DENSELBEN vier Checkboxen wie die 2D-Ansicht - was in 2D aus ist, fehlt auch auf dem Skin. Marker sind bewusst groesser als in 2D (s=40 -> s=110): die 3D-Textur hat map_size (bis 1024 px) statt der wenigen hundert Pixel des 2D-Canvas. **Headless verifiziert:** Rasterisierung liefert korrektes (H,W,4) uint8, Hintergrund transparent (4.1 % gedeckte Pixel bei typischer Belegung), alle vier Objektarten an ihrer erwarteten (x,y)-Stelle (prueft zugleich die Zeile==y-Konvention), Leerfall vollstaendig transparent; 3D-Dispatch mit gemocktem GL: erster Frame laedt hoch und zeichnet, zweiter Frame zeichnet ohne erneuten Upload (Cache greift), neue Daten loesen genau einen neuen Upload und genau ein `glDeleteTextures` aus, `renderMode` ist 5 (der einzige Zweig mit Alpha-Behandlung, siehe 6.1). Regressionsfrei (`smoke_test_display_2d.py` alle 30 Darstellungen, `smoke_test_layer_2d_3d_parity.py`, `smoke_test_settlement_sites.py`, `smoke_test_adaptive_terrain_mesh.py`). **LIVE BESTAETIGT 2026-08-13:** "3D staedte sieht gut aus." | 1 |
| **[x]** | **6.24** | **Kleinere Anzeige-Korrekturen im Siedlungsreiter** *(beide Nutzer-Vorgaben aus der Sichtpruefung 2026-08-13)*. (a) **Regionsfaerbung war zu subtil** ("etwas zu subtil mach mal 20% mehr von der region-spezifischen farbe", nach dem ersten Versuch mit 0.30 dann ausdruecklich "nein mach mal 0.4") - Deckkraft des Regionen-Overlays im globalen Siedlungsreiter von `alpha=0.25` auf **`0.40`** angehoben. Bleibt bewusst deutlich unter dem Terrain-Reiter (0.55), damit Staedte und Wege im Vordergrund bleiben - die urspruengliche Begruendung fuer den subtilen Ton gilt weiter, sie war nur zu weit getrieben. (b) **Civ-Value und Potential Field aus dem GLOBALEN Reiter entfernt** ("sind NUR regional. koennen hier entfernt und bei regional hinzugefuegt werden") - beides sind FLAECHIGE Felder, und der globale Reiter zeigt seit 2026-08-10 bewusst nur Punkt-/Linienhaftes (dieselbe Trennung wie schon bei Stadtgrenze und Plot-Feingewebe). | 0.5 |
| **[~]** | **6.25** | **3D-Ansicht zeichnet beim Umschalten nicht sofort, erst nach einer Mausbewegung** *(Nutzerbefund 2026-08-13: "klicke auf 3D-View (hier wird dann nie automatisch schon 3D geladen oder es wird nicht dargestellt, erst wenn ich mich bewege ist es zu sehen)")*. **Untersucht:** `update_heightmap()` ruft am Ende bereits `self.update()` - die Anforderung geht also raus. Sie verpufft aber vermutlich, weil `switch_view()` (`base_tab.py`) sie ausloest, BEVOR Qt den `setCurrentIndex()`-Wechsel des Anzeigestapels verarbeitet hat: das QOpenGLWidget ist zu dem Zeitpunkt formal noch nicht sichtbar, und Qt verwirft Neuzeichnen-Anforderungen fuer unsichtbare Widgets. Das erste `paintGL()` kommt dann erst durch das naechste Ereignis - in der Praxis die erste Mausbewegung, genau wie berichtet. **Behoben** durch ein zweites `update()` am ENDE von `switch_view()`, nach `update_display_mode()`, wo der Stapelwechsel abgeschlossen ist (auf Wrapper und inneres Widget, kostet nichts - Qt fasst mehrere Anforderungen zu einem Neuzeichnen zusammen). **[~] statt [x]:** headless nicht pruefbar, ob es wirklich greift - die Erklaerung ist plausibel und deckt den berichteten Ablauf, bewiesen ist sie erst am Bildschirm. Sollte es weiterhin auftreten, ist die naechste Spur, ob das Mesh zu diesem Zeitpunkt ueberhaupt schon steht. | 0.5 |
| **[x]** | **6.26** | **Ein- und Ausschalten von Settlements/Roads dauert 1-2 s** *(Nutzerbefund 2026-08-13, ausdruecklich als unwichtig eingestuft: "was sicher noch optimiert werden koennte (aber wen interessierts, fuer viel viel spaeter eintragen). ausser es geht ganz einfach und schnell")*. Jede Checkbox-Aenderung laeuft durch den vollen `update_settlement_display()`-Pfad, der die komplette 2D-Figur neu aufbaut (matplotlib-Neuzeichnung der Basiskarte samt aller Overlays), obwohl sich nur die Sichtbarkeit einzelner Overlays aendert. Sauberer waere, die Overlay-Kuenstler einmal anzulegen und nur ihr `set_visible()` umzuschalten. **Bewusst nicht jetzt angegangen** - es ist genau die Sorte Umbau, die an matplotlib-Zustandsverwaltung haengt und dabei leicht neue Fehler einbringt, fuer einen Gewinn, den der Nutzer selbst als unwichtig bezeichnet hat. | 2 | **ERLEDIGT 2026-08-24:** gemessen 150.3 ms je `update_display()` bei 512 px, davon `canvas.draw()` allein 90.6 ms - und beim Umschalten wird zweimal gezeichnet. Alle 17 `self.canvas.draw()` auf `draw_idle()` umgestellt, Qt fasst sie zusammen. **150.3 -> 42.2 ms**, alle 30 Darstellungen weiter gruen. Die `rasterize_*`-Modulfunktionen bleiben bei `draw()`, sie lesen direkt danach `buffer_rgba()`.

| **[~]** | **6.20** | **UMGESETZT 2026-08-16: Kuestenlinie als Breakline statt rasterausgerichteter Punkte** *(Nutzeridee 2026-08-13)*. Nutzer: "kann man das mesh nicht aus dem hoehenmodell erstellen und dann in ein mesh umwandeln das mit dem hoehenmodell nichts mehr gemein hat ... ein remesh kann auf jeder tangente passieren". **Die Praemisse stimmt:** exportiert werden 16-Bit-PNG-Rasterebenen (`gui/utils/map_export.py`), das 3D-Netz ist reine Anzeige und an kein Raster gebunden. **Und das Netz IST bereits xyz** - der Vertexpuffer fuehrt `[x,y,z,nx,ny,nz,u,v]`, OpenGL kennt keine Heightmap. Das Speicherformat ist also nicht die Huerde. **Die Huerde ist, WO die Punkte liegen:** jeder Vertex sitzt auf einer Pixelecke, deshalb folgt die Kuestensilhouette dem Raster und wird als Treppe sichtbar. **Wichtige Korrektur zu 6.19:** dort stand, die Treppe sei nicht behebbar - das war zu pessimistisch. Sie ist es, und zwar genau ueber freie Punktsetzung: Kuestenlinie als Kontur ziehen (Marching Squares auf der 0-Linie, ergibt einen glatten, NICHT rasterausgerichteten Linienzug), diesen als erzwungene Kante in eine Triangulierung geben (Constrained Delaunay / "TIN mit Breaklines", in GIS Standard). Was dadurch NICHT verschwindet, ist die 90-Grad-Steilheit selbst - die steckt in den Hoehenwerten, nicht in der Vernetzung. **Kosten:** scipy kann Delaunay, aber KEIN Constrained Delaunay - zusaetzliche Bibliothek oder Eigenbau noetig: der Quadtree (6.16) wuerde groesstenteils ersetzt: Shader, Normalen und UVs blieben unveraendert. **War zurueckgestellt**, weil sich die Kuestengeometrie durch 3.10 ohnehin aenderte. Das ist inzwischen durch, deshalb jetzt gebaut.

### UMGESETZT 2026-08-16: `gui/widgets/kuesten_mesh.py`

**Bestaetigt vorab, was die Praemisse war** (gemessen, nicht angenommen): das adaptive Quadtree-Mesh (6.16) hat **alle** Vertices auf Pixelecken - Abweichung 0.000004 px. Es fasst Rasterzellen zusammen, es setzt nichts frei. Die Treppe an der Kueste kommt genau daher.

**Verfahren:** Kuestenlinie per `skimage.measure.find_contours` auf der 0-Linie (Marching Squares, interpoliert also ZWISCHEN den Pixeln), gleichmaessig entlang der Bogenlaenge abgetastet, dazu ein gestreutes Innengitter, das nahe der Kueste ausduennt - alles zusammen durch `scipy.spatial.Delaunay`.

**KEIN Constrained Delaunay, und das ist eine bewusste Naeherung.** Eine Triangulierung, die die Kuestenkanten ERZWINGT, braeuchte `triangle` (nicht installiert, geprueft) oder Eigenbau; scipy kann nur unconstrained. Der Ausweg: die Kuestenpunkte DICHTER setzen (1.2 px) als das Innengitter (6.0 px) - dann verbindet Delaunay bevorzugt Konturnachbarn. **Weil das nur wahrscheinlich und nicht garantiert ist, misst `kuestentreue()` nach**, welcher Anteil der Konturkanten tatsaechlich Dreieckskante wurde. Ohne diese Messung waere der Unterschied zum echten Verfahren nicht zu bemerken.

**DREI FEHLVERSUCHE BEI DERSELBEN FRAGE - alle gemessen, keiner erraten.** Die Kuestenvertices sollen auf Hoehe 0 liegen; sie lagen es nicht:

| Versuch | Vorgehen | groesste Abweichung |
|---|---|---:|
| 1 | Douglas-Peucker ausduennen, dann Zwischenpunkte auf der **Geraden** einfuegen | **97 m** |
| 2 | Douglas-Peucker ausduennen, dann entlang der **ausgeduennten** Linie abtasten | **88 m** |
| 3 | direkt auf der Originalkontur abtasten, Hoehe **nachschlagen** | **75 m** |
| 4 | direkt abtasten, Hoehe **setzen** | **0.00000 m** |

Versuch 2 sah aus wie die Behebung von 1, war aber derselbe Fehler: **jede Ausduennung ersetzt die Kontur durch Sehnen, und Sehnen liegen nicht auf der Kueste.** Versuch 3 hatte eine ganz andere Ursache: `find_contours` interpoliert linear entlang Pixelkanten, die Hoehenabfrage bilinear ueber die Flaeche - an einer Steilklippe (800 m ueber wenige Pixel) laufen beide auseinander. Die Loesung war die einfachste: ein Punkt aus der 0-Kontur LIEGT auf Hoehe 0, man muss sie nicht nachschlagen, sondern setzen.

**Gemessen (echtes Weltgelaende, Seed 20260804):**

| | 256 px | 384 px | 512 px |
|---|---:|---:|---:|
| Vertices | 3195 | 6231 | 10328 |
| Dreiecke | 6266 | 12276 | 20422 |
| Kuestentreue | 98.7 % | 99.2 % | 99.1 % |
| Kuestenvertices frei vom Raster | 99.9 % | 99.7 % | 99.8 % |
| Bauzeit | 0.02 s | 0.05 s | 0.09 s |

Bei 512 px: **20422 Dreiecke gegen 131709 des Quadtree-Meshes (6.4x weniger) und 522242 des vollen Gitters (25x weniger)** - bei gleichzeitig glatter Kueste. Topologie sauber (keine Kante von mehr als zwei Dreiecken), Normalen normiert, Werte endlich.

**Nebenbefund:** das Kuesten-Mesh braucht **keine Zweierpotenz** als Kantenlaenge. `build_adaptive_mesh()` liefert bei 384 px `None` (Vorbedingung aus 6.16), das neue Verfahren laeuft dort.

Gesichert in `tests/smoke_test_kuesten_mesh.py` (sechs Gruppen, echte Kartengroessen): Hoehe-0-Zusicherung, Freiheit vom Raster **mit Gegenprobe am Quadtree**, Kuestentreue, Topologie, Abdeckung und Groessenvergleich, Randfaelle (Karte ohne Kueste, Karte ganz unter Wasser). Regressionsfrei (`smoke_test_adaptive_terrain_mesh.py`, `smoke_test_wege_geometrie.py`, `smoke_test_spielkarten.py`, `smoke_test_display_2d.py`, `smoke_test_display_methoden_existieren.py`).

### NACHGEMESSEN 2026-08-16: DAS VERFAHREN IST DEM QUADTREE UNTERLEGEN

Nach dem Bauen habe ich die Hoehentreue gegen die echte Heightmap gemessen -
also das, was ein Terrainnetz eigentlich leisten soll. Ergebnis (512 px, nur
Land, Abweichung der aus dem Netz interpolierten Hoehe):

| Variante | Vertices | Dreiecke | Median | p99 | max | Kuestentreue |
|---|---:|---:|---:|---:|---:|---:|
| Kuesten-Mesh (Innengitter 6 px) | 10328 | 20422 | 11.2 m | 210.9 m | 380.2 m | 99.1 % |
| dito, Innengitter 2 px | 66311 | 132048 | 1.9 m | 76.6 m | 217.5 m | ~99 % |
| **Quadtree** | 65954 | 131709 | **0.0 m** | **5.2 m** | **11.6 m** | — |
| Kueste + ganzer Quadtree | 69822 | 139389 | 0.0 m | 5.2 m | 11.6 m | **25.0 %** |

**Bei GLEICHER Dreieckszahl ist der Quadtree im p99 rund 15-fach genauer**
(5.2 m gegen 76.6 m). Der Grund ist grundsaetzlich: ein gleichmaessiges
Innengitter verteilt Punkte gleichmaessig, der Quadtree setzt sie dorthin,
wo das Gelaende sie braucht. 380 m Abweichung heisst, dass ein Berg schlicht
falsch dargestellt wird.

**Und die naheliegende Rettung funktioniert auch nicht:** nimmt man die
Kuestenpunkte PLUS alle Quadtree-Punkte, stimmt die Hoehe wieder exakt - aber
die **Kuestentreue faellt von 99 % auf 25 %**. Der Grund ist derselbe
Mechanismus, auf dem der ganze Trick beruht: er verlangt, dass die
Kuestenpunkte DICHTER liegen als alles ringsum. Genau an der Kueste ist der
Quadtree aber am feinsten (dort stehen die Klippen), seine Punkte gewinnen,
und Delaunay verbindet sie statt der Kontur.

**Damit ist belegt: ohne echtes Constrained Delaunay sind glatte Kueste und
fehleroptimales Innennetz nicht gleichzeitig zu haben.** Der Dichte-Trick ist
kein Ersatz fuer erzwungene Kanten, er ist ihr Gegenteil - er braucht ein
grobes Umfeld, und ein gutes Terrainnetz ist an der Kueste gerade nicht grob.

### Empfehlung

**Nicht anschliessen.** Der Quadtree bleibt das bessere Terrainnetz. Drei
Wege, falls die Treppe an der Kueste weiterhin stoert:

* **(a) Constrained Delaunay per Edge-Flips selbst bauen** - nach der
  gewoehnlichen Triangulierung die fehlenden Kuestenkanten durch
  Kantentausch erzwingen. Bekanntes, ueberschaubares Verfahren, keine neue
  Abhaengigkeit, headless pruefbar (die Treue-Messung steht ja bereits).
  **Das waere mein Vorschlag.**
* **(b) `triangle` einbinden** (Python-Wrapper um Shewchuk's Triangle).
  Technisch der kuerzeste Weg - aber **Lizenz pruefen, bevor irgendetwas
  darauf aufbaut**: Triangle ist fuer kommerzielle Nutzung nicht frei, und
  dieses Projekt zielt auf ein Spiel.
* **(c) Bei 6.19 bleiben:** hoehere Aufloesung mildert die Treppe, beseitigt
  sie nicht.

Das Modul bleibt erhalten - `kuestenlinien()`, `_entlang_abtasten()` und
`kuestentreue()` sind fuer (a) unveraendert brauchbar, nur die
Delaunay-Stufe muesste ersetzt werden.

### NUTZERLISTE 2026-08-16: VIER ANSAETZE, ALLE VIER GEPRUEFT

Der Nutzer hat vier Verfahren vorgelegt und um ehrliche Pruefung gebeten. Die Ergebnisse, jeweils gemessen:

**1. Constrained Delaunay / TIN mit Breaklines** - sachlich richtig, es ist die saubere Loesung. Braucht Eigenbau (Edge-Flips) oder eine Bibliothek; `triangle` ist nicht installiert, und seine Lizenz ist fuer kommerzielle Nutzung nicht frei - das waere vor jeder Verwendung zu klaeren, da dieses Projekt auf ein Spiel zielt.

**2. Grid Vertex Snapping** - gemessen, mit gemischtem Ergebnis:

| | Silhouette | p99-Hoehenfehler |
|---|---:|---:|
| Quadtree unveraendert | 75.8 % | 5.4 m |
| Snapping + Neutriangulierung | **99.5 %** | 85.6 m |
| Snapping, Topologie fest (wie im Text) | 43.4 % | 110.6 m |

Mit fester Topologie - also so, wie der Text es beschreibt - wird die Silhouette SCHLECHTER. Vermutete Ursache: der Text setzt ein REGULAERES Raster voraus, unseres ist ein adaptives Quadtree mit unterschiedlich grossen Zellen; mehrere Vertices verschiedener Groesse snappen auf denselben Konturabschnitt und ihre alten Verbindungen kreuzen sich. **Wichtige Einschraenkung zu diesem Befund:** meine Umsetzung war grob - alle Vertices im Radius wurden gesnappt (n:1 statt 1:1 je Konturabschnitt), und es wurde nicht geprueft, ob Dreiecke dabei umklappen. **Punkt 2 ist damit nicht widerlegt, nur meine schnelle Fassung davon.** Eine sorgfaeltigere Variante koennte deutlich besser abschneiden.

**3. Spline-basierte Klippen-Meshes** - UMGESETZT, siehe unten. Der pragmatische Treffer, und zwar aus einem Grund, der im Text nicht steht: die Bausteine liegen in diesem Projekt bereits fertig und geprueft (`wege_geometrie.py` macht genau das - zwei Vertexreihen entlang eines Linienzugs).

**4. GPU Distance Field / adaptive Tessellation** - scheidet aus: Tessellation-Shader brauchen OpenGL 4.0, saemtliche Shader dieses Projekts sind `#version 330`.

### UMGESETZT: `klippenband()` in `gui/widgets/kuesten_mesh.py`

**Das ist die eigentliche Antwort auf die Nutzerfrage** ("heightmap soll senkrechte flaechen oder diagonale flaechen oder horizontale flaechen gleichermaassen darstellen koennen"): eine Heightmap kann eine Senkrechte per Definition NICHT - ein z-Wert je (x,y). Zwei Vertexreihen koennen es sehr wohl. Die untere liegt auf der Wasserlinie (z=0), die obere einen Pixel landeinwaerts auf der dortigen Gelaendehoehe; dazwischen spannen zwei Dreiecke eine **echte Wand** auf, so steil wie die Hoehendifferenz es vorgibt - unabhaengig von der Rasteraufloesung, ohne Treppe.

Das Terrain-Mesh bleibt dabei **unangetastet**. Es ist als Hoehenmodell gut (p99 5.4 m) und soll es bleiben; das Band liegt darueber.

**Zwei Entscheidungen, die sonst Fehler geworden waeren:**
* **Die Landseite wird aus der Gelaendehoehe bestimmt, nicht aus der Umlaufrichtung der Kontur.** `find_contours` garantiert keine einheitliche Orientierung - eine geratene Seite haette die Waende ins Meer gestellt, ein Fehler, der auf einem Uebersichtsbild leicht durchgeht. Jetzt werden beide Seiten abgetastet und die hoehere gewinnt.
* **Nur Abschnitte ueber `KLIPPE_MINDESTHOEHE_M` (12 m) bekommen eine Wand.** An einem Flachstrand gibt es keine Klippe, dort waere ein Band nur ein stoerender Streifen.

**Ein Fehler beim Bauen, gefunden und behoben:** die erste Fassung legte zwei Vertices je Konturpunkt an, zeichnete Dreiecke aber nur fuer hohe Abschnitte - an flachen Kuesten blieb dadurch toter Ballast im Puffer stehen (die niedrigste mitgefuehrte Oberkante lag bei 1 m, obwohl erst ab 12 m gezeichnet wird). Jetzt werden nur tatsaechlich verwendete Vertices behalten; der Test prueft das ausdruecklich.

**Gemessen:** 256 px -> 3426 Vertices, 3362 Dreiecke, 1681 Wandabschnitte in 0.01 s. 512 px -> 6630 Vertices, 6460 Dreiecke, 3230 Abschnitte in 0.02 s. Untere Kante exakt auf 0.00000 m, Oberkanten 12-311 m. Gegen das Terrain-Mesh (131709 Dreiecke) ist das Band vernachlaessigbar.

Gesichert in `tests/smoke_test_kuesten_mesh.py` (Gruppe `klippenband`): Wasserlinie exakt auf 0, keine Wand unter der Mindesthoehe, gleich viele Unter- wie Oberkanten-Vertices, keine ungenutzten Vertices, Flachkarte ohne Kueste liefert kein Band.

### ANGESCHLOSSEN 2026-08-16

**Das Band gehoert zum GELAENDE, nicht zu einem Reiter** - es wird deshalb in `_render_terrain_base()` gezeichnet und erscheint damit in JEDER Ansicht, nicht nur im Terrain-Reiter. Gebaut wird es zusammen mit dem Terrain-Mesh in `_generate_terrain_mesh()`: gleiche Voraussetzung (eine Heightmap), gleiche Lebensdauer, also automatisch zwischengespeichert und nicht je Frame neu gerechnet. Nutzt wie die Wegbaender das bestehende `wind_shader_program` - kein neuer Shader.

**Ein Fehler, der ohne Pruefung sehr teuer geworden waere:** die Overlay-Methoden dieser Klasse rufen **kein eigenes `glUseProgram`** - nachgesehen, es gibt im ganzen Datei nur drei solche Aufrufe, alle drei fuer das Wind-Programm. Sie verlassen sich darauf, dass das Terrain-Programm aktiv ist. Da das Klippenband MITTEN in `_render_terrain_base()` laeuft und auf das Wind-Programm umschaltet, haetten alle nachfolgenden Overlays mit dem falschen Shader gezeichnet - ein Fehler, der sich als voellig unerklaerliche Darstellung gezeigt haette und schwer zu finden gewesen waere. Das Terrain-Programm wird jetzt im `finally`-Block wiederhergestellt; der Test prueft die Reihenfolge der `glUseProgram`-Aufrufe ausdruecklich (`[wind, terrain]`).

Zwei weitere Details: **Backface-Culling wird fuer den Draw-Call abgeschaltet** (eine Wand sieht man beim Umrunden der Insel zwangslaeufig von beiden Seiten) und danach wiederhergestellt; die Felsfarbe ist nach der Hoehe abgestuft (unten dunkler), weil eine einfarbige Wand wie eine flache Papierkante wirkt.

**Verifiziert mit gemocktem GL:** genau ein Draw-Call, Culling aus und wieder an, Puffer freigegeben, Terrain-Programm am Ende aktiv, kein Draw ohne Klippendaten. Regressionsfrei (`smoke_test_kuesten_mesh.py`, `smoke_test_wege_geometrie.py`, `smoke_test_adaptive_terrain_mesh.py`, `smoke_test_display_2d.py`, `smoke_test_layer_2d_3d_parity.py`, `smoke_test_display_methoden_existieren.py`).

### AM PROGRAMM GESEHEN UND WIEDER AUSGEBAUT 2026-08-16

**Nutzerurteil nach dem Sichttest:** *"beim verschieben sitzen die klippen nicht richtig ... beim strafen, beim drehen immer ein anderer fehler. es sieht an sich in ruhe auch nicht gut aus, auch wenn es an der richtigen stelle sitzt, weil das alte mesh ja durchschimmert. ich glaube ich will es nicht haben."*

**Der Ausbau ist vollzogen** - `_render_klippenband()`, der Bau im Mesh-Schritt und das Cache-Feld sind aus `map_display_3d.py` entfernt. `klippenband()` bleibt in `kuesten_mesh.py` erhalten (samt Test), wird aber nicht mehr aufgerufen.

**Was der Befund lehrt, und das ist die eigentliche Erkenntnis dieses Punktes:** ein aufgesetztes Band ist grundsaetzlich der falsche Weg, egal wie sauber die Geometrie stimmt. Es hat zwei Fehler, die sich nicht wegparametrieren lassen:

* **Das darunterliegende Mesh schimmert durch.** Die Wand verdeckt die Rastertreppe nicht, sie steht daneben - man sieht beides.
* **Es flackert bei Kamerabewegung.** Zwei Flaechen an fast derselben Stelle konkurrieren um dieselben Tiefenwerte; jede Kamerabewegung entscheidet neu, welche gewinnt.

Beim Bauen hatte ich zusaetzlich gemessen, dass die Landseite entlang einer Kontur **15 mal auf 2810 Punkten umschlaegt** (0.5 %) - jeder Umschlag verdreht das Band um 180 Grad. Das waere behebbar gewesen (Mehrheitsentscheid je Kontur statt Pruefung je Punkt), haette die beiden Grundprobleme oben aber nicht beruehrt.

**Nutzer-Vorgabe fuer den weiteren Weg:** *"ich will ein Remesh, das ist was ich will."* Also kein Aufsatz auf das Rastermesh, sondern ein Netz, dessen Vertices von vornherein dort liegen, wo das Gelaende sie braucht. Recherche dazu siehe 6.30.

**Der frueher hier stehende Delaunay-Ansatz bleibt zum Nichtgebrauch empfohlen** (Messung siehe oben) - `kuestenlinien()`, `_entlang_abtasten()` und `kuestentreue()` werden aber vom Klippenband mitbenutzt, waren also nicht umsonst. `map_display_3d._generate_terrain_mesh()` benutzt weiterhin den Quadtree. Das Umschalten ist der naechste Schritt und braucht eine Entscheidung, die ich nicht allein treffen sollte: das Kuesten-Mesh hat **weniger Vertices im Landesinneren** (Innengitter alle 6 px statt fehlergesteuert), die Gelaendeform abseits der Kueste wird also groeber. Ob das ein guter Tausch ist - glatte Kueste gegen etwas weniger Detail im Bergland - ist eine Sichtentscheidung. Der Regler dafuer ist `INNEN_ABSTAND_PX`.

**Was dadurch NICHT verschwindet:** die 90-Grad-Steilheit selbst (6.19). Die steckt in den Hoehenwerten, nicht in der Vernetzung. Das Verfahren beseitigt die TREPPE entlang der Kuestenlinie, nicht die Wand. | 4 |
| **[~]** | **6.21** | **Overlay-Texturen wurden bei JEDER `paintGL()` neu gerechnet, nicht nur bei neuen Daten** (Nutzerbefund 2026-08-13: "Kuestentyp ruckelt im 3D jetzt wieder. Slope auch noch."). Erst falsch vermutet als Push-Problem (der Nutzer hatte ein Konsolenlog mit `_push_data_to_current_display`-Zeiten mitgeschickt) - die Push-Zeiten waren aber schnell (0.008-0.009 s, 6.18-Fix wirkt), das eigentliche Problem lag im PRO-FRAME-Renderpfad, ausgeloest von jeder Mausbewegung waehrend des Kamera-Drehens, nicht vom Push. Gemessen (1024px): `rasterize_kuesten_archetypen_rgba()` **0.6-0.64 s/Aufruf**, `_colorize_layer()` fuer Slope **0.56-0.59 s/Aufruf** - beide wurden bei jedem einzelnen `paintGL()` neu gerechnet UND jedesmal per `glGenTextures`/`glTexImage2D`/`glDeleteTextures` neu hochgeladen, rund 20-fach ueber einem 30-fps-Bildbudget (33 ms). Betraf nicht nur das neue Kuesten-Overlay, sondern denselben generischen Pfad, ueber den auch Slope/Rock/Hardness/alle Wetter-Layer laufen. **Behoben ueber Objekt-Identitaets-Cache** (`self._overlay_texture_cache`, `map_display_3d.py`): `_render_overlay()` (Scalar-Layer wie Slope), `_render_dict_rgba_overlay()` (Regionen/Kuestentypen) und `_render_plot_boundaries()` (Siedlungs-Plots) pruefen jetzt VOR dem teuren Rechenschritt (nicht nur vor dem Textur-Upload), ob sich das zugrundeliegende Datenobjekt seit dem letzten Frame ueberhaupt geaendert hat (`id()`-Vergleich - anders als der inhaltsbasierte Mesh-Cache aus 6.18, weil `overlay_data[...]` nur bei einem echten `update_overlay_data()`-Push neu zugewiesen wird, ueber wiederholte `paintGL()`-Aufrufe waehrend des Drehens also dasselbe Objekt bleibt). Neue Hilfsmethoden `_overlay_cache_pruefen()`/`_overlay_textur_hochladen()`/`_render_rgba_textur()` fassen den gemeinsamen Cache-/Upload-/Zeichen-Code fuer alle drei Aufrufer zusammen. Ergaenzend `_cleanup_overlay_texturen()` (loescht den ganzen Cache, aufgerufen aus `_cleanup_mesh_buffers()` bei jedem Mesh-Neubau) - ohne das haetten alte Texturen ueber viele Neugenerierungen hinweg im VRAM liegen bleiben koennen, weil eine neue Generierung immer neue Datenobjekte erzeugt und alte Cache-Eintraege dadurch nie wieder getroffen werden. **[~] statt [x]:** headless nur die Cache-Logik selbst pruefbar (Syntax, Regressionssuite gruen inkl. `smoke_test_layer_2d_3d_parity.py`) - ob das Ruckeln im laufenden Programm tatsaechlich verschwindet, muss der Nutzer am Bildschirm bestaetigen (GL-Rendering, siehe CLAUDE.md). | 1 |
| **[x]** | **6.22** | **Monats-Wetterkarten pushen bei jedem Timer-Tick die volle Pipeline neu** (aufgefallen am selben Konsolenlog wie 6.21: wiederholte, ansteigend langsame `weather.precip_map`/`weather.wind_map`/`weather.temp_map`-Pushes, 0.36 s -> 0.85 s). `weather_tab.py`s `_month_cycle_timer` (1000 ms) ruft bei jedem Tick `_on_month_cycle_tick()` -> den vollen `_push_data_to_current_display()`-Pfad, obwohl sich zwischen den Monaten NUR die angezeigte Monatsschicht aendert, nicht die zugrundeliegende Generierung - die Daten fuer alle 6 Monate stehen bereits fest. Empfehlung (noch nicht umgesetzt, Nutzer hat sich dazu noch nicht geaeussert): alle 6 Monatstexturen einmal beim Push vorab bauen/hochladen und der Timer schaltet nur noch um, WELCHE gebunden wird, statt bei jedem Tick neu zu rechnen/hochzuladen. Die ansteigende Dauer selbst (0.36 s -> 0.85 s) ist NICHT aufgeklaert - keine Textur-Leck gefunden (`_render_overlay()` hatte schon vor dem 6.21-Umbau ein korrektes `glDeleteTextures` im `finally`-Block), am ehesten allgemeine Systemlast waehrend der Sitzung, nicht verifiziert. | 1 | **ERLEDIGT 2026-08-24:** `_on_month_cycle_tick()` zeichnet nur noch, wenn `viewport_widget.isVisible()` - gleiche Ueberlegung wie 6.14. Der Monatsindex laeuft im Hintergrund bewusst nicht weiter.

| [ ] | 6.27 | **Staedte/Landmarks/Roadsites als echte 3D-Objekte statt als Textur - und anklickbar** *(Nutzerwunsch 2026-08-13: "die darstellung der wege und staedte auf der karte als 'skin' des meshs funktioniert schon, aber besser waere ja eine hochwerte darstellung auf dem mesh oben drauf ... und man soll dann spaeter staedte und landmarks selecten koennen um etwas ueber die einzelnen staedte landmarks und roadsites, vielleicht auch wege, herauszufinden (laenge der strasse) oder traffic oder so")*

**Der heutige Stand (6.23) ist bewusst ein Skin** - die Marker werden in eine RGBA-Textur gezeichnet und aufs Gelaende gelegt. Das war die richtige Wahl, um ueberhaupt etwas zu sehen (`_render_settlement_markers()` war ein leerer Stub, siehe 6.23), hat aber zwei prinzipielle Grenzen: die Marker liegen FLACH auf dem Hang statt aufrecht zu stehen, und eine Textur kann man nicht anklicken - es gibt kein Objekt, das getroffen werden koennte.

**Teil 1 - aufrechte Marker.** Der uebliche Weg sind **Billboards**: zwei Dreiecke je Marker, im Vertex-Shader immer zur Kamera gedreht, an der Weltposition des Ortes auf Gelaendehoehe. Alle Marker eines Typs in EINEM Puffer (instanziert), damit es bei 30 Staedten plus Landmarks und Roadsites ein Draw-Call bleibt statt hundert. Das braucht ein eigenes kleines Shaderpaar - anders als bei allen bisherigen Overlays laesst sich das NICHT ohne neuen GLSL-Code loesen, weil es echte Geometrie ist und keine Textur auf dem vorhandenen Mesh. Die Icons selbst koennen als kleine Texturatlas-Kacheln kommen (ein Bild mit Stadt/Landmark/Roadsite-Symbolen), dann bleibt es bei einer Textur und einem Draw-Call.

**Teil 2 - Anklicken.** Zwei gaengige Verfahren:

* **Color-Picking** - die Szene ein zweites Mal in einen unsichtbaren Puffer rendern, jedes Objekt in einer eindeutigen Farbe (= seine ID). Beim Klick das Pixel unter der Maus auslesen, Farbe zurueck in die ID uebersetzen. **Einfach, exakt (trifft auch teilweise verdeckte Objekte pixelgenau) und robust gegen jede Kameraposition** - der Mehraufwand ist ein zusaetzlicher Renderdurchgang, der nur beim Klick noetig ist, nicht in jedem Frame.
* **Ray-Casting** - aus der Mausposition einen Strahl in die Szene schicken und gegen die Marker-Positionen pruefen. Kein zweiter Renderdurchgang, aber man muss die Projektionsmathematik selbst korrekt hinbekommen, und bei Billboards (die sich zur Kamera drehen) ist die Trefferflaeche nicht trivial.

**Empfehlung: Color-Picking**, wegen der Robustheit - und weil dieses Projekt bereits eine Erfahrung mit selbstgebauter Projektionsmathematik hat (die Ost-West-Spiegelung der View-Matrix, siehe `_update_view_matrix()`), die fuer Ray-Casting erneut fehleranfaellig waere.

**Teil 3 - was im Infofeld steht.** Die Daten liegen groesstenteils schon vor: Kultur, Rang, Haeuserzahl, Stadttyp (5.16) je Ort; Kategorie und Katalogname je Landmark/Roadsite (5.18/5.20). **Was noch NICHT existiert und eigens gerechnet werden muesste:** Strassenlaenge (aus dem Pfad summierbar, trivial) und "Traffic" - der ist erst dann eine echte Zahl, wenn der Grenznutzen-Ausbau aus **5.21** steht, denn dort entsteht mit `w_ij` und den Netzwegen genau die Information, wie viel Handel ueber welche Kante laeuft. **Vorher waere jede Traffic-Anzeige eine erfundene Zahl.**

**Reihenfolge-Empfehlung:** erst 5.21 (dann gibt es Traffic ueberhaupt), dann Teil 1+2 hier. Teil 1 allein (aufrechte Marker ohne Klickbarkeit) waere aber auch fuer sich schon ein sichtbarer Gewinn und unabhaengig machbar. | 4 |

| **[~]** | **6.28** | **Wege als echte Bandgeometrie statt als Textur - Geometrie steht, Rendering fehlt noch** *(Nutzerwunsch 2026-08-13: "es geht mir darum das ich in der karte was auswaehlen kann und es nicht so schoen aussieht mit den strassen als textur. ich will es etwas schoener haben ist das zuviel verlangt? ich will nicht etwas das ich in godot habe, ich will fuer den editor der nur in python ist ein bisschen schoenere optik")*

**Zuerst eine Korrektur an mir selbst:** ich hatte in 13.7 geschrieben, im Editor bleibe der Textur-Skin "und das ist kein Kompromiss". Das war abweisend und ging am Wunsch vorbei - der Nutzer will den EDITOR schoener haben, und ich hatte ueber Godot doziert. Der Wunsch ist ohne weiteres erfuellbar.

**Warum die Textur die Optik begrenzt** (6.23 zeichnet die Wege in eine RGBA-Textur auf dem Terrain-Mesh): die Textur hat feste Kantenlaenge (map_size), beim Heranzoomen wird der Weg also pixelig - er hat schlicht nicht mehr Bildpunkte. Und eine Textur ist kein Objekt: es gibt nichts, was ein Mausklick treffen koennte.

**Gebaut: `gui/widgets/wege_geometrie.py`** - jeder Weg wird zu einem Band aus echten Dreiecken (zwei Vertices je Pfadpunkt, links und rechts der Laufrichtung), knapp ueber dem Gelaende. Das bleibt bei jedem Zoom scharf, hat echte Breite, laesst sich einfaerben (z.B. nach Traffic aus 5.22) - und jedes Band traegt eine ID, ist also anklickbar. Alle Wege landen in EINEM Vertex-/Indexpuffer plus einer Bereichsliste `(weg_index, index_start, index_anzahl)`; die ist zugleich die Grundlage fuers spaetere Anklicken einzelner Strassen (6.27).

Die Umrechnung Pixel -> Welt ist bewusst dieselbe Formel wie im Terrain-Mesh (`adaptive_terrain_mesh`) - eine zweite Rechnung haette das Band gegen das Gelaende verschoben, unauffaellig genug, um lange unbemerkt zu bleiben.

**Zwei Fehler beim Bauen, beide durch Messen gefunden:**

1. **Das Band versank im Hang.** Der erste Entwurf gab jeder Bandkante ihre EIGENE Gelaendehoehe. Am Querhang liegt der Rand aber bis zu einer halben Bandbreite entfernt und dort tiefer - gemessen lag ein Randvertex **0.002 Welteinheiten UNTER dem Terrain**, obwohl der Schwebeaufschlag 0.006 betrug. Behoben, indem das Band **quer eben** ist: die Hoehe kommt aus dem Maximum von Wegmitte und beiden Raendern. Das ist zugleich das realistischere Bild - eine Strasse wird in den Hang geschnitten und folgt ihm nicht seitlich.
2. **Auf kleinen Karten waere der Weg unsichtbar gewesen.** Bei 256 px (83 m/px) ist ein 60-m-Weg **0.72 Pixel** breit. Neue `MINDEST_BREITE_PX = 2.5` haelt ihn sichtbar; die Metervorgabe bestimmt die Breite auf grossen Karten, die Pixeluntergrenze auf kleinen. Gemessen ergibt das 208 m (256 px), 104 m (512 px), 60 m (1024 px) - auf der grossen Karte also genau die Wirkung, die der Nutzer am bisherigen Skin als passend bezeichnet hat.

Die Breite ist kartografisch ueberhoeht, nicht realistisch - nach ausdruecklicher Klarstellung des Nutzers: *"natuerlich kann die etwas groesser sein als realistisch, ist ja auch bei den bergen, das 1000m berge hohe 4000er symbolisieren"*.

**Verifiziert** (`tests/smoke_test_wege_geometrie.py`, echte Kartengroessen 256/512/1024 auf WELLIGEM Testgelaende - ebenes waere der einfachere, aber nutzlose Fall, weil der Versink-Fehler gerade am Querhang auftrat): kein Vertex unter dem Gelaende, Band an jedem Segment quer eben, Breite trifft die Vorgabe bei jeder Groesse, Indizes gueltig, Bereichsliste lueckenlos, Randfaelle (leere Wegliste, fehlende Heightmap, Weg ueber den Kartenrand) liefern leere Ergebnisse statt Abstuerzen.

### OPTIK NACHGEBESSERT 2026-08-24 (Nutzerwunsch: Methoden 1-3 von 10)

Drei Eingriffe, die zusammen aus dem flachen Rechteck eine Strasse machen:

**1. `glPolygonOffset` statt Weltversatz.** Bisher hielt allein `SCHWEBE_ANTEIL` das Band ueber dem Gelaende - ein Versatz in der WELT. Der hat zwei Nachteile: bei flachem Blickwinkel sieht man UNTER die Strasse, und er muss fuer den schlimmsten Querhang reichen, wirkt also ueberall sonst zu hoch. `glPolygonOffset(-2.0, -4.0)` verschiebt nur im TIEFENPUFFER; das Band liegt geometrisch auf dem Boden und gewinnt trotzdem den Tiefentest. `SCHWEBE_ANTEIL` von 0.0006 auf 0.0001 gesenkt, als blosses Sicherheitsnetz.

**2. Weicher Rand statt Polygonkante.** Neues Vertexattribut `deckung` (Vertexformat damit 7 statt 6 float), 1 auf der Fahrbahn und 0 an den Kanten; `wegband.frag` multipliziert es in die Alpha (mit `sqrt`, sonst wirkt der deckende Teil zu schmal) und verwirft Fragmente nahe 0. Die lineare Interpolation zwischen Schulter und Kante ERZEUGT den Verlauf von allein - kein zusaetzlicher Rechenaufwand.

**3. Querprofil statt flachem Rechteck.** Fuenf Bahnen je Wegpunkt (Kante / Schulter / Scheitel / Schulter / Kante) mit Woelbung `h(t) = w * (1 - t^2)`. Die geometrische Woelbung ist klein (2 % der Bandbreite, gemessen 2.1 m), die NORMALEN werden aber um `NORMALEN_WOELBUNG = 0.40` verkippt - absichtlich viel staerker. So zeigt die Schattierung die Woelbung deutlich, waehrend der Umriss sie nur andeutet; waere beides gleich, muesste man fuer sichtbares Relief eine Wulst bauen, die von der Seite als Wurst auffaellt. Derselbe Kniff wie bei den kartografisch ueberhoehten Berghoehen.

**Dazu, weil es beim Anfassen auffiel: GL-Puffer nur noch einmal je Geometrie.** `_render_wegbaender()` legte bei JEDEM Frame VAO/VBO/EBO neu an, lud die kompletten Baender zur GPU und loeschte alles wieder. Kein Leck - der finally-Zweig raeumte auf -, aber der Cache darueber sparte nur das Rechnen in Python, nicht die Uebertragung (gemessen 1.1 MB bei 40 Wegen). Jetzt haengen die Puffer am Cache; `_wegband_puffer_freigeben()` raeumt sie beim Geometriewechsel und in `_cleanup_mesh_buffers()`.

**Und ein stiller Rueckfall beseitigt:** `_render_wegbaender()` stieg ohne `wegband_shader_program` kommentarlos aus - "der Shader liess sich nicht uebersetzen" war damit von "es gibt gerade keine Wege" nicht zu unterscheiden. Meldet jetzt einmal laut.

**Zwei Messbefunde beim Umbau:**
* **Die Hoehe stuetzte sich auf drei Querstellen** (Mitte, linker, rechter Rand) - ein Grat DAZWISCHEN wurde uebersehen. Jetzt `QUER_STUETZSTELLEN = 9` ueber die volle Breite.
* **Die Laengsglaettung schneidet an Kuppen ins Gelaende**, gemessen 0.7 m bei 256 px. Das ist kein Fehler, sondern was eine echte Trasse tut - und mit dem Tiefenversatz ist es unsichtbar. Die Zusicherung im Test wurde von "kein Vertex unter dem Gelaende" auf "hoechstens 2 m Einschnitt, im Median darueber" GELOCKERT; die Begruendung steht im Test.

**`tests/smoke_test_wege_geometrie.py` angepasst** (6/6 gruen): "Band ist quer eben" ist mit der Woelbung nicht mehr wahr - geprueft wird jetzt SYMMETRIE (beide Haelften gleich hoch, Scheitel am hoechsten, Woelbung trifft die Vorgabe, Deckung 0 an den Kanten und 1 auf der Fahrbahn). Die eigentliche Absicherung, dass die Fahrbahn nicht seitlich mitkippt, bleibt damit erhalten.

**`tests/smoke_test_shader_paths.py` um eine vierte Zusicherung erweitert:** die Varyings von `wegband.vert` und `.frag` muessen zusammenpassen und die Attributplaetze lueckenlos sein. Ein Link-Fehler waere sonst zur Laufzeit unsichtbar. (Erste Fassung war selbst loechrig - `dict()` auf (typ, name)-Paare macht den TYP zum Schluessel, `FragPos` verschwand hinter `Normal`, weil beide vec3 sind.)

**Noch NICHT am Bildschirm bestaetigt** - OpenGL laesst sich headless nicht pruefen.

### RENDERING ANGESCHLOSSEN 2026-08-13

**Kein neues Shaderpaar noetig gewesen.** Das bestehende `wind_shader_program` (`shaders/3d_display/wind_vector.vert/.frag`) nimmt genau Position + Farbe - exakt das, was ein eingefaerbtes Band braucht. Ein eigenes Paar waere doppelte Pflege ohne Zugewinn gewesen; das Terrain-Programm schied aus, weil es Normalen und UVs erwartet, die ein Wegband nicht hat. Neue `_render_wegbaender()` in `map_display_3d.py`, neuer Layer `settlement.wegbaender`.

**Die Wege sind aus dem Textur-Skin HERAUSGENOMMEN.** `SettlementTab.apply_3d_overlays()` rasterisiert jetzt nur noch die Punktobjekte (Staedte, Landmarken, Roadsites) in die Textur und uebergibt die Wege als Geometrie. Laegen beide uebereinander, saehe man die pixelige Texturfassung durch das scharfe Band hindurch - der ganze Zweck der Aenderung waere dahin.

**Zwei Details, die sonst Fehler geworden waeren:**
* **Geometrie-Cache** ueber Objekt-Identitaet der Wegliste plus Heightmap-Form und Skalierung. Die Baender je Frame neu zu bauen waere derselbe Fehler wie bei den Overlays (6.21), nur teurer - dort waren es gemessen 0.6 s je Frame.
* **Backface-Culling wird fuer den Draw-Call abgeschaltet** und danach wiederhergestellt. Die Baender sind wie das Terrain einseitig gewickelt, aber beim Drehen von unten auf eine Strasse zu sehen ist voellig normal - mit aktivem Culling waeren sie dabei verschwunden.

**Verifiziert mit gemocktem GL:** erster Frame laedt hoch und zeichnet, zweiter Frame zeichnet ohne Neuaufbau der Geometrie (Cache greift), ohne Daten gibt es keinen Draw-Call, Culling wird aus- und wieder eingeschaltet. Regressionsfrei (`smoke_test_wege_geometrie.py`, `smoke_test_display_2d.py`, `smoke_test_layer_2d_3d_parity.py`, `smoke_test_display_methoden_existieren.py`, `smoke_test_settlement_sites.py`).

**[~] bleibt, weil GL-Rendering headless nicht beurteilbar ist** (CLAUDE.md): dass die Baender gezeichnet WERDEN, ist belegt - ob sie gut AUSSEHEN und ob Breite und Schwebehoehe im laufenden Programm stimmen, muss der Nutzer sehen. **Noch offen:** das Anklicken (6.27) - die Bereichsliste `(weg_index, index_start, index_anzahl)` aus `baue_wegbaender()` ist dafuer bereits die passende Grundlage, sie wird beim Zeichnen aber noch nicht mitgefuehrt. | 2 |

| **[~]** | **6.29** | **Orte und Wege in der 3D-Ansicht anklickbar** *(Nutzerwunsch 2026-08-13: "man soll dann staedte und landmarks selecten koennen um etwas ueber die einzelnen staedte landmarks und roadsites, vielleicht auch wege, herauszufinden (laenge der strasse) oder traffic oder so")*

**Ich revidiere hier meine eigene Empfehlung aus 6.27.** Dort hatte ich Color-Picking vorgeschlagen (Szene ein zweites Mal in einen unsichtbaren Puffer rendern, jedes Objekt in seiner ID-Farbe, Pixel unter der Maus auslesen). Das ist das robustere Verfahren bei vielen, teils verdeckten Objekten - fuer diesen Fall aber der falsche Aufwand: es braucht einen zweiten Renderpass samt Framebuffer-Verwaltung, es braucht fuer die Punktobjekte ueberhaupt erst Geometrie (Billboards, die es noch nicht gibt), und es ist **headless nicht pruefbar**, weil es ohne GL-Kontext nicht laeuft.

**Gewaehlt: Projektion + Abstandstest** (`gui/widgets/karten_auswahl.py`). Jede Weltposition wird ueber DIESELBEN Matrizen in Bildschirmkoordinaten gerechnet, die auch der Shader benutzt, dann das naechstliegende Objekt zum Mauszeiger gesucht. Das ist reine Mathematik - **vollstaendig headless testbar**, ausreichend fuer einige Dutzend Objekte, und sofort verfuegbar.

**Zwei Entscheidungen, die sonst als Fehler aufgefallen waeren:**
* **Orte haben Vorrang vor Wegen**, wenn beide in Reichweite liegen. An einem Ort enden fast immer mehrere Wege - ohne Vorrang waere eine Stadt am Wegende praktisch nie anklickbar. Wege haben zusaetzlich einen engeren Fangradius (10 statt 18 px), weil sie lang sind und sonst jeden Klick in ihrer Naehe abfangen.
* **Punkte hinter der Kamera liefern NaN** statt eines Treffers. Ohne diese Pruefung erzeugt die Perspektivdivision dort gespiegelte Bildschirmkoordinaten - man haette Objekte im Ruecken angeklickt.

**Drei Fehler beim Bauen, alle durch Pruefen gefunden:**
1. **`self.logger` gibt es in `MapDisplay3D` gar nicht** (nachgesehen, nicht vermutet). Mein `except`-Block haette damit im Fehlerfall SELBST eine Ausnahme geworfen und die eigentliche Ursache verschluckt. Jetzt ueber das vorhandene `rendering_error`-Signal, das auch die uebrigen Methoden nutzen.
2. **Landmarks wurden als "Ort" beschriftet.** `Location.settlement_type` traegt auch bei Landmarks und Roadsites den Vorgabewert `"sonstige"`, der ueber `STADTTYPEN` zu "Ort" wird - ein Steinkreis erschien so mit einem Stadttyp, den er nicht hat. Jetzt nur bei Siedlungen und nur, wenn der Typ nicht der Auffangwert ist.
3. **`
` in einem RichText-Feld** ist kein Zeilenumbruch. Beide Stellen auf `<br>` umgestellt.

**Angezeigt wird**, was tatsaechlich belegt ist - eine Zeile "Einwohner: 0" waere schlechter als gar keine: Stadttyp, Kultur, Rang, Haeuserzahl bei Siedlungen; Katalogname und Lage-Kategorie bei Landmarks/Roadsites; bei Wegen **Laenge in km** (neue `weglaenge_km()`) und Stuetzpunktzahl. **"Traffic" fehlt bewusst noch:** die Zahl entsteht erst, wenn `kanten_traffic()` (5.22) an das gezeichnete Netz angeschlossen ist - vorher waere sie erfunden.

Gesichert in `tests/smoke_test_wege_geometrie.py` (Gruppe `auswahl`): Bildmitte, Achsenrichtungen, NaN hinter der Kamera, Treffer auf Ort und auf Weg, Klick ins Leere ohne Fehler, Ortsvorrang, Weglaenge. Regressionsfrei (`smoke_test_display_2d.py`, `smoke_test_layer_2d_3d_parity.py`, `smoke_test_display_methoden_existieren.py`, `smoke_test_settlement_sites.py`, `smoke_test_stadttypen.py`).

**[~] weil die Wirkung im Programm unbestaetigt ist:** die Trefferrechnung ist exakt geprueft, ob sich das Anklicken im laufenden Fenster gut ANFUEHLT (Fangradius gross genug? richtiges Objekt bei schraeger Kamera?) muss der Nutzer beurteilen. **Ehrliche Grenze des Verfahrens, die auch im Modulkopf steht:** ein Objekt hinter einem Berg wird mitgetroffen - die Tiefe wird berechnet und mitgeliefert, aber es gibt keinen Sichtbarkeitstest gegen das Gelaende. Wer das braucht, kommt an Color-Picking nicht vorbei. | 2 |

| [ ] | 6.30 | **Echtes Remesh statt Rastergitter - Recherche 2026-08-16** *(Nutzer-Vorgabe nach dem gescheiterten Klippenband: "ich will ein Remesh, das ist was ich will. suche mir github projekte die das richtig machen")*

**Der Stand des Problems:** unser adaptives Quadtree (6.16) setzt jeden Vertex auf eine Pixelecke (gemessen 0.000004 px Abweichung). Deshalb folgt die Kuestensilhouette dem Raster und wird als Treppe sichtbar. Alle bisherigen Umgehungen sind gescheitert: Delaunay ueber Kuestenpunkte plus Gitter war beim Hoehenfehler 15-fach schlechter (6.20), Vertex-Snapping verschlechterte die Silhouette, ein aufgesetztes Klippenband flackerte und liess das Raster durchschimmern.

### Gefunden: Delatin (Garland & Heckbert 1995)

**Das ist der Standardalgorithmus fuer genau diese Aufgabe** - "Fast Polygonal Approximation of Terrains and Height Fields". Verfahren: **greedy insertion**. Man beginnt mit zwei Dreiecken ueber die ganze Karte und fuegt iterativ den Punkt mit dem GROESSTEN Hoehenfehler ein, jeweils Delaunay-korrekt. Das Ergebnis ist ein Netz, dessen Vertices **frei liegen** - genau dort, wo das Gelaende sie braucht.

Der entscheidende Unterschied zum Quadtree: der Quadtree kann nur Rasterzellen zusammenfassen, seine Vertices bleiben auf Pixelecken. Delatin waehlt Punkte aus, ist also nicht ans Raster gebunden - und an einer Kuestenklippe ist der Hoehenfehler gross, dort landen die Punkte also von selbst.

| Projekt | Was | Sprache | Lizenz |
|---|---|---|---|
| **[pydelatin](https://github.com/kylebarron/pydelatin)** | Python-Bindings zu `hmm`, **`pip install pydelatin`** | C++/Python | **MIT** |
| [hmm](https://github.com/fogleman/hmm) | die C++-Referenzimplementierung von Delatin | C++ | MIT |
| [pymartini](https://github.com/kylebarron/pymartini) | RTIN/Martini-Port, schneller aber groeber | Cython | ISC |
| [rqtreemesh](https://pypi.org/project/rqtreemesh/) | restricted quadtree - dasselbe Verfahren wie unser 6.16 | C++/Python | - |

**pydelatin ist der klare Kandidat:**
* **MIT-Lizenz** - anders als `triangle` (Constrained Delaunay), dessen Lizenz kommerzielle Nutzung ausschliesst und das deshalb fuer ein Spiel ausschied.
* **Arbeitet auf beliebigen Rastern** - keine Zweierpotenz-Bedingung, an der unser Quadtree bei 384 px scheitert.
* **Steuerbar ueber `max_error` ODER `max_triangles`/`max_points`** - man gibt vor, was man will, statt eine Fehlertoleranz zu erraten.
* Laut Vergleich der Autoren gegenueber Martini: 25 % langsamer, aber **40 % weniger Vertices und Dreiecke** bei gleicher Qualitaet.

**Ehrlich benannt, was die Doku NICHT zusagt:** Breakline-Erhaltung. Delatin optimiert auf den Hoehenfehler, nicht auf die Silhouette. Die Erwartung ist, dass die Kuestenkante als Nebenwirkung sauber wird (dort ist der Fehler am groessten), **aber das ist eine Vermutung und muss gemessen werden** - mit derselben `kuestentreue()`, die schon existiert.

**Naechster Schritt:** `pip install pydelatin` in die Projekt-venv, dann gegen das Quadtree messen - Hoehenfehler, Dreieckszahl, Bauzeit und Kuestentreue, mit den Verfahren aus `tests/smoke_test_kuesten_mesh.py`. **Eine neue Abhaengigkeit gehoert vorher abgesprochen**, deshalb hier erst die Recherche.

### Gemessen am 2026-08-16: der eigene Nachbau VERLIERT, und zwar deutlich

pydelatin liess sich nicht installieren - kein Windows-Wheel fuer Python 3.13 (letzte Veroeffentlichung 2021, `pip download --only-binary` findet nichts), und aus der Quelle bauen verlangt "Microsoft Visual C++ 14.0 or greater". Also wurde das Verfahren in `gui/widgets/delatin_mesh.py` selbst gebaut - mit **einem Zugestaendnis**: statt einen Punkt einzufuegen und lokal nachzutriangulieren (ohne C++-Bibliothek zu aufwendig) werden je Runde 50 % der vorhandenen Punktzahl auf einmal eingefuegt und danach einmal komplett neu trianguliert.

**Genau dieses Zugestaendnis bringt das Verfahren um.** Vergleich bei EXAKT gleicher Vertexzahl (512 px, Seed 20260804, 65954 V):

| | Quadtree (jetzt) | Delatin-Nachbau |
|---|---:|---:|
| Bauzeit | **0.2 s** | 126.3 s |
| RMS-Hoehenfehler Land | **1.34 m** | 6.89 m |
| p99 Land | **5.29 m** | 23.08 m |
| groesster Fehler | **12.25 m** | 201.84 m |
| Vertices ueber Land | **78.8 %** | 33.8 % |

Die letzte Zeile ist die Erklaerung: Land ist 29.1 % der Karte, das Quadtree steckt trotzdem 78.8 % seiner Vertices dorthin - der Nachbau nur 33.8 % und verschwendet zwei Drittel auf den offenen Meeresboden. Im Drahtgitterbild ist das unuebersehbar. Ursache ist die Stapelverarbeitung: die Fehlerschaetzung ist beim Einfuegen des halben Stapels laengst veraltet, und die letzte Runde setzt ~22000 Punkte auf einmal nach einer Rangliste, die nicht mehr stimmt.

**Zusaetzlich gefunden und behoben:** `waehle_punkte()` gab ein zu kleines Netz kommentarlos zurueck. Bei `stichprobe`=s gibt es nur (H/s)·(W/s) waehlbare Positionen; mit s=3 auf 512x512 war bei 28479 von angeforderten 65954 Punkten Schluss - ohne Hinweis, aussehend wie ein fertiges Ergebnis (dieselbe Falle wie beim GPU-Rueckfall und beim Quadtree-`2^n+1`, siehe CLAUDE.md). Die Stichprobe verfeinert sich jetzt selbst, und `stats["abbruch"]` sagt, warum Schluss war.

**Bewertung:** der Nachbau wird NICHT angeschlossen - er ist bei gleicher Vertexzahl 5-fach ungenauer auf Land und 600-fach langsamer. Damit ist das VERFAHREN nicht widerlegt, nur diese Umsetzung; die Frage haengt allein an der Einzelpunkt-Einfuegung, also an C++. **Entscheidung des Nutzers noetig:** Visual-C++-Buildtools installieren (~2 GB, Eingriff in die Arbeitsumgebung) und `hmm` direkt binden - oder die Rastertreppe an der Kueste hinnehmen. | 3 |

| [ ] | 6.31 | **Wege ins Gelaende einschneiden statt darueber schweben lassen** *(Nutzerfrage 2026-08-16 nach einem Fremdvorschlag: Wege/Fluesse als Vektorbaender per Constrained Delaunay ins Mesh einbetten)*

**Zuerst der Stand, damit nichts doppelt gebaut wird — fuer FLUESSE ist das Verfahren hier laengst umgesetzt, ohne CDT.** `taeler_eingraben()` in `core/terrain_weltfluesse.py` (gerufen aus `core/terrain_generator.py:1965`) schneidet das Flussbett direkt in die Heightmap: `anteil = 0.22 + 0.78 * gebiet**0.40` skaliert die Breite mit dem Einzugsgebiet, also mit den Zufluessen, und `tiefe_feld * gebiet**0.30` die Tiefe. Dazu eine Erosionsbasis (kein Schnitt unter den Meeresspiegel) und ein `grey_erosion`-Kegel gegen Messerschneiden zwischen nahen Laeufen. Das Quadtree-Mesh folgt dem automatisch, weil es auf der Heightmap arbeitet und bis `min_leaf_size=1` unterteilt.

**Was FEHLT, ist dasselbe fuer Wege.** `gui/widgets/wege_geometrie.py` LIEST die Heightmap nur (`_hoehe_an`) und legt ein Band mit `SCHWEBE_ANTEIL = 0.0006` darueber. Deshalb musste das Band quer flachgelegt werden (`hoehe_band = max(mitte, links, rechts)`) — es versank sonst am Querhang. Das ist die Umkehrung: statt das Band an das Gelaende anzupassen, das Gelaende an den Weg.

**KORREKTUR 2026-08-16 (Nutzereinwand):** oben stand, fuer Wege brauche es kein CDT, weil das Band eigene Geometrie mit scharfer Kante sei. Der Nutzer hat widersprochen — er will die Zerschneidung des Gelaendes ENTLANG der beiden Wegkanten, nicht ein Band ueber eingeebnetem Raster. Der Einwand ist berechtigt: Einschneiden macht das Gelaende unter dem Weg flach, aber die Gelaendedreiecke enden weiterhin an Pixelecken, nicht an der Wegkante. Der Uebergang bleibt eine Rastertreppe, nur eine flachere. Wer die Kante will, braucht die Zwangskante. Siehe 6.32.

**Geprueft, was an CDT hier verfuegbar ist** (2026-08-16): `shapely` 2.1.2 mit GEOS 3.13.1 ist installiert und hat `constrained_delaunay_triangles` — schnell (20000 Randpunkte in 0.07 s), nimmt Polygone mit Loechern und MultiPolygone. **Setzt aber KEINE Steiner-Punkte:** ein leeres Quadrat ergibt 2 Dreiecke aus 4 Ecken, innere Punkte kommen nicht dazu. Damit taugt es fuer das Wegband SELBST, aber nicht zum Einbetten in ein Gelaende mit Relief — dafuer braucht es Punktwolke PLUS Zwangskanten. `triangle` fehlt und ist lizenzbedingt ausgeschlossen; `artem-ogre/CDT` und `PythonCDT` haetten den Buildtools-Blocker.

**Drei Fallen, die vorher geklaert sein muessen:**
1. **Henne und Ei.** Wege werden per A* auf einem Kostenfeld AUS der Heightmap geroutet. Schneidet man danach ein, aendern sich Hangwerte — speist man das zurueck, laeuft der Graph im Kreis. Der Schnitt muss NACH dem Routing liegen und darf nicht Eingang des Routings werden.
2. **Nachgelagerte Verbraucher.** Die Heightmap traegt Hydrologie, Biome und Siedlungseignung. Ein Wegschnitt darf keinen Fluss aufstauen und keine Biomgrenze verschieben — also entweder in eine reine Anzeigekopie schneiden oder den Knoten hinter alle Verbraucher haengen. Das ist eine Entscheidung ueber die Graphreihenfolge, keine Kleinigkeit.
3. **Wege kreuzen Fluesse.** Dort darf nicht geschnitten, sondern muss ueberbrueckt werden — ein Querschnitt durchs Flussbett wuerde es zuschuetten, und `taeler_eingraben` lief vorher. | 3 |

| [ ] | 6.32 | **Der Fluss selbst fehlt — nur das Tal ist da** *(Nutzerfrage 2026-08-16: "schlussendlich haben wir ja ganz unten dann den fluss, der wenige meter breit ist und eine einwoelbung. haben wir den so drin?")*

**Nein. Und in der Heightmap kann er auch gar nicht drin sein.** Gemessen:

| Kartengroesse | Meter je Pixel | schmalstes Tal, das `taeler_eingraben` schneidet (2.5 px) |
|---|---:|---:|
| 256 px | 83.2 m | 208 m |
| 512 px | 41.6 m | 104 m |
| 1024 px | 20.8 m | 52 m |

Ein 5 m breiter Lauf ist bei 41.6 m/px physikalisch nicht darstellbar — er ist ein Achtel Pixel breit. `taeler_eingraben()` bildet also das TAL ab, und das ist auch alles, was es kann. Zusaetzlich: `grep -n "fluss\|river" gui/widgets/map_display_3d.py` findet **keine einzige Stelle** — in der 3D-Ansicht gibt es derzeit ueberhaupt keine Flussgeometrie.

**Folgerung: der Lauf muss eigene Vektorgeometrie werden, genau wie die Wege.** Das Werkzeug dafuer steht schon: `gui/widgets/wege_geometrie.py` baut Baender aus einem Pfad (`band_aus_pfad`, `baue_wegbaender`). Ein Flussband ist dieselbe Maschinerie mit zwei Unterschieden — statt quer flach (`max(mitte, links, rechts)`) ein nach unten gewoelbtes Profil, und die Breite kommt aus dem Einzugsgebiet statt aus einer Konstanten. Die Groesse liegt bereits vor: `gebiet[i]` in `taeler_eingraben()` ist genau der kumulierte Zufluss, mit dem dort schon Talbreite und -tiefe skaliert werden.

**Keine neue Abhaengigkeit noetig.** Das ist der billigste echte Fortschritt aus dieser Runde. | 2 |

| [ ] | 6.33 | **Remesh: die Richtung, die noch nicht probiert wurde — von oben statt von unten**

**Der Nutzereinwand vom 2026-08-16, der ernst zu nehmen ist:** *"irgendwie hab ich das gefuehl du lehnst alles gute ab weil die umsetzung nur so halbherzig ist."* Das trifft zu. Von vier Absagen gehen drei auf eigene Kompromisse zurueck, nicht auf das Verfahren: das Kuesten-Delaunay (6.20) hatte kein CDT, das Vertex-Snapping war grob, der Delatin-Nachbau (6.30) fuegte 50 % der Punktzahl im Stapel ein statt einzeln — und in allen drei Faellen stand der Kompromiss als Ursache im Bericht, das Fazit lautete trotzdem "nicht anschliessen".

**Alle bisherigen Versuche waren VON UNTEN: grob anfangen, Punkte einfuegen.** Sie scheitern alle an derselben Stelle — inkrementelles Delaunay ist in reinem Python nicht bezahlbar. Die andere Richtung ist ungetestet:

**Von oben (Decimation).** Man startet mit dem VOLLEN Gitter (512x512 = 262144 Vertices, 522242 Dreiecke) — das hat keine Treppen, es ist die feinste Darstellung, die wir haben. Dann faellt man Kanten zusammen nach der Quadric Error Metric (Garland & Heckbert 1997, der Nachfolgeraufsatz zu dem von 6.30). Der Punkt: **der verschmolzene Vertex landet an der Stelle, die den Quadric-Fehler minimiert — also NICHT auf einer Pixelecke.** Genau das, was der Nutzer verlangt: *"wenn wir die treppen loswerden wollen muessen wir die vertices verschieben."* An einer Klippe kostet das Zusammenfallen quer zur Kante viel, also bleibt die Kante stehen und die Dreiecksseiten legen sich von selbst an sie an — Breakline-Erhaltung als Nebenwirkung, ohne Zwangskanten.

### Kandidaten, auf Installierbarkeit GEPRUEFT (nicht nur nachgeschlagen)

Nach der pydelatin-Erfahrung (kein Wheel fuer 3.13, Buildtools noetig) wurde jeder Eintrag mit `pip download --only-binary=:all:` gegen cp313/win_amd64 getestet:

| Paket | Was | Lizenz | Wheel cp313 |
|---|---|---|---|
| **[fast_simplification](https://github.com/pyvista/fast-simplification)** | QEM-Decimation in C++ (nach Sven Forstmann) | **MIT** | **ja**, 0.2.0 |
| **[vtk](https://vtk.org)** | `vtkDelaunay2D` mit **Zwangskanten ueber beliebiger Punktwolke**, dazu `vtkQuadricDecimation` | **BSD-3** | **ja**, 9.7.0 |
| shapely (schon da) | `constrained_delaunay_triangles`, sehr schnell | BSD/LGPL | schon installiert |
| mapbox_earcut | Polygon-Triangulierung, kein Delaunay | ISC | ja, 2.0.0 |
| pymeshlab | MeshLab-Bindung, viele Filter | **GPL — fuer ein Spiel unbrauchbar** | ja |
| open3d | QEM-Decimation | MIT | **nein** (kein 3.13) |
| PythonCDT | Bindung an artem-ogre/CDT | MPL-2.0 | **nein** |
| pydelatin | greedy insertion, das Original zu 6.30 | MIT | **nein** (Stand 2021) |

**Empfehlung, in dieser Reihenfolge:**
1. **fast_simplification** zuerst — MIT, ein Wheel, eine Funktion. Zu messen ist genau eines: **verschiebt es die Vertices wirklich, oder waehlt es nur einen der beiden Kantenendpunkte?** Nur im ersten Fall verschwinden die Treppen. Das ist eine Viertelstunde Arbeit und entscheidet die ganze Frage.
2. **vtk** falls Zwangskanten gebraucht werden — es ist das einzige geprueft installierbare Paket, das Punktwolke PLUS Zwangskanten kann, und loest damit auch 6.31 (Wege ins Gelaende schneiden).

**Beide sind neue Abhaengigkeiten und brauchen die Zustimmung des Nutzers** (vtk ist mit ~100 MB deutlich schwerer als fast_simplification).

### GEMESSEN 2026-08-16: fast_simplification verschiebt die Vertices — die Antwort ist JA

`fast_simplification` 0.2.0 installiert (`--no-deps`, damit die numpy-Bindung haelt; danach geprueft: numpy 2.4.6, `import numba` laeuft weiter). Volles Gitter 512x512 = 262144 Vertices auf Quadtree-Groesse dezimiert, in den Weltkoordinaten der App (die Quadric-Metrik gewichtet die Achsen gegeneinander, ein anderes Hoehe-zu-Breite-Verhaeltnis gaebe ein anderes Netz):

| | Quadtree (jetzt) | QEM-Decimation |
|---|---:|---:|
| Vertices | 65954 | 66332 |
| Dreiecke | 131709 | 131708 |
| Bauzeit | 0.2 s | **0.4 s** |
| **Abstand zur naechsten Pixelecke, Median** | **0.000000 px** | **0.296685 px** |
| Anteil weiter als 0.01 px von einer Ecke | 0 % | **80.5 %** |
| RMS-Hoehenfehler gesamt | 1.90 m | **1.30 m** |
| davon Land | **1.59 m** | 2.33 m |
| davon Meer | 2.01 m | **0.42 m** |

Alle bisherigen Versuche waren von unten (Punkte einfuegen) und scheiterten am inkrementellen Delaunay; von oben (Kanten zusammenfallen lassen) geht es — in 0.4 s, mit MIT-Lizenz und einem fertigen Wheel.

**KORREKTUR 2026-08-16, nachgemessen beim Bau der Testsuite:** hier stand zuerst "damit ist die Kernforderung des Nutzers erfuellt". **Das war zu stark.** Der Median-Versatz von 0.297 px ist ein Mittel ueber die GANZE Karte. Getrennt nach Lage gemessen (512 px, Band von 3 px um die Wasserlinie):

| | Vertices je Kuestenpixel | dort frei verschoben |
|---|---:|---:|
| Quadtree | 0.79 | 0 % |
| Remesh | 0.50 | 39.2 % |

**An der Kueste ist der Anteil frei verschobener Vertices KLEINER als im Kartenmittel** (39.7 % gegen 54.2 %), und es sitzen dort auch weniger Vertices als beim Quadtree. Das ist kein Fehler, sondern die Bauart: QEM verschiebt einen Vertex nur, wenn es zwei verschmilzt — und an einer Kante mit hoher Kruemmung verschmilzt es gerade NICHT, der Vertex ueberlebt also unveraendert auf seiner Pixelecke.

**Folgerung:** das Remesh gewinnt beim Hoehenfehler ueber die Flaeche (siehe Tabelle unten), aber es loest die Kuestenlinie NICHT vom Raster. Wer das zwingend will, braucht die Kuestenlinie als Zwangskante — also Constrained Delaunay ueber einer Punktwolke, und dafuer ist `vtk` das einzige geprueft installierbare Paket (siehe Tabelle oben, loest zugleich 6.31). Ein dritter Bereich "Kuestenband mit eigenem, hohem Budget" waere der billigere Zwischenschritt; die Maschinerie dafuer steht bereits (`remesh_roh` zerlegt schon in zwei Bereiche).

**Zwei eigene Messfehler auf dem Weg, beide korrigiert — und beide haetten das Verfahren zu Unrecht erledigt:**
1. `target_count` zaehlt **Dreiecke, nicht Vertices**. Erster Lauf verglich 33335 gegen 65954 Vertices, also halbes Budget.
2. Der Hoehenfehler wurde ueber eine **neue Delaunay-Triangulierung der Vertices** gemessen statt ueber die echten Dreiecke. Genau das wirft QEMs Staerke weg — die absichtlich langgestreckten Dreiecke entlang der Grate. Mit echter Konnektivitaet kippt das Ergebnis von "RMS 3.81 m, deutlich schlechter" auf "RMS 1.30 m, besser als das Quadtree". Fuer die Messung muss selbst gerastert werden; `matplotlib.tri.TrapezoidMapTriFinder` lehnt das Quadtree-Netz als "invalid" ab (entartete Dreiecke an den Nahtstellen).

**Was noch offen ist, bevor das angeschlossen werden kann:**
* **Normalen und Texturkoordinaten** — `fs.simplify()` liefert nur Punkte und Dreiecke, `build_adaptive_mesh()` gibt interleaved [pos,normal,uv] zurueck. Nachzurechnen wie in `delatin_mesh.baue_delatin_mesh()`.
* **Zustimmung zur Abhaengigkeit** — `fast-simplification` gehoert dann in `requirements.txt`.

### OERTLICH steuerbar — und dadurch besser als global (Nutzerfrage: "nur in bestimmten bereichen? das meer sieht ja gut aus")

**Erst der Weg, der NICHT geht:** zwei verschiedene Netzarten nebeneinander (Quadtree im Meer, QEM an der Kueste) erzeugen an der Naht T-Stuecke und damit Risse — dieselbe Falle wie beim gescheiterten Klippenband.

**Auch nicht gegangen: Wichtung ueber die Eingabe.** Naheliegender Versuch war, die Hoehenachse unter Wasser um k zu stauchen, damit Zusammenfallen dort weniger kostet (streng monoton, also exakt zurueckrechenbar). Gemessen fuer k = 1.0 / 0.5 / 0.25 / 0.10 / 0.03: der Landanteil der Vertices blieb bei 43.7 → 43.5 %, der Landfehler wurde sogar minimal schlechter. **Grund im Quelltext gefunden** (`Simplify.h:385`): `threshold = 0.000000001*pow(iteration+3, agressiveness)` mit `if (t.err[3] > threshold) continue;` — die Bibliothek ist ein **Schwellendurchlauf, keine Prioritaetswarteschlange**. Sie sammelt pro Runde alles unter einer steigenden Schwelle ein und verteilt das Budget dadurch grob gleichmaessig, egal wie die Fehler gewichtet sind.

**Was geht: Bereiche getrennt dezimieren, `preserve_border=True`.** Zerlegt man das volle Gitter an einer Tiefenlinie (hier -25 m) in "offenes Meer" und "Land + Kueste + Schelf", ist die Trennlinie fuer BEIDE Teile Rand — beide behalten dort exakt dieselben Vertices, die Naht ist dicht, kein T-Stueck. Danach je Bereich ein eigenes Dreiecksbudget:

| Variante | Land RMS | Land p99 | Meer RMS | Vertices ueber Land | Zeit |
|---|---:|---:|---:|---:|---:|
| Quadtree (jetzt) | 1.59 m | 6.05 m | 2.01 m | 78.8 % | 0.2 s |
| QEM global | 2.33 m | 7.81 m | 0.42 m | 43.7 % | 0.4 s |
| QEM, Meer 20 % des Budgets | 1.63 m | 6.09 m | 0.53 m | 57.3 % | 1.7 s |
| **QEM, Meer 10 %** | **1.07 m** | **3.51 m** | **0.94 m** | 64.5 % | 1.6 s |
| QEM, Meer 5 % | 0.94 m | 3.07 m | 2.63 m | 67.9 % | 1.5 s |

**Bei 10 % ist es auf Land UND im Meer zugleich besser als das Quadtree**, bei gleicher Vertexzahl. Der p99 auf Land faellt von 6.05 m auf 3.51 m, also fast auf die Haelfte — das ist der Wert, der die sichtbaren Ausreisser beschreibt.

**Preis, ehrlich benannt:** die Vertices auf der Trennlinie bleiben rastergebunden (`preserve_border` haelt sie fest). Der Anteil frei verschobener Vertices sinkt von 80.5 % auf 54.2 %. Die Trennlinie liegt aber bei -25 m, also unter Wasser und unsichtbar — die eigentliche Kuestenlinie bei 0 m liegt INNERHALB des feinen Teils und wird weiter frei verschoben.

**Nebenbefund fuer die 9 Spielkarten (6.29):** genau dieses `preserve_border`-Verfahren loest auch das Zerschneiden der Welt in Teilkarten ohne Risse an den Kanten.

### Gebaut 2026-08-16: Modul, Haken im Display, Werkstatt und Testsuite

* **`gui/widgets/terrain_remesh.py`** — liefert `(vertices, indices, stats)` im selben interleaved Format wie `build_adaptive_mesh()`, ist also an derselben Stelle einsetzbar. Normalen werden bilinear aus dem Hoehenfeld abgetastet (nicht aus den Dreiecken gemittelt), sonst saehe das Remesh facettiert aus waehrend das Quadtree glatt schattiert ist — der Vergleich waere dann der zweier Beleuchtungen statt zweier Netze. Zwischenspeicher ueber den Heightmap-Inhalt wie beim Quadtree.
* **`MapDisplay3D.setze_mesh_bauer(bauer)`** — Haken fuer einen alternativen Netzbauer, Voreinstellung `None` = **unveraendertes Verhalten der App**. Liefert der Bauer nichts, wird das laut gemeldet und auf das Quadtree zurueckgefallen.
* **`tools/mesh_werkstatt.py`** — Fenster mit der ECHTEN 3D-Anzeige der App rechts und links Radio-Buttons fuer vier Netzarten (Quadtree / Remesh getrennt / Remesh global / Delatin), Reglern fuer Budget, Meeranteil, Tiefenlinie, Aggressivitaet und Quadtree-Toleranz, dazu Kartengroesse 256/384/512. Nach jedem Bau stehen Vertexzahl, Dreieckszahl, Bauzeit und der Versatz zur Pixelecke da; auf Wunsch auch der Hoehenfehler nach Land/Meer getrennt. Knopf "Alle Netzarten vergleichen" schreibt eine Tabelle in die Konsole.
* **`tests/smoke_test_terrain_remesh.py`** — 7 Gruppen, alle gruen. Prueft mit den ECHTEN Kartengroessen 256/384/512 (384 ist wichtig: dort faellt das Quadtree auf das Gleichmaessig-Gitter zurueck, das Remesh nicht). Kern ist die **Dichtheit der Naht** — keine Kante an mehr als zwei Dreiecken, keine doppelten Vertexpositionen, keine entarteten Dreiecke — also genau das, woran das Klippenband gescheitert ist.

**Zwei eigene Messfehler beim Bau der Testsuite, beide gefunden und im Test dokumentiert:**
1. Abstand zur Kuestenlinie als `minimum` der beiden Distanztransformationen gerechnet — das ist ueberall 0 und waehlte damit die ganze Karte aus. Aufgefallen, weil die "Kuestenwerte" bis auf die Nachkommastelle mit den Gesamtwerten uebereinstimmten.
2. Die erste Zusicherung verlangte einen Median-Versatz ueber ALLE Vertices von mehr als 0.01 px und schlug bei 256 px fehl. Untaugliche Kennzahl: bei 75 % Reduktion bleibt gut die Haelfte der Vertices von jedem Kollaps unberuehrt und liegt weiter auf ihrer Pixelecke — dort ist das Gelaende flach, es gibt nichts zu verbessern.

**Beim Nachpruefen "laeuft das ueberhaupt?" gefunden und behoben** (die Werkstatt war vorher nur kompiliert und importiert, nicht ausgefuehrt): `gelaende_neu()` stand am Ende von `__init__` und stuerzte mit `NullFunctionError: glBindVertexArray` ab — `update_heightmap()` schreibt das Mesh sofort in GL-Buffer, den OpenGL-Kontext des Widgets gibt es aber erst beim ersten Anzeigen. Das waere auch auf einem echten Bildschirm passiert, nicht nur headless. Jetzt loest `showEvent()` den Erstaufbau aus. Zwei kleinere Sachen gleich mit: die Werkstatt rechnet das Netz selbst statt es aus dem Display zurueckzulesen (Kennzahlen haengen damit nicht an GL, und vermessen wird garantiert dasselbe Netz, das gezeichnet wird), und der Quadtree-Bauer prueft jetzt `ist_fuer_adaptives_mesh_geeignet()` wie die App — sonst haette die Werkstatt bei 384 px ein Quadtree gezeigt, das die App dort gar nicht baut.

**Geprueft, was ohne Bildschirm pruefbar ist** (`QT_QPA_PLATFORM=offscreen`): Fenster baut auf, alle vier Netzarten liefern ein Netz, Regler wirken, Fehlermessung erscheint in der Statuszeile. **NICHT geprueft und nicht pruefbar:** die OpenGL-Darstellung selbst — unter `offscreen` bekommt ein `QOpenGLWidget` keinen Kontext (siehe CLAUDE.md). Ob das Bild im Fenster steht und ob es dem Nutzer gefaellt, kann nur er sehen.

### Beim ersten echten Start gefunden: die Shaderpfade haengen am Arbeitsverzeichnis (App-Fehler, nicht Werkstatt-Fehler)

Der Nutzer startete `tools/mesh_werkstatt.py` — und der Prozess starb mit **`exit code -1073740791 (0xC0000409)`**, ohne Python-Traceback. Davor im Log: alle sechs Shader "does not exist", `Current working directory: ...\MapGenerator\tools`.

**Ursachenkette:**
1. `MapDisplay3D._load_shader_from_file()` suchte **ausschliesslich relativ zum Arbeitsverzeichnis** (`shaders/3d_display/terrain.vert` usw.). Aus `tools/` gestartet findet es nichts. Die App merkt davon nichts, weil sie immer aus dem Projektstamm laeuft — ein latenter Fehler, der nur noch nie ausgeloest wurde.
2. `_prepare_rendering()` gab danach trotzdem `True` zurueck. `_activate_fallback_rendering()` klingt so, als baue es einen Ersatz — es schaltet aber **nur den Polygonmodus auf Linien um** und laesst `shader_program = None`.
3. `glDrawElements` ohne aktives Programm ist im Core-Profile undefiniert. Der Treiber brach den Prozess hart ab.

**Behoben, beides:**
* `_load_shader_from_file()` sucht jetzt zusaetzlich vom Projektstamm aus, den es aus `__file__` ableitet (diese Datei liegt in `gui/widgets/`, also drei Ebenen hoch). Reihenfolge unveraendert: erst die bisherigen relativen Pfade, dann der absolute — was heute laeuft, laeuft gleich weiter.
* `_prepare_rendering()` gibt ohne Shaderprogramm `False` zurueck und meldet das einmal laut. Lieber ein leeres Fenster mit klarer Meldung als ein Absturz ohne Traceback.

### Zweiter Nutzerbefund: "Delatin crasht bzw. keine Rueckmeldung fuer min. 1 Minute"

Kein Absturz — Delatin rechnet in jeder Runde eine komplette Delaunay-Triangulierung neu (gemessen 126 s fuer 65954 Punkte bei 512 px) und blockiert dabei den GUI-Faden. Behoben in drei Schritten:

* **Zeitgrenze** in `waehle_punkte()`/`baue_delatin_mesh()` (`zeitgrenze_s`), in der Werkstatt auf 20 s gesetzt. Sie prueft **vorausschauend**: rein rueckblickend ueberzog sie um eine ganze Runde (27.4 s bei 20 s Vorgabe), weil die Runden immer teurer werden; mit der Schaetzung aus der letzten Rundendauer sind es 23.5 s bei 384 px und 15.6 s bei 512 px. Der Abbruchgrund steht in `stats["abbruch"]`.
* **Fortschrittsmeldung** (`fortschritt(runde, punkte, groesster_fehler_m)`) — die Werkstatt schreibt Runde, Punktzahl und verbleibenden Groessfehler in die Statuszeile, damit sichtbar ist, dass etwas passiert.
* **Bedienung gesperrt**, solange gerechnet wird — sonst stapelt jeder weitere Klick eine zweite Rechnung obendrauf.

**Dabei noch ein Fehler gefunden:** die Werkstatt gab der Anzeige den *Bauer* und liess ihn damit ein zweites Mal rechnen. Bei Quadtree und Remesh fiel das nicht auf (beide haben einen Zwischenspeicher), bei Delatin verdoppelte es die Rechenzeit — und der Fortschrittstext des zweiten Laufs ueberschrieb den schon fertigen Messbericht in der Statuszeile. Jetzt wird das fertige Netz weitergereicht statt der Vorschrift.

**`tests/smoke_test_shader_paths.py` um eine dritte Zusicherung erweitert:** die sechs Anzeige-Shader werden aus einem FREMDEN Arbeitsverzeichnis gesucht, und der Quelltext von `_load_shader_from_file`/`_prepare_rendering` wird darauf geprueft, dass die Reparatur noch drinsteht. Es ist dieselbe Lektion wie beim `SHADERS_ROOT`-Umzug 2026-07-30, an einer zweiten Stelle — und wieder war sie beim blossen Importieren unsichtbar. | 3 |

# 6a — Fuenf Pipeline-Ausgaben bleiben leer *(gemessen 2026-08-27)*

`smoke_test_pipeline_outputs.py` ist mit 5 Befunden rot, und zwar schon
seit dem 24.08. Am 27.08. nachgemessen, was genau leer ist und was nicht.
**Die Vorstufen sind gesund** - das ist der Punkt:

```
settlement.settlements / settlement_list      OK
settlement.suitability / combined_suit...     OK
settlement.city_boundary / city_mask          OK
settlement.pathfinding / roads                OK
settlement.roadsites / roadsite_list          OK
settlement.plot_nodes / plot_nodes            OK
settlement.plot_nodes / plot_edges            OK
settlement.plot_nodes / plot_map              OK

settlement.landmarks / landmark_list          NUR NULL
settlement.landmark_roads / landmark_roads    NUR NULL
settlement.pathfinding / sea_roads            NUR NULL
settlement.plot_nodes / plots                 NUR NULL
geology.intrusions / height_delta             NUR NULL
```

Damit sind die naheliegenden Erklaerungen weg: es liegt nicht daran, dass
keine Siedlungen entstehen, und nicht daran, dass die Wegsuche nicht
laeuft. Beides funktioniert.

Vier Beobachtungen, die die Suche eingrenzen:

1. **`plots` ist leer, waehrend `plot_nodes`, `plot_edges` und `plot_map`
   desselben Knotens gefuellt sind.** Wahrscheinlich ein Altschluessel aus
   dem alten PlotNodeSystem, den niemand mehr schreibt. Zuerst pruefen, ob
   ihn ueberhaupt noch jemand LIEST - wenn nicht, gehoert er aus
   `output_keys` heraus statt gefuellt.
2. **`height_delta` ist leer, waehrend `intrusion_delta` desselben Knotens
   gefuellt ist.** Derselbe Verdacht, andere Ecke.
3. **`sea_roads` braucht Wasser vom Seegrad 1 aufwaerts.** Die Testkarte
   ist 15 km breit - moeglicherweise hat sie schlicht keins. Das waere
   dasselbe Muster wie bei Vendee (6b): der Test misst seine eigene
   Testkarte. **Nicht belegt.**
4. **`landmark_roads` haengt an `landmark_list`.** Ein Befund, nicht zwei.

**Ein Versuch, der nichts bewiesen hat, und warum:** ich habe denselben
Durchlauf mit `SIZE = 512` statt 128 gefahren, um Punkt 3 zu pruefen. Die
Ausgabeform blieb `(128, 128)` - der `DataLODManager` bestimmt die
Aufloesung ueber das LOD, nicht ueber `map_size`. Der Vergleich hat also
zweimal dasselbe gemessen. Wer Punkt 3 pruefen will, muss am LOD oder an
`map_distance_km` drehen, nicht an `SIZE`.


# 6b — `_segmente_schliessen()` ist aufloesungsabhaengig

**Gemessen 2026-08-27, kein Handlungsbedarf fuer die echte Karte, aber
bekannt zu halten.**

`MIN_SEGMENT_M` (750 m) ist eine ABSOLUTE Laenge. Archetypen mit kurzer
Reichweite verlieren dadurch auf groben Karten Anteil an ihre Nachbarn:

| Vendee-Straende (Reichweite 0.18 km) | Saatanteil | Laengenanteil |
|---|---:|---:|
| 384 px (55 m/px) | -5.0 | **-20.1** |
| 768 px (28 m/px) | -4.9 | **-0.2** |

Bretagne-Klippen, sein Regionsnachbar, spiegelt das exakt: +11.4 bei
384 px, -5.8 bei 768 px.

**Warum es liegen bleibt:** die echte Karte laeuft mit 1024 px (21 m/px),
also noch feiner als die 768 px, bei denen der Effekt schon verschwunden
ist. Es betrifft ausschliesslich Tests und Vorschauen mit grober
Aufloesung.

**Was es trotzdem heisst:** jede Messung an der Archetypverteilung, die
mit weniger als etwa 700 px rechnet, misst zum Teil das Einschmelzen mit.
`smoke_test_archetyp_verteilung.py` trennt das seit dem 27.08. sauber in
zwei Zusicherungen (Saatanteil scharf, Laengenanteil weich).

Waere ein Ausgleich gewuenscht, gehoerte er nach `_segmente_schliessen()`
und muesste `MIN_SEGMENT_M` an `self.mpp` koppeln - nicht in die Quote,
die nachweislich stimmt.


# 7 — GPU und Paritaet

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **7.0** | **Schattenwurf: drei Einheitenfehler.** Gradient in Meter je PIXEL (Median-Hang 86.5° statt 11.1°), Strahl stieg Meter je Pixel (Sonne 83× zu tief), Shader ohne Einfallswinkel. Land war dunkler als Wasser. Behoben, beide Pfade bitgleich, **163× schneller**. | 1.5 |
| **[x]** | **7.5** | **Schattengitter skaliert mit der Karte** (ein Viertel der Kante statt fester 64 px). Exposition jetzt **aufloesungsunabhaengig** (0.685 → 0.671 statt 0.665 → 0.453), Korrelation GPU/CPU 0.99 statt 0.51. GPU bei gleichem Gitter **304× schneller**. | 0.5 |
| **[x]** | **7.1** | **`water.lake_detection / lake_map` auf der GPU konstant — keine Seen.** Ursache gefunden: `_classify_lake_basins_vectorized()` (managers/shader_manager.py) verglich die rohe Wassertiefen-Summe (Meter-Pixel) direkt gegen `lake_volume_threshold`, das seit 2026-07-27 ein echtes m³-Volumen ist — ohne Multiplikation mit der Zellflaeche (`meters_per_pixel**2`) wie auf der CPU-Seite. Bei 100 m/px fehlte ein Faktor 10000x, jedes Becken verfehlte die Schwelle. Fix: `meters_per_pixel` durchgereicht, `total_volume = depth_sum * meters_per_pixel**2` vor dem Vergleich. Verifiziert per `tests/smoke_test_water_lake_detection_gpu.py` (neu, 3 Seeds): GPU findet jetzt ueberall dort Seen, wo CPU welche findet (vorher immer 0). KEIN Anspruch auf Bit-Identitaet mit CPU — die Becken-ZUORDNUNG selbst bleibt ein bekannter Algorithmus-Unterschied (GPU: Jump-Flooding: CPU: echter Watershed-Transform seit 2026-07-27). | 1 |
| **[x]** | **7.2** | `water.evaporation / evaporation_map` auf der GPU komplett null. Beim Pipeline-Audit 2026-08-11 gezielt gegengeprueft: **tritt nicht mehr auf** (vermutlich Nebeneffekt einer der Wasser-Aenderungen dieser Session, nicht gezielt gefixt). | 1 |
| **[x]** | **7.3** | `biome.climate_classification` auf der GPU konstant. Beim Pipeline-Audit 2026-08-11 gezielt gegengeprueft: **tritt nicht mehr auf** (selbe Einschraenkung wie 7.2 — kein gezielter Fix, nur bestaetigt verschwunden). | 1 |
| [ ] | 7.4 | Vier Schwellwertkipper aus float32 — als **Toleranz** fuehren, nicht beheben | 0.5 |
| **[~]** | **7.6** | **GPU laeuft im Lauf der Generierung aus dem VRAM** (Nutzerlog 2026-08-11, 1024px): ab `water.jumpFloodLakes` scheiterte praktisch jeder weitere Shader-Aufruf mit `GL_OUT_OF_MEMORY` (err 1285). Untersucht: der Compute-Shader-Pfad selbst (`managers/shader_manager.py`) ist SAUBER - eine zentrale Allokations-Kapsel (`_gl_allocation_scope()`, seit einem frueheren Fix) gibt jede Textur/jedes SSBO auch im Fehlerfall zuverlaessig frei, alle `glGenTextures`/`glGenBuffers`-Aufrufe im Datei laufen bereits ueber diese Kapsel. GEFUNDEN wurde stattdessen ein echtes, aber KLEINES Leck in `gui/widgets/map_display_3d.py`: die Wasserplatten-VAO/VBO (`_wasser_vao`/`_wasser_vbo`) wurde bei jedem Mesh-Neubau nur auf `None` gesetzt statt vorher geloescht - orphaned bei jeder Regeneration mit Wasser. Behoben (`glDeleteVertexArrays`/`glDeleteBuffers` vor dem Neuaufbau). **Reicht das als Erklaerung fuer den grossen OOM?** Unsicher - die Wasserplatte selbst ist winzig (6 Vertices). Plausiblerer Haupttreiber: 6.14 (alle zehn Tabs redraw'ten vorher bei jeder Generator-Fertigstellung) - falls mehrere Tabs im 3D-Modus stehen, hielt/erneuerte das mehrfach volle Mesh-Texturen gleichzeitig. Mit 6.14 behoben trat der OOM in einem Folgelauf nicht erneut auf - kein Beweis, dass er weg ist, aber ein starkes Indiz, dass beide Befunde (6.14 und dieser) denselben Ursprung hatten. | 1 |
| [ ] | 7.7 | **`water.flow_network`-CPU-Fallback katastrophal langsam bei 1024px**: 492.9s (ueber 8 Minuten) in dem Lauf, in dem 7.6 zuschlug - derselbe Knoten lief bei 512px auf GPU in 2.7s, auf GPU bei 1024px (Folgelauf, kein OOM) in 3.2s. Der CPU-Pfad ist offenbar nie fuer diese Groesse profiliert worden. Wird durch einen Fix von 7.6 vermutlich seltener relevant, sollte aber unabhaengig davon nicht so schlecht sein - ein Fallback darf langsamer sein, nicht um zwei Groessenordnungen. | 1.5 |

| [ ] | 7.8 | **`settlement.landmarks / landmark_list` und `landmark_roads`: GPU liefert Daten, CPU nicht** *(neu aufgenommen 2026-08-12, `smoke_test_pipeline_outputs.py` bei 128 px)*. Der Test meldet beide Outputs als `PFADE VERSCHIEDEN` — GPU `OK`, CPU `NUR NULL`. **Stand bisher in keinem Dokument.** Nicht untersucht; die Landmark-Platzierung haengt an `civ_map` und am Wegenetz, von denen mindestens eines pfadabhaengig sein duerfte. Wahrscheinlich harmlos (128 px ist klein, und die Platzierung hat Mindestabstands-Regeln, die bei wenigen Pixeln leicht alles ausschliessen), aber ungeprueft — bei 512 px gegenmessen, bevor Aufwand hineingesteckt wird. | 1 |

| [ ] | 7.9 | **Geology hat als einziger Generator keine GPU-Anbindung** *(aus `docs/backlog.md` Punkt 41, 2026-07-08, hier neu aufgenommen 2026-08-12)*. Lag ausserhalb des Water/Weather/Biome-Fokus der damaligen GPU-Sitzung. **Heute weniger dringend als damals:** die beiden fruehen Geologie-Engpaesse sind auf der CPU behoben (6.8, 18- bis 29-fach schneller), im 1024px-Lauf braucht die gesamte Geologiekette nur noch rund 6 s. Also eher Vollstaendigkeit als Not. | 2 |
| [ ] | 7.10 | **`jumpFloodLakes.comp` ordnet Becken nicht ueber echte Erreichbarkeit zu** *(aus `docs/backlog.md` Punkt 42)*. Die GPU nutzt "current_height >= seed_height" plus Luftlinien-Distanz; die CPU-Seite laeuft seit 2026-07-27 ueber einen echten Priority-Flood. **Bewusst zurueckgestellt, mit Begruendung:** Dijkstra ist in der Verarbeitungsreihenfolge inhaerent sequenziell, echte parallele Watershed-Transforms sind ein eigenes Forschungsthema. Der Spill-Point-Filter daemmt die Auswirkung ein, ersetzt aber keine korrekte Zuordnung. Bekannter, benannter Algorithmusunterschied — kein Fehler, solange er dokumentiert bleibt (siehe auch 7.1). | 3 |
| [ ] | 7.11 | **Die uebrigen GPU-Dispatchfunktionen sind nie auf die Fehlerklasse der Seenerkennung geprueft worden** *(aus `docs/backlog.md` Punkt 45)*. Am 2026-07-09 und erneut bei 7.1 wurde in `shader_manager.py` zweimal derselbe Fehlertyp gefunden: **ein Bezugswert, der auf CPU- und GPU-Seite verschieden gemeint ist** (Referenzhoehe; spaeter Volumen gegen Summe ohne Zellflaeche). Geprueft wurde jeweils nur die Seenerkennung. Erosion/Sedimenttransport, Wetter und Biome sind auf dieselbe Klasse **nicht** untersucht. Dass 7.2 und 7.3 spaeter "von selbst" verschwanden, ist ein Hinweis darauf, dass dort aehnliche Verstimmungen sassen — verstanden ist keine davon. **Lohnendster Test dieser Liste**, weil er eine ganze Fehlerklasse auf einmal erwischt statt eines Einzelfalls. | 2 |
| [ ] | 7.12 | **`terrain.redistribution` ist ein 33-Sekunden-Monolith-Knoten — betrifft sowohl moegliche Parallelisierung als auch den Ladebalken** *(Nutzerbefund 2026-08-13 aus dem Pipeline-Log: `terrain.redistribution \| GPU \| Dauer 32.938s \| ... heightmap=... ridge_map=... river_mask=... river_order=... river_generation=... region_map=... klima_map=...` — sieben Ausgaben auf einmal nach 33s. Frage des Nutzers: "Kann man das in kleinere Prozesse trennen, falls das Sinn macht? Fruehere Daten herausgeben, damit andere Prozesse schon starten koennen? Und der Ladebalken steht dann ganz lange auf 3%, die muessen ja auf die Prozesslaenge genormt sein." ENTWURF/BEFUND, NICHT umgesetzt.)*<br><br>**Codebefund zur ersten Frage (Splitten):** `_calc_redistribution()`/`_weltkarte_heightmap()` (`core/terrain_generator.py`) ist EIN Calculator-Knoten, der intern drei serielle Stufen durchlaeuft — `weltfeld()` (Rauschen/Regionen/Kuesten/Meerestiefe, liefert u.a. schon `region_map`/`klima_map`/`wind_ziel_map`), dann den Erosionsfilter, dann das Fluessnetz (drei Rechenstufen + Taeler eingraben) — und schreibt alle sieben Ausgaben erst GANZ AM ENDE in einem einzigen `set_calculator_output()`. Kein nachgelagerter Knoten kann etwas davon frueher lesen, selbst wenn er nur einen Teil braucht — der Calculator-Graph (`managers/calculator_graph.py`) fuehrt Abhaengigkeiten ohnehin nur auf KNOTEN-Ebene, nicht auf Feld-Ebene. **Konkret ungenutztes Potenzial:** `region_map`/`klima_map`/`wind_ziel_map` liegen bereits VOR dem Erosionsfilter und dem Fluessnetz fertig vor (im `felder`-dict direkt nach der `weltfeld()`-Rueckgabe), werden aber trotzdem erst zusammen mit dem fertigen Fluessnetz freigegeben. `weather.temperature`/`weather.humidity` haengen im Graph an `terrain.redistribution`, lesen aber tatsaechlich nur `klima_map`/`region_map` — sie koennten heute schon parallel zur (typischerweise langsamsten) Fluessnetz-Stufe starten, wenn der Knoten dafuer aufgeteilt waere. **Kein Gewinn dagegen fuer `terrain.slope`/`terrain.shadow`:** die `heightmap` selbst ist erst NACH allen drei Stufen final (die Fluesse graben sich in die bereits erodierte Form ein) — ein Splitten wuerde diese beiden Knoten nicht frueher starten lassen, sie muessten ohnehin auf die letzte Stufe warten. Der moegliche Gewinn beschraenkt sich also auf den Wetter-Zweig, nicht auf die ganze Pipeline — das muss gegen den Umbauaufwand (drei neue Knotendefinitionen in `calculator_graph.py`, `_weltkarte_heightmap()` in drei Teilfunktionen mit eigenen Zwischenspeichern zerlegen) abgewogen werden, bevor daran gearbeitet wird.<br><br>**Codebefund zur zweiten Frage (Ladebalken):** `_refresh_footer_progress()` (`gui/map_editor.py`) gewichtet JEDEN Calculator-Knoten gleich — `total = sum(dispatcher.target_lod.values())` ueber ALLE Knoten (aktuell rund 34-39), `done` die Summe der erreichten LOD-Runden je Knoten; ein Knoten zaehlt unabhaengig davon, ob er 0.01s oder 33s braucht. `terrain.redistribution` ist EIN Knoten unter ~35-39, macht rechnerisch **rund 3%** aus — exakt der Wert, bei dem der Balken haengt, weil waehrend dieser 33s kein einziger LOD-Fortschritt gemeldet wird, und danach in einem einzigen Sprung weiterspringt. **Der Balken ist nach ANZAHL Knoten normiert, nicht nach tatsaechlicher Rechenzeit je Knoten** — die Beobachtung des Nutzers trifft die Ursache genau. Ein zeitbasierter Fix (je Knoten eine gemessene/geschaetzte Referenzdauer statt der pauschalen Gewichtung 1) ist UNABHAENGIG vom Splitten oben und liesse sich auch ohne Graph-Umbau umsetzen — vermutlich der schnellere Hebel der beiden.<br><br>**Offen fuer eine spaetere Entscheidung:** (a) ob der Splitten-Umbau angesichts des auf Wetter beschraenkten Nutzens den Aufwand lohnt, (b) ob zuerst nur der Ladebalken zeitbasiert normiert wird (kleinerer, unabhaengiger Schritt) und das Splitten separat bewertet wird. | 3 |

| [ ] | 12.6 | **Fluesse muessen aus Binnenseen wieder herausfliessen (Massenerhaltung)** *(Nutzer-Vorgabe 2026-08-13: "fluesse koennen in binnenseen abfliessen, aber per Konti-Gleichung muss die Summe aller Fluesse am einfachsten Weg zum Meer wieder weiterfliessen, Verdampfung vernachlaessigt. wie kann man das umsetzen?" - ENTWURF, NICHT umgesetzt.)*

**Ist-Zustand:** in `terrain_weltfluesse.py::baue_stufe()` ist JEDER Knoten mit `H<=0` ein gueltiger Dijkstra-Ausgang - ein Binnensee wirkt heute wie ein zweites Meer, Fluesse enden dort und sind fertig. Es gibt keinen Mechanismus, der das Wasser am See wieder herausfuehrt.

**Das ist ein Standardproblem der Hydrologie** ("Depression Filling"/"Fill-and-Spill"): jede abflusslose Senke wird gedanklich mit Wasser gefuellt, bis sie an ihrem NIEDRIGSTEN Randpunkt (dem "Pour Point"/Ueberlaufpunkt) ueberzulaufen beginnt - ab dann fliesst JEDER weitere Tropfen dort ab, in Richtung des naechsttieferen Gebiets. **Das Projekt hat diesen Algorithmus bereits, nur an anderer Stelle:** `core/water_generator.py::_apply_priority_flood_watershed()` nutzt seit 2026-07-27 `skimage.segmentation.watershed` fuer genau diese Klasse Problem (Seebecken-Erkennung/-Fuellung). Naheliegend, ihn wiederzuverwenden statt eine zweite Fuell-Logik in `terrain_weltfluesse.py` zu bauen.

**Konkreter Umsetzungsvorschlag, drei Schritte:**
1. Fuer jeden Binnensee (siehe `_hauptmeer_maske()`, 3.12/3.13 - Binnenseen sind bereits als eigene Zusammenhangskomponenten identifizierbar) den POUR POINT bestimmen: den niedrigsten Punkt auf dem Rand der Seeflaeche, ueber den Wasser ins naechsttiefere Nachbargebiet uebertreten wuerde. Direkt aus `H` ablesbar (Rand der See-Maske, Minimum der Hoehe dort).
2. In `baue_stufe()`: Seeflaechen-Knoten NICHT mehr als eigene Dijkstra-Ausgaenge zulassen (aus der `unter_wasser`-Liste ausschliessen). Stattdessen den Pour-Point-Pixel als GEZWUNGENEN Durchgangsknoten behandeln - jeder Lauf, der im See endet, wird auf den Pour Point umgehaengt, der seinerseits normal weiter zum naechsten echten Ausgang (Hauptmeer oder der naechste See) fliesst.
3. **Massenerhaltung im eigentlichen Sinn ("Summe aller Fluesse") braucht zusaetzlich Akkumulation:** die `flaeche`/`gebiet`-Berechnung (bereits vorhanden, akkumuliert Knotenzahl den Baum hoch) muesste am Pour-Point-Knoten die Summe ALLER in den See eingehenden Zufluesse aufnehmen, nicht nur die eines einzelnen Laufs - technisch eine Vorstufe, die alle auf den See gerouteten Knoten VOR der normalen Baumkonstruktion am Pour Point zusammenfasst.

**Warum nicht in dieser Sitzung umgesetzt:** beruehrt zwei bisher getrennte Systeme (`terrain_weltfluesse.py`/`water_generator.py`), und die Pour-Point-Bestimmung braucht eigene Sorgfalt bei mehreren ineinander abfliessenden Seen (ein See kann in einen tieferen See abfliessen, nicht nur direkt ins Meer) - das verdient eine eigene, fokussierte Sitzung mit eigener Pruefung, nicht einen Nebensatz in einer bereits sehr grossen. | 3 |

# 8 — LOD entfernen

| | # | Sache | Risiko | Aufwand |
|---|---|---|---|---|
| [ ] | 8.1 | Kosmetik: Logzeilen und Fortschrittstexte (~373 Stellen) | keins | 1 |
| [ ] | 8.2 | Rundensteuerung durch topologische Reihenfolge ersetzen (~190) | **hoch** | 3 |
| [ ] | 8.3 | `lod_level` aus Signaturen (~1036 Stellen) | Menge | 3 |

# 9 — Portierung P4 bis P6

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **9.3** | **Kuestentemperatursprung 5.6 K** — Ursache war seit Wochen offen. Er entstand, weil das Meer als Landflaeche mit Hoehe 0 behandelt wurde; mit dem festgelegten Seeklima ist er weg. | — |
| [~] | 9.1 | Biome/Wasser/Wetter eichen — **Wetter erledigt**, Biome-Wurzelursache gefunden+behoben, Wasser offen. **Erstmessung 2026-08-11** (512px, Seed 20260804, gegen `docs/spezifikation/13_KLIMA_UND_BIOME.md` Abschnitt 8 "erwartete Grundbiome"): 6/9 Regionen trafen ihre Erwartung gut, drei Abweichungen (Macchia 47% Bruchwald statt Steineichenwald/Macchia, Fjordlands Bergwald fast weg, "Feuchtwiese" in trockenen Regionen). **Ursache gefunden: `classify_base_biomes()`/`apply_super_biome_overrides()` bekamen das JAHRESMITTEL aus `weather.temperature`s `temp_map`, obwohl `biome_definitions` UND die Alpin-/Firn-Schwellen (BAUMGRENZE_JULI_C/FIRN_JULI_C, siehe 2.7) explizit gegen JULI kalibriert sind** - am Beispiel Macchia lag der Median-Landpixel damit bei 16.4°C statt der Juli-Referenz 25.6°C, wodurch Steineichenwald (21-28°C-Bereich) fast ueberall ausserhalb seines Bereichs fiel. Fix: neuer `temp_map_juli`-Fetcher (`_get_prepared_biome_inputs()`), liest die ohnehin schon berechnete `temp_map_monthly[3]` (Periode 3 = Juli bei TICKS_JE_JAHR=6) statt averaging - kein neuer Rechenweg. **Nachgemessen:** Macchia jetzt Macchia 38%/Steineichenwald 31%/Bruchwald 15% (vorher 0%/0%/47%) - trifft die Erwartung. Thalassia/Samarcia ebenfalls verbessert ("Feuchtwiese" in Samarcia von 8% auf 2%). Alpine-Realisierung (2.7) mit der jetzt korrekten Juli-Temperatur neu gemessen: **0.89% statt der vorher (mit dem falschen Jahresmittel) gemessenen 8-9%** - deutlich plausibler fuer eine echte Alpinzone. Regressionsfrei (`smoke_test_biome_supersampling.py`, `smoke_test_display_2d.py`, `smoke_test_pipeline_outputs.py` unveraendert). Fjordlands Bergwald-Anteil NICHT gezielt untersucht, koennte an derselben Wurzelursache haengen oder eigenstaendig sein - noch offen. Wassereichung (Seen-/Flussdichte je Region) weiterhin komplett ungeprueft. | 2 |
| [ ] | 9.2 | Reglerspannen gegen Modellgrenzen: neun Niederschlagskappungen | 1 |

Offene Einzelbefunde: Landluftrate (+0.13 K/100 m) — **koennte mit 1.1 erledigt
sein, nachmessen**; Bodenfeuchte-Median 0.002.

| [ ] | 9.3 | **Wasser und Water-Depth sind halbfertig — erst Erosion reparieren, dann Wasser neu angehen** *(Nutzerbefund aus der Sichtpruefung 2026-08-13)*. Wortlaut: *"Wasser und Water depth ist ein thema fuer spaeter. da fehlen noch einige sachen. also wasser sammelt sich in den meeresmulden, kein blau ueber die gesamte meereskarte, was ehrlich gesagt erstmal ok ist. wir sollten erstmal die erosion nochmal repariert haben (was auch ein thema fuer spaeter ist) und dann water neu angehen und die seenbildung und so weiter. also halbfertig."* **Die vom Nutzer vorgegebene Reihenfolge ist der eigentliche Inhalt dieses Punkts:** erst Erosion (Abschnitt 4), dann Wasser samt Seenbildung - nicht umgekehrt, weil die Wasserwege auf der erodierten Form entstehen. Beruehrt zugleich 9.1 (Wassereichung, bisher als "letzte ungeeichte Achse" gefuehrt) und 12.6 (Abfluss aus Binnenseen) - beide sollten in derselben Sitzung mitgedacht werden statt einzeln. | 3 |

# 10 — Altlasten

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **10.1** | **`geology.intrusions / height_delta` immer null - GEPRUEFT 2026-08-16, KEIN BUG.** Steht so im Modul-Docstring von `core/geology_generator.py` (Zeile 19-25): explizite Nutzer-Vorgabe *"Stoerungen greifen nicht in das Terrain ein"*, spaeter auf Intrusionen erweitert ("die fruehere gekappte Dom-Hebung erzeugte eine Hoehenaenderung, die nicht gewollt war"). `_calc_intrusions()` setzt `height_delta = np.zeros_like(intrusion_delta)` bewusst - Intrusionen wirken nur auf `layer_id_map` (Ausbiss), nicht auf die Hoehe. Nichts zu beheben, dieser Punkt war veraltet. | — |
| **[x]** | **10.2** | `regionskern` / `pruefe_regionen` tot und falsch messend — geloescht 2026-08-16 (`core/terrain_weltkarte.py`). Nur noch in einem Kommentar in `smoke_test_regionen_welt.py` erwaehnt (erklaert, warum der Test sie NICHT mehr benutzt - der Test rechnet Hang/Wasseranteil seit der Voronoi-Umstellung selbst). `fingerabdruck()` war nach dem Loeschen der beiden anderen ebenfalls ohne jeden Aufrufer - mitgeloescht. | 0.5 |
| **[x]** | **10.3** | `tools/regionen_welt.py` und `tools/regionen_fluesse.py` nur Weiterleitungen — beide geloescht 2026-08-16. Einziger echter Aufrufer war `tests/smoke_test_regionen_welt.py` (`import regionen_welt as rw`), auf `import core.terrain_weltkarte as rw` umgestellt (identische Attribute, die Weiterleitung war ein reiner `import *`). `regionen_fluesse.py` hatte ausserhalb seiner selbst und der Doku ueberhaupt keinen Aufrufer mehr. Nebenbefund: `core/terrain_weltkarte.py`s eigener Datei-Kopf-Kommentar sagte noch "Path: tools/regionen_welt.py" - stammte offenbar vom Verschieben des Rechenkerns dorthin (2026-08-05) und wurde nie nachgezogen, korrigiert. | 0.5 |
| **[x]** | **10.4** | **`octaves` bis 8 erlaubt, wirksam max 4 (Nyquist) - GEPRUEFT 2026-08-16, KEIN BUG.** `BaseTerrainGenerator._max_safe_octaves()` (`core/terrain_generator.py`) clampt bereits korrekt auf die tatsaechliche Nyquist-Grenze (0.5 Zyklen/Pixel), einheitlich fuer GPU-/CPU-/Fallback-Pfad, UND meldet das Clamping laut (`logger.debug`, wenn `effective_octaves < octaves`) - genau die in CLAUDE.md geforderte Regel fuer stille Ruecklaeufe. Den Slider selbst auf 4 zu deckeln waere sogar FALSCH: die sichere Obergrenze haengt von `frequency`/`lacunarity` ab und liegt bei anderen Kombinationen nachweislich hoeher (Kommentar bei FREQUENCY nennt 5 bei verdoppelter Frequenz) - ein fester Deckel wuerde gueltige Kombinationen unnoetig einschraenken. Die Reglerbeschreibung erklaert das Verhalten bereits im Tooltip. Nichts zu aendern. | — |
| **[x]** | **10.5** | `_is_in_shadow_cpu` / `_calculate_slope_shading_cpu` nach dem Umbau tot — geloescht 2026-08-16 (`core/terrain_generator.py`, Klasse `ShadowCalculator`; nicht `BaseTerrainGenerator`, wie der Name vermuten liesse). Beide seit dem 2026-08-07-Umbau auf den vektorisierten Pfad (`_verschattung_cpu`/`_einfallswinkel`) unbenutzt, per Grep repoweit bestaetigt (nur noch Kommentar-Erwaehnungen). `_interpolate_height_cpu` blieb stehen - wird von `_resize_2d()` weiterhin gebraucht. Zwei jetzt veraltete Kommentare (Zeile ~854, ~1016) auf Vergangenheitsform/Verweis auf diesen Punkt umgestellt, die Fehlergeschichte im SONNENEXPOSITION-Block bewusst NICHT geloescht (erklaert dreimal denselben Pixel/Meter-Fehler, weiterhin lehrreich). | 0.5 |
| **[x]** | **10.6** | Erosionskette `EROSION_AKTIV` — *Nutzer 2026-08-06: "vorerst nein"*, dann **doch eingeschaltet am 27.08.2026** (`gui/config/value_default.py:1088`, `True`). Die Zeile stand bis zum 16.09.2026 offen und behauptete damit weiter, die Kette sei aus. Sie ist an: drei Erosionstests sind rot, ihre Ursache ist ungeklaert (`docs/TESTBERICHT.md`, Abschnitt 3 — dort korrigiert von "vier" auf "drei", Ticket #63), und der Paritaetsbruch Faktor 385 zwischen GPU und CPU steht damit mitten im laufenden Betrieb statt vor einer Reaktivierung. | — |

# 11 — Oberflaeche *(aus dem TESTBERICHT, inzwischen erledigt)*

| | # | Sache |
|---|---|---|
| **[x]** | **A1** | 36 wirkungslose Regler gesperrt, aus den Schaltern abgeleitet |
| **[x]** | **A2** | Weltflussnetz bedienbar — 7 Regler wirken, 2 neue dazu, Vorgabewerte liefern bitgleich dieselbe Welt |
| **[x]** | **A3** | Die "toten" Regler untersucht — **drei der vier Befunde waren falsch** |
| **[x]** | **D1** | Sollhaenge zweimal nachgezogen, Ordnungspruefung als Waechter |


---

### 13.0 — Grundsatzentscheidung 2026-08-13: EINE Terrain3D-Welt, Detail per Streaming (loest die Frage aus 5.15)

**Nutzer-Vorschlag im Wortlaut:** *"ehrlich gesagt koennte ja die Terrain3D karte immer die gleiche sein, also wenn das weniger probleme macht, dann haben wir eine einfachere LOD in den aussenbezirken und FoW und haben dort nur sehr sporadisch ein paar baeume die durch den FoW stechen, aber nicht viel mehr. sobald die karte gewechselt wird werden alle zusaetzlichen elemente eingeladen."*

**Das ist der bessere Ansatz, und die Zahlen stuetzen ihn deutlich.** Er loest zugleich die in 5.15 offen gelassene Frage "neun getrennte Spielkarten oder eine zusammenhaengende Welt" — die Antwort ist: **beides, auf verschiedenen Ebenen.** Die Daten sind eine Welt, die Spielabschnitte sind neun Zonen.

**Warum das billiger ist — der Engpass war nie das Terrain:**

| | Menge |
|---|---:|
| Terrain-Daten ganze Welt bei 10.4 m/px (13.1) | **48 MB**, EINMALIG |
| dito bei 4 m/px | 325 MB |
| Baum-Instanzen aktive Zone (4x4 km, ~200/ha) | 0.32 Mio. |
| Baum-Instanzen gesamtes Land (132 km²) | 2.64 Mio. |

Das Terrain ist bei diesen Groessen unkritisch und liegt ohnehin nur einmal im Speicher, egal welche Zone aktiv ist. **Was mit der Flaeche skaliert, sind die Streuobjekte — Faktor 8.2 zwischen aktiver Zone und ganzer Welt.** Genau dort greift der Vorschlag: Vollbestand nur in der aktiven Zone, ausserhalb sporadisch (bei 3 % Restdichte: 0.32 + 0.07 = **0.39 Mio. statt 2.64 Mio., Faktor 6.8 gespart**) — und zwar ohne dass irgendwo eine Weltgrenze sichtbar wird.

**Der Export wird dadurch EINFACHER, nicht komplizierter:**

* vorher gedacht: neun Kartensaetze, je Hoehe+Control+Color, mit ueberlappenden Raendern, neun Koordinatensystemen und Naht-Pflege an acht inneren Grenzen
* jetzt: **ein** Satz Hoehe/Control/Color fuer die ganze Welt (nahtlos, ein Koordinatensystem) **plus neun Polygonzuege** als Zonenmasken (wenige KB, im selben Format wie die Vektordaten aus 13.5)

**Welche Rolle die Spielkarten-Zerlegung (5.15) dabei behaelt:** sie wird nicht ueberfluessig, sondern wechselt die Rolle — von einer **Terrain-Schnittgrenze** zu einer **Streaming- und FoW-Zone**. Was dadurch weniger wichtig wird: die Forderung "kompakt, auf einem Quadrat darstellbar" galt der Darstellung als eigene Karte; fuer eine FoW-Zone ist sie nachrangig (sie bleibt fuer die 2D-Editoransicht sinnvoll). Was wichtig BLEIBT: die gleiche Landmasse je Zone, denn die bestimmt, wie viel Spielinhalt ein Abschnitt traegt.

**Ein Gestaltungspunkt, der dabei zu loesen ist:** ein harter Sprung von Vollbestand auf 3 % waere als Ring sichtbar. Die Dichte sollte ueber eine Strecke abfallen — **das ist exakt dasselbe Maskenmuster mit weichem Rand, das der Nutzer schon fuer Kuesten und Fluesse beschrieben hat** ("Strahlkraft", siehe 3.11/6.20). Die Zonenmaske waere also keine harte 0/1-Grenze, sondern ein Graustufenfeld — und laesst sich mit derselben Rechnung erzeugen.

**Was daraus fuer 13.1 folgt:** die dort beschlossenen 2048 px / 10.4 m/px fuer die ganze Welt passen zu diesem Modell unveraendert. Die frueher erwogene Alternative "je Region 4096 px" ist damit endgueltig vom Tisch — sie war die teure Loesung fuer ein Problem, das dieser Ansatz gar nicht erst hat.

**Nicht belegt und ehrlich benannt:** ob die Bildrate traegt, ist damit NICHT bewiesen. Die Zahlen sagen, dass der Ansatz die richtige Groessenordnung trifft und der bisher groesste Posten um Faktor ~7 faellt; sie sagen nichts ueber Godots tatsaechliches Verhalten mit 0.4 Mio. Instanzen, MultiMesh-Batching und Schattenwurf. Der Versuch aus 5.15 bleibt noetig: flache 21-km-Testheightmap in Terrain3D, durchfliegen, Bildrate und Speicher ablesen — jetzt zusaetzlich mit einer Handvoll gestreuter MultiMesh-Baeume, um den eigentlichen Engpass mitzumessen statt nur das Terrain.

### 13.7 — Wie Strassen und Staedte dargestellt werden: Decal, Rasterkanal oder Mesh? *(Nutzerfrage 2026-08-13, beantwortet)*

**Vorweg eine Korrektur an mir selbst.** Ich hatte zuerst mit REALISTISCHEN Strassenbreiten gerechnet (5 m) und daraus geschlossen, ein Rasterkanal sei unmoeglich - bei 10.4 m/px waere eine solche Strasse 0.48 Pixel breit. Der Nutzer hat die Annahme berichtigt: *"natuerlich kann die etwas groesser sein als realistisch, ist ja auch bei den bergen, das 1000m berge hohe 4000er symbolisieren. ist nunmal so wenn man eine karte baut."* Das ist richtig und aendert das Ergebnis vollstaendig - meine Rechnung beantwortete eine Frage, die niemand gestellt hatte.

**Neu gerechnet mit der Breite, die heute tatsaechlich dargestellt wird** (der 3D-Skin zeichnet Wege rund 3 Texturpixel breit):

| Textur | m/px | wirkende Wegbreite |
|---|---:|---:|
| 384 px | 55.5 | 166 m |
| 512 px | 41.6 | 125 m |
| 1024 px | 20.8 | 62 m |

Und ob DIESE Breite in eine Terrain-Control-Textur passt:

| Export | m/px | 30 m | 60 m | 100 m |
|---|---:|---:|---:|---:|
| 2048 px | 10.4 | 2.9 px | 5.8 px | 9.6 px |
| 4096 px | 5.2 | 5.8 px | 11.5 px | 19.2 px |

**Damit ist der Rasterkanal problemlos machbar** - ab rund 31 m Breite bei 2048 px Export.

### Empfehlung, getrennt nach Zweck

**Im EDITOR (dieses Werkzeug) bleibt alles wie es ist, und das ist kein Kompromiss.** Der Zweck hier ist zu BEURTEILEN, ob die Generierung stimmt - nicht, schoen auszusehen. Ein Strassen-Overlay als Textur zeigt genau das Richtige: wo laeuft der Weg, wie liegt er im Gelaende. Ein Stadtmodell waere hier sogar schlechter, weil es das Gelaende verdeckt, auf das es ankommt. Fuer die Orte kommen Billboards mit Symbolen (6.27), keine Gebaeude.

**Im EDITOR ist das inzwischen ueberholt - siehe 6.28:** die Wege bekommen dort echte Bandgeometrie, weil der Nutzer die Textur-Optik ausdruecklich verbessert haben wollte. Der Absatz oben ueber den Editor gilt insofern nicht mehr.

**In GODOT - Strassen: Control-Textur zuerst, Mesh spaeter falls noetig.**
* **Rasterkanal in der Control-Textur** (13.3 sieht diese Ebene ohnehin vor): kostet KEINEN zusaetzlichen Draw-Call, folgt dem Gelaende exakt, kann kein Z-Fighting bekommen und kein Schweben ueber Senken. Bei 3-10 Pixeln Breite sieht das gut aus. **Das ist der klare Startpunkt** - der Aufwand besteht darin, die vorhandenen Wegpfade beim Export in den Kanal zu rastern, mehr nicht.
* **Mesh entlang der Spline** (aus den Vektordaten von 13.5) waere erst noetig, wenn es Bruecken, Hohlwege, Strassenbelag-Kanten oder Boeschungen geben soll - also Dinge, die AUS der Terrainoberflaeche herausragen. Groessenordnung dafuer gemessen: ein 60-km-Wegnetz mit 10-m-Segmenten sind rund **12000 Dreiecke** gegen die ~450000 des Terrain-Meshes, also vernachlaessigbar. Es scheitert nicht an der Leistung, sondern lohnt den Mehraufwand erst mit diesen Extras.
* **Klassische projizierte Decals** (Deferred Decals) sind hier die schlechteste der drei Optionen: sie kosten Fuellrate, brauchen eigene Projektionsvolumen je Wegstueck und loesen ein Problem, das bei Terrain gar nicht besteht - der Rasterkanal projiziert bereits perfekt.

**In GODOT - Staedte: KEIN Stadtmodell, sondern prozedural aus den vorhandenen Daten.**
Ein heruntergeladenes "Stadt"-Modell passt prinzipiell nie: die erzeugten Orte sind verschieden gross (15-50 Haeuser laut `house_count`), verschieden geformt und liegen in verschiedenem Gelaende. **Das Projekt hat die noetigen Daten bereits** - `plot_nodes` (mit `node_type`: standard/wilderness/city_border), `plot_edges` (mit `classification` und `traffic`), `plot_cores` und `city_mask`. Daraus lassen sich in Godot einzelne Gebaeude an den Parzellenkernen instanziieren.

Was dafuer **heruntergeladen oder gebaut** werden muss, sind **Bausteine, keine Staedte**: pro Kultur eine Handvoll Modelle (Wohnhaus, Scheune, Werkstatt, Sakralbau, Turm/Tor) plus ein paar Varianten. Frei nutzbare CC0-Saetze (Kenney, Quaternius) reichen fuer den ersten Durchgang voellig; sie spaeter durch eigene zu ersetzen, aendert am Verfahren nichts. Die Zuordnung Gebaeudetyp -> Parzelle folgt derselben Katalog-Logik, die es fuer Landmarks und Roadsites schon gibt (45 Arten je Kultur).

**Reihenfolge:** Strassen als Control-Kanal ist der billigste sichtbare Gewinn und sollte zuerst kommen. Die Stadt-Instanziierung haengt an 13.4 (Streuung ueber Dichtekarten) - dasselbe Verfahren, nur mit Gebaeuden statt Baeumen, und lohnt daher, zusammen damit gemacht zu werden. | 3 |

# 13 — Godot-Export *(Entscheidungen des Nutzers 2026-08-13)*

Vier Grundsatzfragen sind beantwortet. Umgesetzt ist noch nichts.

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **13.1** | **Weltgroesse fuer den Export: 2048 px fuer die GANZE Karte, Feindetail per Rauschen in Godot** *(Nutzerentscheidung)*. Ergibt **10.4 m/px** bei 21.3 km - halb so grob wie heute (20.8). Die Alternative "je Region 4096 px = 1 m/px" wurde bewusst verworfen (30-60 Min Backzeit je Region). **Bekannte Folge, ehrlich benannt:** eine Spielfigur steht auf 10-m-Dreiecken, das Detail muss also weitgehend aus dem Engine-Rauschen kommen - echte Gelaendeformen (Rinnen, Felsabbrueche, Geologie-Aufschluesse) entstehen daraus nicht, nur Rauheit. | 1 | **ERLEDIGT 2026-08-24:** `EXPORT_KANTENLAENGE_PX = 2048` in `gui/utils/map_export.py`, jeder Layer wird darauf gebracht - Kategorien/Farbbilder per naechstem Nachbarn, skalare bilinear. Manifest fuehrt Massstab und je Layer einen `groesse`-Vermerk. Geprueft von `tests/smoke_test_export_2048.py`.
| **[x]** | **13.2** | **Daempfungsmaske fuers Engine-Rauschen** *(Folge aus 13.1, nicht vom Nutzer angefragt sondern hier aufgeworfen)*. Addiert Godot Hoehe, verschiebt sich der Boden UNTER Wegen, Grundstuecken und Siedlungen - ein bei uns ebener Weg bekommt Wellen, ein Stadtgrundriss steht schief. Gegenmittel: eine Graustufenmaske mitexportieren (0 = nicht rauschen an Wegen/Bauflaechen/Ufern, 1 = volle Wildnis), mit der die Engine die Rauschamplitude multipliziert. Datenquellen dafuer sind alle vorhanden (`roads`, `city_mask`, `plot_map`, Wasserflaechen). | 1 | **ERLEDIGT 2026-08-24:** `daempfungsmaske()` in `map_export.py`, exportiert als `noise_damping_mask.png`. Radien in METERN, damit die Korridorbreite nicht an der Exportaufloesung haengt.
| [ ] | 13.3 | **Erster Export: Hoehe + Biomfarbe + Texturzuordnung** *(Nutzerentscheidung)*. Terrain3D erwartet drei Rasterebenen je Region: **Hoehe als 32-Bit-Float** (unser heutiger Export ist 16-Bit-PNG, `gui/utils/map_export.py` - das quantisiert und muss auf R32F/EXR umgestellt werden), **Control** (Texturzuordnung als Bitfeld) und **Color** (Albedo + Rauheit in Alpha). Die Biomkarte liefert Control und Color. | 3 |
| [ ] | 13.4 | **Streuung ueber Dichtekarten + Seed** *(Nutzerentscheidung)*. Je Art eine Graustufenkarte aus der Biomkarte plus ein Seed: Godot streut beim Laden, gleicher Seed ergibt exakt dieselben Positionen auch nach Neustart. Dateien bleiben klein, Dichte und Artenliste sind aenderbar ohne Neugenerierung der Welt. **Was der Nutzer dafuer vorbereiten muss:** je Art ein Modell mit LOD-Stufen und Kollision, dazu eine Zuordnungstabelle "Biom -> Arten + Dichte". | 2 |
| **[x]** | **13.5** | **Vektordaten exportieren: Fluesse, Wege, Grundstuecksgrenzen, Siedlungen**. Liegen bereits als Linienzuege/Polygone vor (`roads`, `sea_roads`, Flussnetz, `plot_edges`, `settlement_list`) - als JSON-Punktlisten in Weltkoordinaten ausgeben. In Godot daraus Splines: Wege als Decal oder Mesh, Fluesse als Wassermesh entlang der Spline, Grenzen als Zaun/Linie. Meeresoberflaeche braucht keine Daten, nur eine Ebene auf Hoehe 0 mit Wasser-Shader. | 2 | **TEILWEISE ERLEDIGT 2026-08-24:** `vektordaten()` schreibt `vektor.json` mit Wegen, Seewegen, Grundstuecksgrenzen und Ortslagen in METERN. **Fluesse fehlen:** `core/terrain_generator.py:1707` behaelt nur `river_mask`/`river_order` und wirft den Knotengraphen weg - der muesste zuerst aufgehoben werden.
| [ ] | 13.6 | **Hauptmenue mit gebackenen Karten je Ordner** *(Nutzerentscheidung)*. Ein Ordner je Welt: alle Exportdateien plus eine Beschreibungsdatei (Seed, Parameter, Vorschaubild aus der Biomkarte - die gibt es bereits). Das Menue listet die Ordner, Laden rechnet nichts neu. Reines Seed-Speichern wurde verworfen, weil jede Neuberechnung Minuten kostet. | 2 |

# 12 — Aus `docs/TODO.md` uebernommen *(2026-08-12, jene Datei ist damit geloescht)*

`TODO.md` (Stand 2026-08-06) war die Vorgaengerliste. Ihre Bloecke A/B/C/D/E/G
sind vollstaendig in den Abschnitten 1 bis 11 oben aufgegangen — A1/A2/A3/D1
als Abschnitt 11, B1 in Abschnitt 1, C0 als 7.0, D2 als 5.14/6.2, E als
Abschnitt 5, G als Abschnitt 9. Was hier steht, ist der Rest, der oben noch
keine Nummer hatte.

| | # | Sache | Aufwand |
|---|---|---|---|
| **[x]** | **12.1** | **Aufloesungsfrage entschieden** *(TODO B2)*. Die Frage war, ob 1024 px tragbar sind, nachdem `weather.temperature` 171 s von 199 s frass. Nach B1 (`WETTER_GITTER = 256`) und den Messungen vom 2026-08-11 ist sie beantwortet: **Vorgabe steht auf 1024 px** (`value_default.py`, `MAPSIZEMAX = 1024`), eine volle Generierung laeuft dort in **2:41 Min** (siehe 6.14). Kein Zuruecknehmen auf 512 noetig. | — |
| **[x]** | **12.2** | **Erosionsreiter kennzeichnen** *(TODO A4)*. `EROSION_AKTIV = False` macht 9 Ansichten und 13 Regler wirkungslos — das ist gewollt, **sieht aber aus wie ein Fehler**. Entscheidung des Nutzers 2026-08-06: die Kette bleibt vorerst aus, spaeter moeglicherweise wieder an. Daraus folgt die Bauvorgabe: ein Hinweisstreifen oben im Reiter plus gesperrte Regler, aber **an den Schalter gekoppelt statt fest verdrahtet** (derselbe Mechanismus wie A1), damit ein spaeteres Einschalten die Sperre von selbst aufhebt. Der Reiter meldet heute bereits "13 von 13 Reglern gesperrt", es fehlt also nur die sichtbare Begruendung. | 0.5 | **ERLEDIGT 2026-08-24:** `_create_stilllegungs_hinweis()` in `gui/tabs/erosion_tab.py` - gelber Hinweisstreifen, an `EROSION_AKTIV` gekoppelt statt fest verdrahtet.
| **[x]** | **12.3** | **`erosion_strength` ist ZWEIMAL definiert** *(TODO A5, Nebenbefund)* — derselbe Parameterschluessel in zwei Gruppen mit verschiedenen Spannen und Vorgaben: EROSION `0.0–2.0`, Vorgabe **0.5**; WATER `0.1–5.0`, Vorgabe **2.5**. Beide schreiben nach `parameters['erosion_strength']`; welcher gewinnt, haengt an der Reihenfolge der Zusammenstellung (heute EROSION mit 0.5). Der WATER-Eintrag gehoert zum stillgelegten Droplet-Verfahren und verschwindet mit 12.4. **Eine stille Mehrdeutigkeit — genau die Sorte Befund, die spaeter als unerklaerliches Verhalten zurueckkommt.** | 0.5 | **ERLEDIGT 2026-08-24:** nicht geloescht (water_generator liest die Schluessel weiter), sondern in `DOPPELTE_SCHLUESSEL` angemeldet. `tests/smoke_test_parameter_eindeutig.py` schlaegt bei jedem NEUEN unangemeldeten Doppelschluessel fehl. Fand dabei sofort einen zweiten: `octaves` - harmlos, nur der Attributname kollidiert.
| **[x]** | **12.4** | **Acht unerreichbare Regler** *(TODO A5)*: `city_reach_factor`, `civ_influence_range`, `erosion_passes`, `sediment_capacity_factor`, `settling_velocity`, `thermal_erosion_strength`, `frequency`, `map_longitude` — definiert, aber in keinem Reiter. Die Frage "gehoert das nicht zu Erosion?" ist **beantwortet: ja, und es IST schon Erosion** — die vier Wasserregler gehoeren zu den auskommentierten Knoten `water.erosion_sedimentation`/`water.thermal_erosion`, im Graph als `ALTBESTAND DROPLET-EROSION (stillgelegt 2026-07-28)` gefuehrt und ausdruecklich "ersetzt durch `erosion.hydraulic`". Es fehlt keine Erosion; das sind ihre **Vorgaenger**. Vorschlag: die vier Wasserregler mit dem toten `DropletErosionSystem` zusammen loeschen, `map_longitude` neben `map_latitude` in den Wetter-Reiter, die zwei Siedlungsregler mit Block 5 pruefen. | 0.5 | **ERLEDIGT 2026-08-24:** die stillgelegten Droplet-Regler tragen jetzt eine Begruendung in `stillgelegte_regler()`, geprueft von `tests/smoke_test_parameter_eindeutig.py`.
| [ ] | 12.5 | **Rundenbetrieb des Orchestrators ist von keinem Test erfasst** *(TODO H4)*. Faellt weg, sobald 8.2 kommt (topologische Reihenfolge statt Rundensteuerung) — bis dahin eine bekannte Luecke, kein Befund. | — |

# 16 — Nachtrag aus der Uebergabe 2026-08-24 *(nachgetragen 2026-09-23)*

Beim Archivieren am 2026-09-21 (Ticket #51) und am 2026-09-23 (Ticket #44)
sind `docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md`,
`docs/archiv/2026-08-24_ANZEIGE_UND_SEEN.md` und
`docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md` in den Archivordner gewandert,
**ohne dass ihre offenen Punkte vorher hierher uebernommen wurden**.
`docs/archiv/README.md` behauptete zusaetzlich, beide Plaene seien
abgearbeitet (richtiggestellt am 2026-09-23, siehe 16.11). Die elf Punkte
unten sind der wiedergefundene Rest. Jeder Datei:Zeile-Verweis ist am
2026-09-23 am Arbeitsbaum nachgeprueft.

| | # | Sache | Aufwand |
|---|---|---|---|
| [ ] | 16.1 | **Layerwert unter dem Cursor in der 2D-Koordinatenzeile** *(Nutzerwunsch 2026-08-24: "einfach nur bei den koordinaten die angezeigt werden steht ueber was man gerade drueber haelt")*. `_on_mouse_move()` in `gui/widgets/map_display_2d.py:1718-1734` schreibt bis heute nur `"Coordinates: (x, y)"`. Zu ergaenzen ist der Wert des gerade angezeigten Layers — als Zahl bei skalaren Layern, als Name bei kategorischen (Biom, Kuestentyp, Gestein). Pixelposition und Layerdaten liegen im Handler bereits vor. *(frueher A.1 in `docs/archiv/2026-08-24_ANZEIGE_UND_SEEN.md`)* | 1 |
| [ ] | 16.2 | **Maus -> Gelaendepunkt im 3D (Raycast)**. Existiert nicht. Der einzige Raycast im Projekt ist `ShadowCalculator._raycast_shadow_cpu()` (`core/terrain_generator.py:1011`, in `gui/` nur als Kommentarverweis in `gui/widgets/map_display_3d.py:895`), also Schattenwurf — nicht Mausabfrage. Gebraucht wird: Strahl vom Auge durch das Mauspixel, schrittweises Marschieren gegen das Hoehenfeld, dann Verfeinern. Rein rechnerisch und damit headless pruefbar — anders als das Color-Picking aus 6.27 (dort: anklickbare Marker, hier: der Gelaendepunkt). *(frueher A.2)* | 3 |
| [ ] | 16.3 | **Dieselbe Auslesezeile im 3D**. Reine Verdrahtung, sobald 16.1 und 16.2 stehen; faellt unter die stehende 2D/3D-Regel aus `CLAUDE.md`. *(frueher A.3)* | 0.5 |
| [ ] | 16.4 | **NUTZERENTSCHEIDUNG: Standardansicht im Biome-Reiter auf `biome_map_super` umstellen?** `gui/tabs/biome_tab.py:343` waehlt weiterhin "Base Biomes" = `biome_map` vor; dort sind alle Wahrscheinlichkeits-Biome 0, maschinell festgehalten in `tests/smoke_test_flussstufen.py:176-207`. Die Umstellung braechte Straende, Klippen und Ufersaeume sichtbar ins Bild. Gegenueber 2026-08-24 geschrumpft: `alpine_level` und `snow_level` sind seit Commit `14e7e0c` als wirkungslos gesperrt, von sechs Biomen bleiben vier. **Entscheidung, kein Bauauftrag — nicht ungefragt umstellen.** | 0.5 |
| [ ] | 16.5 | **Seeboden der Binnenseen glaetten** *(Nutzerwunsch: "diese koennten ... eine glatte flaeche bekommen damit sie homogen aussehn")*. Gemessen 2026-08-24 (512 px, Seed 20260804): 11 Binnenseen ueber 4 Pixel, groesster 252 Pixel, in Macchia/Estrande/Thalassia/Clonagh. Zwei Wege: mit `_seetiefe_aus_archetyp()` senken (arbeitet heute nur auf `_hauptmeer_maske`, `core/terrain_weltkarte.py:2092-2128`) oder eine glatte Flaeche auf Ueberlaufhoehe — letzteres einfacher. Seit Commit `f09c312` sind Binnenseen aus dem Kuestenband ausgeschlossen (`core/terrain_weltkarte.py:2241-2248`), eine Glaettung muesste also ein eigener Schritt sein. Unabhaengig von 12.6 baubar. *(frueher C.1)* | 1 |
| **[x]** | **16.5a** | **Lake gegen Sea unterscheiden** *(Schwesterpunkt zu 16.5, frueher C.1/C.2)*. **ERLEDIGT 2026-08-25:** `core/biome_generator.py:1519-1556` stuft jedes Becken unter Meeresspiegel ohne Randverbindung als Lake ein (nicht als Ocean — diese Becken sind per Definition nicht mit dem Meer verbunden, und Seewege/Kuestenlogik/Seegliederung lesen `ocean`). Der Archiveintrag trug noch `[ ]`; hier festgehalten, damit es niemand ein zweites Mal baut. | — |
| [ ] | 16.6 | **Verdunstung vom Abfluss abziehen**. Das Flussknotengewicht ist heute reiner Niederschlag; "Abfluss = Niederschlag − Verdunstung" wuerde den Regionskontrast von Faktor 2.2 auf geschaetzt 3-4 verstaerken. `evaporation_map` liegt als Wetterfeld vor (`core/__init__.py:227`), wird im Flussnetz aber nicht gelesen. **Nachrangig — erst messen, ob noetig.** *(frueher 1.3 in `docs/archiv/2026-08-24_FLUESSE_UND_WASSER.md`)* | 1 |
| [ ] | 16.7 | **Fjordarme als Vorfluter, und der Fjord bekommt Kuestentyp Geiranger**. `docs/spezifikation/12_WASSER.md:209-211` fuehrt beide als offen, `docs/OFFENE_PUNKTE.md` kannte sie bisher nicht. Braucht eine Reihenfolgeaenderung: die Kuestenschritte laufen heute VOR den Fluessen (`core/terrain_weltkarte.py:2881-2930` in `weltfeld()`, Fluesse erst in `core/terrain_generator.py:1883`), die Kueste weiss also nicht, wo ein Fluss muendet. Zwei Wege: Fluesse vorziehen (grosser Eingriff) oder ein zweiter Durchgang ueber die Kuestenzone, der den Archetyp an der Muendung nachtraeglich korrigiert. **VOM NUTZER AM 2026-09-23 AUSDRUECKLICH VERTAGT** — steht hier nur, damit es nicht verlorengeht. *(frueher 5.1/5.2)* | 3 |
| [ ] | 16.8 | **Saisonale Schneeschmelze / saisonale Fluesse**. Nutzer 2026-08-24: *"saisonal kommt spaeter denke ich."* Die sechs saisonalen Perioden existieren im Wettergenerator (`core/weather_generator.py:271-273`), das Flussnetz rechnet dagegen mit dem Jahresschnappschuss. *(frueher 5.3)* | 2 |
| [ ] | 16.9 | **Seenlandschaft in der Morobora** ("so wie in Lappland"). Nutzer: *"ich finde taiga sollte seenlandschaften haben."* Gemessen 2026-08-24: keiner der 11 Binnenseen liegt in der Morobora oder im Skerrheim. Ein Lappland-See liegt auf ~200 m Hoehe — dafuer fehlt der Mechanismus; die laengliche Form waere glaziale Rinnenbildung. Erster Schritt: `water.lake_detection` (`core/water_generator.py:3126`). Gehoert sachlich unter 9.3 ("erst Erosion, dann Wasser samt Seenbildung"), wo es bisher nicht genannt ist. *(frueher Block 6)* | 3 |
| [ ] | 16.10 | **Es gibt keine gueltige Live-Pruefliste mehr.** `docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md` wurde am 2026-09-21 archiviert, ausdruecklich mit dem Vermerk, dass Punkte "erledigt oder ueberholt" sind, "ohne dass das im Dokument nachgefuehrt wurde". Ein Nachfolger existiert nicht. `docs/spezifikation/15_ANZEIGE.md:236-237` verweist inzwischen hierher (auf `docs/OFFENE_PUNKTE.md`, Eintraege markiert als "nur live pruefbar") — **entweder diese Regelung bestaetigen oder eine eigene Liste anlegen.** | 0.5 |
| **[x]** | **16.11** | **Falschaussage in `docs/archiv/README.md`** — die Ursache dafuer, dass 16.1 bis 16.9 verlorengingen: das Archiv-README erklaerte die Bloecke beider 2026-08-24-Plaene fuer abgearbeitet, obwohl `2026-08-24_FLUESSE_UND_WASSER.md` bei 1.3/5.1/5.2/5.3/Block 6 und `2026-08-24_ANZEIGE_UND_SEEN.md` bei A.1/A.2/A.3/C.1 selbst noch `[ ]` trug. **ERLEDIGT 2026-09-23:** beide Zeilen benennen jetzt getrennt, was abgearbeitet ist und was hier unter Abschnitt 16 weiterlaeuft; zugleich die ueberholte Zuordnung `§4.1–4.7 -> 5.1–5.7` in der Zeile zu `2026-09-22_SIEDLUNGEN_ENTWURF.md` auf die heutigen Abschnitte 5.1–5.5 berichtigt. | 0.5 |
