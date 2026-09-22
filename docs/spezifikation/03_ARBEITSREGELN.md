# Arbeitsregeln, Tests und Werkzeuge

**Was das hier ist.** Wie gearbeitet wird: in welcher Reihenfolge, mit
welchen Werkzeugen, und was einen Test in diesem Projekt brauchbar macht.
Die Prüfliste **was** dabei jedes Mal zu prüfen ist, steht getrennt in
[02_INVARIANTEN.md](02_INVARIANTEN.md).

*Herkunft: `docs/archiv/2026-07-29_SPEZIFIKATION.md` (Stand 2026-07-29), §5
„Arbeitsregeln" und §6 „Werkzeuge", sowie `docs/archiv/2026-09-16_SOLLBESCHREIBUNG.md` (Stand
2026-09-16), Abschnitt „Testing Decisions".*

---

## 1. Reihenfolge, Messfallen, Leitfragen

### 1.1 Reihenfolge

1. **Messen, bevor geändert wird.** Ohne Ausgangswert ist keine Verbesserung
   belegbar.
2. **Eine Sache auf einmal.** Amplitude und Glättung gleichzeitig zu ändern
   kostete eine halbe Stunde Suche, bis eine Gegenprobe zeigte, welche es war.
3. **Ansehen, nicht nur nachrechnen.** Vier Kennzahlen haben die
   45°-Pyramiden nicht gefunden; ein Bild zeigte sie sofort. Die
   Drainage-Dichte stufte die beste Variante als schlechteste ein.
4. **Gegenprobe.** Eine Zusicherung, die auch ohne die Änderung hält, prüft
   nichts.
5. **Nach der Änderung die volle Prüfliste** aus §4 — nicht nur den Test, der
   zur Änderung gehört.

### 1.2 Messfallen, die schon zugeschlagen haben

- Werkzeug liefert stillschweigend etwas anderes als angefragt.
  `build_terrain(512)` gab ein 256er Array zurück; alle „512 px"-Messungen
  waren falsch beschriftet. Seither eine `assert`-Zeile.
- Messung am halb umgebauten Stand. Die Konvergenz der Rückkopplung schien
  sauber einzuschwingen — gemessen wurde vor dem Umstellen der Lesestellen,
  also Wiederholungen ohne Rückkopplung.
- Maschinenrauschen als Ergebnis. Identischer Code, dreimal: 12.9 / 12.6 /
  19.2 s. Zeitmessungen brauchen den Median mehrerer Läufe.
- Eigener Erklärkommentar wird von der Textsuche gefunden, die den Code prüfen
  soll. Für Codeprüfungen den Syntaxbaum benutzen.
- Falsche Formel im Messgerät. Die Senkenmessung hatte Rand und Innenbereich
  vertauscht und meldete 627 m tiefe Löcher, die es nicht gab.

### 1.3 Was immer mitzudenken ist

Bei jeder Änderung diese vier Fragen beantworten:

1. **Wie ginge es besser?** Ist das die Ursache oder ein Symptom?
2. **Rechnen CPU und GPU noch dasselbe?**
3. **Welcher Regler gehört dazu, und mit welchem hängt er zusammen?**
4. **Was in §4 könnte ich damit gerade kaputt machen?**

---

## 2. Werkzeuge

Der Bestand am 2026-09-23: 29 Werkzeuge unter `tools/` und 94 Smoke-Tests
unter `tests/`. Die wichtigsten:

| Werkzeug | Zweck |
|---|---|
| `tools/erosion_lab.py`, `tools/erosion_filter_lab.py` | Erosionsvarianten auf der GPU, Formkennzahlen, Kontaktabzug |
| `tools/weather_lab.py` | Wasserbilanz, Rückkopplungs-Konvergenz, Klima je Breite |
| `tools/regionen_lab.py`, `tools/region_gegen_vorbild.py` | Regionen gegen ihre realen Vorbilder messen |
| `tools/vorbilder_holen.py`, `tools/vergleich_vorbild.py` | echte Geländedaten (SRTM/Copernicus) holen und gegenhalten |
| `tools/flussnetz_lab.py`, `tools/flussnetz_werkstatt.py`, `tools/drainage_lab.py` | Flussnetz und Entwässerung |
| `tools/vektor_kueste_labor.py`, `tools/kuestenkatalog_bauen.py`, `tools/kuestenlaengsschnitt.py` | Küstenarchetypen bauen und vermessen |
| `tools/biome_lab/` | Parzellenphysik außerhalb des Editors — der Beleg, dass eigene Programme neben dem Editor tragen |
| `tools/testlauf.py`, `tools/test_raenge.py` | Testläufe, getrennt nach Wächter und Eichung |
| `tools/nachtlauf.py` | Nachtbranch starten, Ticket abschließen, Morgenbericht |
| `tools/pipeline_kritischer_pfad.py` | wo die Rechenzeit tatsächlich hingeht |

**Kontaktabzug und Querschnitt sind Teil jeder Messung**, nicht Zierde.

### 2.1 Was fehlt

- Ein Prüfstand, der **alle** Komponenten gleichzeitig bewertet, statt eine.
  Solange das fehlt, bleibt „ich reiße mit dem Hinterteil etwas ein"
  unbemerkt.
- Zielwerte für Mäander und Flussbreite ([12_WASSER.md](12_WASSER.md)) und
  für den Biom-Zusammenhang ([13_KLIMA_UND_BIOME.md](13_KLIMA_UND_BIOME.md)).
  Die Windziele sind seit 2026-08-11 erreicht.
- Die noch offenen Zielwerte je Region — Anzahl Flüsse, Flussbreite,
  Biom-Flächenanteile. Siehe [10_REGIONEN.md](10_REGIONEN.md), Abschnitt
  „Noch offen".

---

## 3. Was einen guten Test hier ausmacht

Die Lehren dieses Projekts, alle teuer erkauft:

* **Das Ergebnis prüfen, nicht den Parameter.** Zehn grüne Tests verdeckten
  wochenlang eine Funktion, die im Betrieb nie aufgerufen wurde (das adaptive
  3D-Netz).
* **Mit den echten Eingabegrößen bauen** — 128/256/512/1024. Eine ausgedachte
  Größe prüft eine ausgedachte Situation. Genau daran scheiterte die Prüfung
  des adaptiven Netzes: gebaut mit 129/257/513, während das Programm
  Zweierpotenzen benutzt.
* **Sind mehrere Einzelprüfungen grün und das Ergebnis trotzdem falsch:
  aufhören, Einzelglieder zu prüfen — zwei Enden der Kette gegeneinander
  messen.** Das fand den Verteilungsfehler der Küstenarchetypen in einem
  Schritt, nachdem sechs Einzelhypothesen ergebnislos geblieben waren.
* **Über die Naht prüfen, nicht daran vorbei.** Was ein Test nur erreicht,
  indem er ins Innere greift, hat vermutlich die falsche Form.

*Herkunft: `docs/archiv/2026-09-16_SOLLBESCHREIBUNG.md` (Stand 2026-09-16), Abschnitt „Was einen
guten Test hier ausmacht".*

## 4. Das Prüfumfeld

**Ausgangslage 2026-09-16:** 70 Testdateien, 19 Minuten Gesamtlaufzeit, 60
grün. Die Erzeugung ist seed-abhängig, teils auf der GPU, und die wertvollste
Prüfung ist bis heute der Blick des Nutzers auf den Bildschirm.

Entschieden über [#22](https://github.com/nikolaisoerensen/MapGenerator/issues/22):

* **Zwei Ränge.** **Wächter** laufen unter zwei Minuten und gehören zu jeder
  Änderung; die **Eichung** darf nachts eine Stunde brauchen. 19 Minuten für
  alles ist für keinen der beiden Zwecke die richtige Zahl.
  ([#46](https://github.com/nikolaisoerensen/MapGenerator/issues/46))
* **Zusicherungen als Band, nicht als Punkt.** „Der Wasseranteil liegt
  zwischen 4 und 9 Prozent", nicht „ist 6,5". Die Bandgrenzen stehen als
  **versionierte Daten** im Repo, nicht verstreut im Testcode — dann ist an
  der Dateihistorie ablesbar, wann ein Ziel verschoben wurde und warum.
  ([#47](https://github.com/nikolaisoerensen/MapGenerator/issues/47))
* **Fester Seed für die Wächter, drei wechselnde für die nächtliche Eichung.**
  Der feste Seed macht Fehlschläge reproduzierbar; die wechselnden fangen ab,
  was nur bei genau diesem einen Seed zufällig gut aussieht.
  ([#48](https://github.com/nikolaisoerensen/MapGenerator/issues/48))
* **Jeder rote Test bekommt ein eigenes Ticket mit Frist.** Kein Sammelposten:
  eigene Zeile, eigene Ursache, eigenes Datum. Bis dahin ist er ausdrücklich
  als bekannt markiert, danach ist er ein Fehler.
  ([#49](https://github.com/nikolaisoerensen/MapGenerator/issues/49))
* **Bildvergleich mit Toleranz** ersetzt die Sichtprüfung nicht, grenzt sie
  aber auf das ein, was wirklich neu aussieht: fester Seed, fester Ausschnitt,
  hinterlegtes Referenzbild, Abweichung in Prozent.
  ([#50](https://github.com/nikolaisoerensen/MapGenerator/issues/50))
* **Erst den Bestand sichten, dann neue Tests schreiben.** Es braucht deutlich
  mehr Tests als heute — aber nicht mehr von denselben.
  ([#45](https://github.com/nikolaisoerensen/MapGenerator/issues/45), Ergebnis
  in `docs/TESTBESTAND_BEWERTUNG.md`)

*Herkunft: `docs/archiv/2026-09-16_SOLLBESCHREIBUNG.md` (Stand 2026-09-16), Abschnitt „Das
Prüfumfeld".*

## 5. Der nächtliche Betrieb

Entschieden über [#23](https://github.com/nikolaisoerensen/MapGenerator/issues/23);
das Verfahren im Einzelnen steht in `docs/NACHTBETRIEB.md`.

* **Ein Ticket ist nachttauglich, wenn alle seine Abnahmekriterien als Test
  formuliert sind.** Ein Ticket, dessen Kriterien man nicht als Test schreiben
  kann, ist kein Nachtticket.
* **„Fertig" ist ein Messwert, keine Meinung** — es hängt an grünen Tests,
  nicht an der Einschätzung des Agenten.
* **Eine Sperrliste im Repo** (`nachtbetrieb/sperrliste.toml`) nennt Dateien
  und Themen, an die nachts niemand geht.
  ([#57](https://github.com/nikolaisoerensen/MapGenerator/issues/57))
* **Alles auf einen Nacht-Branch, morgens ein Merge.** Nachts wird nur dort
  geschrieben, `main` bleibt unberührt. Morgens wird gegengelesen — auch mit
  Code-Review — und einzelne Fehler werden korrigiert, bevor zusammengeführt
  wird. ([#58](https://github.com/nikolaisoerensen/MapGenerator/issues/58))
* **Zeitgrenze je Ticket.** Läuft sie ab, bleibt der Branch stehen und eine
  Notiz geht ins Ticket: was versucht wurde, woran es hing. Kein Agent
  arbeitet sich stundenlang an derselben Stelle fest.
  ([#59](https://github.com/nikolaisoerensen/MapGenerator/issues/59))
* **Ein Morgenbericht auf einer Seite:** was geschlossen wurde, was rot ist,
  welche Kennzahlen sich bewegt haben, welche stillen Rückfälle gemeldet
  wurden. ([#60](https://github.com/nikolaisoerensen/MapGenerator/issues/60))

*Herkunft: `docs/archiv/2026-09-16_SOLLBESCHREIBUNG.md` (Stand 2026-09-16), Abschnitt „Der
nächtliche Betrieb".*
