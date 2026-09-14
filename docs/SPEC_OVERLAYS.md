# Spezifikation — Overlays bekommen eine Naht

Angelegt 2026-09-14 aus dem Architekturbericht derselben Sitzung (Kandidat 1
von fuenf). Die Naht wurde vom Nutzer gewaehlt: **die bestehende Naht in
`BaseMapTab` wird verbreitert**, keine neue.

Vokabular dieser Datei: **module**, **interface**, **implementation**,
**seam**, **adapter**, **deep**/**shallow**, **leverage**, **locality** —
absichtlich englisch und unveraendert, weil sie aus dem Entwurfsvokabular
stammen und nicht mit "Komponente", "Schnittstelle" oder "Grenze" vermischt
werden sollen. Alles andere ist deutsch wie im Rest von `docs/`.

---

## Problem Statement

Der Nutzer setzt in einem Reiter einen Haken — "Settlements", "Flussnetz",
"Regionen" — und sieht in der 2D-Ansicht etwas erscheinen. Schaltet er auf
3D um, ist es weg. **Kein Fehler, keine Meldung, keine Logzeile.** Der Haken
bleibt gesetzt und sieht aus, als wirke er.

Das ist seit dem 2026-08-24 **viermal** passiert, jedes Mal mit demselben
Mechanismus, und dreimal wurde es als behoben verbucht:

| Datum | Betroffen | Als behoben verbucht |
|---|---|---|
| 2026-08-24 | `overlay_river_generations` | ja |
| 2026-08-25 | `overlay_river_network` (Biome-Reiter) | ja |
| 2026-08-25 | `overlay_settlements` (Biome-Reiter) | ja |
| **heute, unbemerkt** | **derselbe Biome-Reiter, erneut** | — |

Der vierte Fall steht heute im Arbeitsverzeichnis. `BiomeTab.apply_overlays()`
steigt in der ersten Zeile aus, wenn die Ansicht nicht 2D ist. Die 3D-Zweige
darunter — am 2026-08-25 ausdruecklich als Behebung der Faelle 2 und 3
eingebaut, mit 25 Zeilen Begruendung darueber — **koennen nie ausgefuehrt
werden.** Betroffen sind Siedlungen *und* Flussgenerationen, letztere obwohl
die Methode auf beiden Anzeigen existiert.

Der Waechtertest meldet trotzdem gruen. Schlimmer: seine Ausnahme fuer
`overlay_settlements` **begruendet sich mit genau der unerreichbaren Zeile**
("Der Biome-Reiter ruft seit dem 2026-08-25 direkt daneben
`update_overlay_data(...)` fuer die 3D-Ansicht"). Der Test ist gruen *wegen*
des toten Codes.

Die Ursache ist nicht Nachlaessigkeit. Jeder Reiter baut die Weiche zwischen
2D und 3D selbst, per `hasattr(display, "...")` — 43 solcher Abfragen in
`gui/`. Trifft die Weiche nicht zu, tut der Zweig nichts, und **ein fehlender
3D-Weg ist von einem funktionierenden nicht zu unterscheiden.** Wer den
Fehler machen will, muss nichts falsch tippen; er muss nur eine Methode
aufrufen, die es auf der anderen Anzeige nicht gibt.

Die stehende Nutzervorgabe vom 2026-08-25 lautet woertlich: *"kann man
irgendwo festhalten dass wenn du etwas umsetzt es immer auch in 3D gleich
umgesetzt wird? weil sonst muss ich das immer wieder sagen."* Sie steht in
`CLAUDE.md`. Sie ist eine Regel, an die sich ein Mensch erinnern muss — und
genau daran ist sie viermal gescheitert.

---

## Solution

Jeder Haken wirkt in beiden Ansichten, weil **kein Reiter die Weiche mehr
selbst baut.**

Ein Reiter meldet an, *was* gezeigt werden soll — eine Liste von Overlays als
Werte. Ein einziges **deep module** entscheidet, *wie*: in 2D per
`overlay_*()`-Aufruf, in 3D per RGBA-Haut auf dem Gelaende. Der Reiter kennt
den Unterschied nicht mehr und kann ihn deshalb nicht mehr falsch treffen.

Wo ein Overlay keinen 3D-Weg hat, **sagt das Programm es laut** — beim Start,
als Fehler mit Namen, nicht als stilles Nichtstun. Die Restschuld aus der Zeit
vor der Regel bleibt eine benannte, angemeldete Liste; neu dazukommen kann
nichts mehr, ohne dass es auffaellt.

Die stehende Regel hoert damit auf, eine Gedaechtnisleistung zu sein, und wird
zu einer Eigenschaft eines Moduls.

**Der entscheidende Vorgriff steht schon im Programm.**
`BaseMapTab._push_data_to_current_display()` loest genau dieses Problem fuer
Skalarlayer (Temperatur, Niederschlag, Hangneigung) — und loest es *richtig*:
es schiebt die Daten **immer** an `self.map_display_3d`, unabhaengig von
`current_view`, weil diese Anzeige je Reiter ohnehin existiert, auch waehrend
2D sichtbar ist. Der Kommentar dort begruendet das ausdruecklich. Die
Vektor-Overlays haben diesen Weg nie bekommen — sie haengen bis heute an
`current_view`, und das ist der Fehler. Diese Spezifikation traegt das Muster
nach, das nebenan seit Monaten funktioniert.

---

## User Stories

### Der Kartenbauer

1. Als Kartenbauer moechte ich, dass ein gesetzter Haken in der 3D-Ansicht
   dasselbe zeigt wie in der 2D-Ansicht, damit ich der Anzeige glauben kann.
2. Als Kartenbauer moechte ich, dass ein Haken, den ich in 2D setze, nach dem
   Umschalten auf 3D bereits wirkt, ohne dass ich neu generieren muss.
3. Als Kartenbauer moechte ich, dass ein Haken, den ich in 3D abwaehle, die
   Textur auch wirklich entfernt, statt sie liegen zu lassen.
4. Als Kartenbauer moechte ich Siedlungen, Landmarks und Roadsites in 3D
   sehen, so wie ich sie in 2D sehe, damit ich Lagen im Gelaende beurteilen
   kann.
5. Als Kartenbauer moechte ich das Flussnetz in 3D sehen, damit ich pruefen
   kann, ob die Laeufe den Taelern folgen — das geht im flachen Bild gar
   nicht.
6. Als Kartenbauer moechte ich im Biome-Reiter dasselbe Flusssystem sehen wie
   im Fluss-Reiter, damit ich nicht zwei widersprechende Bilder vergleiche.
7. Als Kartenbauer moechte ich die Regionsfarben in 3D sehen, damit ich
   beurteilen kann, ob eine Regionsgrenze im Gelaende sichtbar ist oder als
   Naht auffaellt.
8. Als Kartenbauer moechte ich das Regionsgitter in 3D sehen, damit ich die
   neun Spielkarten im raeumlichen Bild wiederfinde.
9. Als Kartenbauer moechte ich die Stadtgrenzkontur in 3D sehen, damit ich
   beurteilen kann, ob eine Stadt an einem Hang klebt.
10. Als Kartenbauer moechte ich die Parzellengrenzen in 3D sehen, damit ich
    beurteilen kann, ob das Gewebe der Gelaendeform folgt.
11. Als Kartenbauer moechte ich, dass mehrere gleichzeitig gesetzte Haken sich
    in 3D nicht gegenseitig ausloeschen, sondern uebereinanderliegen.
12. Als Kartenbauer moechte ich, dass ein Haken ohne Daten (noch nicht
    gerechnet) in beiden Ansichten gleich reagiert — naemlich nichts zeigt —
    statt in einer Ansicht etwas Veraltetes stehen zu lassen.
13. Als Kartenbauer moechte ich, dass sich beim Wechsel des Reiters die
    Overlays des vorigen Reiters nicht in die neue Ansicht mitschleppen.
14. Als Kartenbauer moechte ich beim Umschalten zwischen 2D und 3D keine
    spuerbare Wartezeit, die es vorher nicht gab.
15. Als Kartenbauer moechte ich, dass ein Overlay, das es in 3D
    ausdruecklich nicht geben soll (die Plot-Physik-Momentaufnahme), in 3D
    ruhig fehlt — aber weil das entschieden wurde, nicht weil es vergessen
    wurde.

### Der Entwickler

16. Als Entwickler moechte ich ein neues Overlay an einer einzigen Stelle
    anmelden, damit ich nicht daran denken muss, den 3D-Weg zu bauen.
17. Als Entwickler moechte ich, dass ein Overlay ohne 3D-Weg beim Start einen
    Fehler mit Namen wirft, damit der Fehler mich findet und nicht der Nutzer.
18. Als Entwickler moechte ich in einem Reiter keine `hasattr`-Weiche mehr
    schreiben muessen, damit die Fehlerklasse aus dem Reiter verschwindet.
19. Als Entwickler moechte ich, dass die beiden Namensraeume fuer Layer (2D-
    Datenschluessel gegen 3D-UI-Namen) an einer Stelle uebersetzt werden,
    damit ich die Zuordnung nicht in vier Dateien nachschlage.
20. Als Entwickler moechte ich die Zeichenentscheidung ohne Qt und ohne
    OpenGL pruefen koennen, damit ich nicht fuer jede Aenderung das Programm
    starten muss.
21. Als Entwickler moechte ich, dass die Restschuld (was in 2D geht und in 3D
    noch nicht) eine angemeldete Liste bleibt, die nur schrumpfen darf.
22. Als Entwickler moechte ich, dass ein Overlay, das nur aus gutem Grund
    einseitig ist, von einem, das schlicht fehlt, unterscheidbar bleibt.
23. Als Entwickler moechte ich die Reihenfolge, in der Overlays uebereinander
    liegen, an einer Stelle festlegen, statt sie aus der Aufrufreihenfolge in
    sechs Reitern zu erraten.
24. Als Entwickler moechte ich, dass zwei Reiter, die dasselbe Overlay zeigen
    (Siedlungen in Siedlungs-, Biome- und Regional-Reiter), dieselbe
    implementation benutzen, damit sie nicht auseinanderlaufen.
25. Als Entwickler moechte ich beim Lesen eines Reiters sehen, *welche*
    Overlays er zeigt, ohne 140 Zeilen Weichenlogik zu durchsuchen.

### Der Pruefer

26. Als Pruefer moechte ich feststellen koennen, ob ein Reiter einen
    3D-Zweig hat, der **erreichbar** ist — nicht nur, ob eine Methode
    existiert.
27. Als Pruefer moechte ich, dass unerreichbarer Anzeigecode den Test
    fehlschlagen laesst, damit sich der heutige Fall nicht wiederholt.
28. Als Pruefer moechte ich, dass eine Ausnahme im Waechtertest nicht mit
    Code begruendet werden kann, der nie laeuft.
29. Als Pruefer moechte ich fuer jedes angemeldete Overlay pruefen koennen,
    dass beide adapter gerufen werden, mit vergleichbaren Daten.
30. Als Pruefer moechte ich, dass der Test die echten Kartengroessen benutzt
    (128/256/512/1024), nicht ausgedachte — die Lehre aus dem adaptiven Netz.
31. Als Pruefer moechte ich, dass jeder verbleibende stille Rueckfall eine
    laute Logzeile schreibt, damit Erfolg und Ausfall unterscheidbar bleiben.

---

## Implementation Decisions

### 1. Die Naht liegt in `BaseMapTab` — es ist die bestehende

**Entscheidung des Nutzers.** Es entsteht **keine neue seam nach aussen.** Die
Verantwortung "Anzeigedaten ans richtige Display schicken" liegt heute schon
bei `BaseMapTab`; sie wird um Vektor-Overlays erweitert statt neben ihr eine
zweite Stelle aufzumachen.

Die Reiter bekommen eine Methode als Geschwister zu
`_push_data_to_current_display()`. Sie nimmt eine Liste von Overlays
entgegen. Das ist das gesamte interface, das ein Reiter kennen muss.

```
class BaseMapTab:
    def _push_overlays(self, overlays: list[Overlay]) -> None:
        """Wie _push_data_to_current_display, aber fuer Vektor-Overlays.
        Laeuft IMMER gegen beide Anzeigen, nicht gegen current_view."""
```

```
# im Reiter:
self._push_overlays([
    Overlay("siedlungen", sichtbar=self.cb.isChecked(), daten=(orte, marken)),
    Overlay("fluesse",    sichtbar=self.rivers_cb.isChecked(), daten=gen_karte),
])
```

*(Beide Ausschnitte stammen aus der Entwurfsvorlage, die der Nutzer gewaehlt
hat. Sie stehen hier, weil sie die Entscheidung genauer festhalten als Prosa
— nicht als fertiger Code.)*

### 2. Ein Overlay ist ein Wert, kein Aufruf

Ein Overlay traegt: **Name**, **sichtbar ja/nein**, **Daten**. Nichts sonst.
Insbesondere traegt es *nicht*, wie gezeichnet wird — das ist die Aufgabe des
Moduls dahinter, und genau deshalb kann ein Reiter es nicht mehr falsch
machen.

`sichtbar=False` ist ein vollwertiger Zustand, kein "weglassen": erst dadurch
kann das Modul eine liegengebliebene 3D-Textur abraeumen. Das heutige
Vergessen dieses Falls ist der Grund fuer `clear_river_overlay`.

### 3. Hinter der Naht liegt ein Register, kein `if`

Ein **internes module** — privat zur implementation von `BaseMapTab`, aber
mit eigenen Tests — haelt je Overlay-Namen beide Wege:

* den **2D-adapter**: welche `overlay_*()`-Methode mit welchen Argumenten,
* den **3D-adapter**: welche `rasterize_*_rgba()`-Funktion, unter welchem
  `(bereich, layername)` die Textur abgelegt wird.

Ein Name ohne Eintrag ist ein Fehler, keine stille Auslassung. Das Register
ist die einzige Stelle, an der die beiden Dialekte nebeneinander stehen —
das ist die **locality**, um die es hier geht.

### 4. Immer beide Anzeigen bedienen, nie `current_view` fragen

**Die zentrale Regel dieser Spezifikation.** Das Modul schiebt an *beide*
Anzeigen, unabhaengig davon, welche gerade sichtbar ist. Genau so macht es
`_push_data_to_current_display()` heute fuer Skalarlayer, mit
ausgeschriebener Begruendung: die 3D-Anzeige existiert je Reiter immer, auch
waehrend 2D sichtbar ist; wer sie nur bei `current_view == "3d"` fuellt,
zeigt beim Umschalten ein leeres Bild oder muss neu generieren.

`current_view` darf in der neuen implementation **nicht vorkommen.** Das ist
nachpruefbar und wird nachgeprueft.

### 5. Ein fehlender 3D-Weg wird laut

Drei Zustaende, drei verschiedene Reaktionen — heute sind alle drei dasselbe
Nichts:

| Zustand | Reaktion |
|---|---|
| Overlay-Name unbekannt | Fehler beim Start, mit Namen |
| Overlay angemeldet, 3D-Weg fehlt, **nicht** in der Schuldliste | Fehler beim Start |
| Overlay angemeldet, 3D-Weg fehlt, **in** der Schuldliste | laeuft, schreibt eine WARNING je Anlass |

Das folgt der Projektlehre, die zweimal teuer war (Shaderpfade nach dem
Dateiumzug, adaptives Netz): *jeder stille Rueckfall auf einen Ersatzpfad
braucht eine laute Logzeile.*

### 6. Die beiden Namensraeume werden zusammengefuehrt

Heute uebersetzt `_LAYER_NAME_MAP_3D` in `base_tab.py` die Skalarlayer
(`"temp_map"` → `"temperature"`); die 3D-Layernamen der Vektor-Overlays
(`"uebersicht"`, `"wegbaender"`) stehen dagegen als blanke Zeichenketten in
den Reitern, ohne Register — ein Tippfehler dort erzeugt kein `KeyError`,
sondern ein Nichts. Beide Zuordnungen ziehen in das Register aus Punkt 3.

### 7. Was sich nicht aendert

* **Die Anzeigeklassen bleiben, wie sie sind.** `MapDisplay2D` lernt nichts
  Neues, `MapDisplay3D` auch nicht. Die vom Nutzer verworfene Alternative
  waere gewesen, der 3D-Klasse dieselben `overlay_*`-Methoden zu geben; sie
  haette neun Methoden und vier neue Rasterfunktionen gekostet und waere nur
  am laufenden Programm pruefbar gewesen.
* **Die `rasterize_*_rgba()`-Funktionen bleiben, wo sie sind** und werden
  weiterverwendet. Fuenf davon gibt es schon.
* **Wege bleiben echte Bandgeometrie** in `gui/widgets/wege_geometrie.py`,
  keine Textur — sie muessen beim Zoomen scharf und anklickbar bleiben. Das
  Register kennt sie als eigenen 3D-adapter, nicht als Rasterweg.
* **Skalarlayer bleiben bei `_push_data_to_current_display()`.** Die beiden
  Methoden stehen nebeneinander und teilen sich das Register, verschmelzen
  aber nicht.

### 8. Der heutige Fehler wird in derselben Aenderung behoben

Der vorzeitige Ausstieg in `BiomeTab.apply_overlays()` faellt weg, und zwar
**nicht** als Einzelreparatur, sondern weil der Reiter auf `_push_overlays()`
umgestellt wird. Der Kommentarblock darueber wird richtiggestellt: er
beschreibt heute eine Behebung, die nie gewirkt hat.

### 9. Die Ausnahme im Waechtertest wird neu begruendet

Der Eintrag fuer `overlay_settlements` in `NUR_EINE_ANZEIGE` beruft sich auf
unerreichbaren Code. Nach dem Umbau stimmt die Aussage — aber die Begruendung
muss auf das Register zeigen, nicht auf eine Zeile in einem Reiter.

### 10. Reihenfolge des Umbaus

Ein Reiter zuerst, vollstaendig, dann Sichtpruefung, dann die anderen. Das ist
die im Projekt bestaetigte Arbeitsweise und hier besonders angebracht, weil
die Wirkung nur am laufenden Programm zu sehen ist.

Empfohlener erster Reiter: **Biome** — dort steht der lebende Fehler, dort
sind beide betroffenen Overlays (Siedlungen, Fluesse) vertreten, und beide
haben ihren Rasterweg bereits.

Danach: Siedlungen, Regional, Fluss. Der Regional-Reiter ist der
unangenehmste — dort verschluckt ein `except Exception: logger.debug` um den
ganzen Overlay-Block herum jeden Fehler, und **alle fuenf** Weichen greifen in
3D nicht.

---

## Testing Decisions

### Was einen guten Test hier ausmacht

Geprueft wird **beobachtbares Verhalten am interface**: "welche Aufrufe kommen
bei welcher Anzeige an". Nicht geprueft wird, wie das Modul intern entscheidet
— das Register darf sich aendern, ohne einen Test anzufassen.

Der entscheidende Gewinn: die Zeichenentscheidung wird dadurch **ohne Qt und
ohne OpenGL** pruefbar. Bisher war alles zwischen "Daten liegen vor" und "Bild
erscheint" nur am laufenden Programm zu sehen — genau deshalb blieben die vier
Faelle unbemerkt.

### Was geprueft wird

1. **Das Overlay-Modul, gegen mitschreibende Attrappen.** Zwei
   Attrappen-Anzeigen, die jeden Aufruf protokollieren. Je angemeldetem
   Overlay: `sichtbar=True` erzeugt in **beiden** Protokollen einen Eintrag;
   `sichtbar=False` erzeugt in beiden das Abraeumen. Ein unbekannter Name
   wirft. Kein Qt.
2. **`current_view` kommt nicht vor.** Ein Quelltexttest ueber die neue
   implementation. Klingt grob, trifft aber genau die Ursache aller vier
   Faelle.
3. **Erreichbarkeit statt Existenz.** Der bestehende Waechtertest wird
   erweitert: er soll nicht nur fragen, *ob* eine Methode auf einer der
   Klassen existiert, sondern ob der Zweig, der sie ruft, ueberhaupt laufen
   kann. Ein vorzeitiges `return`, das alle 3D-Zweige einer Methode
   abschneidet, muss den Test fehlschlagen lassen. **Ohne diesen Punkt
   wiederholt sich der Fall.**
4. **Die Schuldliste darf nur schrumpfen.** Bleibt wie heute, mit der
   Ergaenzung, dass eine Begruendung nicht auf Reiter-Zeilen zeigen darf.
5. **Echte Kartengroessen.** 128/256/512/1024, nicht ausgedacht — die Lehre
   aus dem adaptiven Netz, wo zehn gruene Tests eine Funktion pruefen, die im
   Betrieb nie lief.

### Vorbild im Projekt

* `tests/smoke_test_display_methoden_existieren.py` — der heutige Waechter.
  Wird erweitert, nicht ersetzt. Seine Trennung zwischen "einseitig und
  richtig so" und "einseitig und eine Schuld" ist gut und bleibt.
* `tests/smoke_test_display_2d.py` — deckt alle 30 2D-Darstellungen ab. Das
  Muster (jede Darstellung einmal anfassen) ist die Vorlage fuer die
  Abdeckung der Overlay-Liste.
* `tests/smoke_test_layer_2d_3d_parity.py` — prueft heute die Zuordnung der
  Skalarlayer zwischen den Namensraeumen. Naechster Verwandter; sollte nach
  dem Umbau gegen das Register laufen statt gegen die Tabelle in `base_tab`.
* `tests/smoke_test_pipeline_outputs.py` — **das methodisch beste Vorbild im
  Repo.** Es faehrt denselben Ablauf zweimal, mit und ohne ShaderManager, und
  meldet jeden Unterschied. Genau diese Form — denselben Zustand ueber beide
  adapter fahren und vergleichen — ist das, was fuer 2D/3D bisher fehlt.

### Was sich nur am laufenden Programm pruefen laesst

Dass die Textur **richtig aussieht**. Die Tests koennen belegen, dass beide
adapter mit vergleichbaren Daten gerufen werden; ob die RGBA-Haut im 3D an der
richtigen Stelle sitzt und lesbar ist, sieht nur der Nutzer. Dafuer je
umgestelltem Reiter ein Eintrag in `docs/PRUEFLISTE_LIVE.md`.

---

## Out of Scope

* **Die fuenf Eintraege der Schuldliste tatsaechlich in 3D bauen.** Diese
  Spezifikation baut den *Mechanismus* und macht die Schuld sichtbar; sie
  loest sie nicht ein. Wo ein Rasterweg schon existiert (Regionen), kann er
  im Vorbeigehen mitgenommen werden — wo einer fehlt (Regionsgitter,
  Stadtgrenzkontur, Hoehenlinien), ist das eigene Arbeit. Ob Parzellengrenzen
  in 3D ueberhaupt sinnvoll sind, ist ausdruecklich unentschieden: Tausende
  Parzellen koennten Pixelmatsch werden.
* **Kandidaten 2 bis 5 aus dem Architekturbericht** — die GPU/CPU-Naht, der
  Regionskatalog als Wert, der `DataLODManager`, die vier Schreibweisen je
  Parameter. Eigene Spezifikationen.
* **`gui/tabs/overview_tab.py`.** Der Reiter ruft vier Methoden auf, die es
  nirgends gibt, und ist damit vollstaendig tot. Er wird hier weder
  umgestellt noch repariert — aber er sollte einen eigenen Punkt bekommen,
  weil ein Waechtertest, der Erreichbarkeit prueft, ueber ihn stolpern wird.
* **Die doppelte Auspackung von `DisplayWrapper`.** Manche Weichen fragen den
  Wrapper, manche das innere Objekt, teils im selben `if/elif`. Beruehrt
  diese Arbeit, wird aber nicht als eigenes Ziel aufgeraeumt.
* **Die 3D-Anzeige um Methoden erweitern.** Vom Nutzer verworfen, Begruendung
  unter Implementation Decisions 7.
* **Wege als Textur.** Bleiben Geometrie.
* **Leistung.** Kein Ziel dieser Arbeit. Die Vorgabe ist lediglich, dass das
  Umschalten zwischen 2D und 3D nicht spuerbar langsamer wird als heute.

---

## Further Notes

**Warum das mehr ist als Aufraeumen.** Die Umstellung entfernt keine
Funktion und fuegt keine hinzu; sie aendert, *wer* eine Entscheidung trifft.
Heute trifft sie jeder Reiter einzeln und kann sie stillschweigend falsch
treffen. Danach trifft sie eine Stelle, laut. Das ist der Unterschied
zwischen einer Regel in `CLAUDE.md` und einer Eigenschaft des Programms.

**Die Fehlerklasse ist aelter als die Regel.** Der Bericht fand denselben
Mechanismus ausserhalb der Anzeige wieder: 13 Aufrufstellen, die den
GPU-Rueckfall je selbst nachbauen, davon zwoelf, die einen Fehlschlag nie
protokollieren; und eine Geologie-Operation, die es im Dispatch-Register gar
nicht gibt und deshalb seit jeher still auf CPU laeuft. Dieselbe Form: ein
Ersatzpfad, der von Erfolg nicht zu unterscheiden ist. Wer diese
Spezifikation umsetzt, sollte Kandidat 2 danach lesen — die Loesung hat
dieselbe Gestalt.

**Der Waechtertest hat hier mehr geleistet, als er zugibt.** Er hat den
vierten Fall nicht gefunden, aber seine Ausnahmenliste hat ihn *aufbewahrt*:
die Begruendung fuer `overlay_settlements` zeigt woertlich auf die
unerreichbare Zeile. Ein Test, der seine Ausnahmen begruenden laesst, macht
falsche Annahmen nachlesbar — das ist ein Verdienst und sollte beim Umbau
nicht verlorengehen.

**Es gibt in diesem Projekt keine `CONTEXT.md` und keine ADRs.** Die
Domaenenwoerter dieser Spezifikation — Overlay, Reiter, Anzeige, Schuldliste,
Skalarlayer, Vektor-Overlay — stammen aus dem Code und den vorhandenen
Dokumenten. Es wurde keine ADR-Entscheidung uebergangen, weil es keine gibt.
Eine `CONTEXT.md` anzulegen ist ein eigener, lohnender Punkt: die
Domaenenwoerter liegen heute verstreut ueber neun Dateien in `docs/`.
