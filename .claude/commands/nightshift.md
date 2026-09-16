---
description: Startet den Nachtbetrieb (docs/NACHTBETRIEB.md) sofort in dieser Sitzung - Nachtbranch, offene Tickets, Code-Review, Morgenbericht.
---

Fuehre den Nachtbetrieb dieses Projekts durch, genau wie in einem der
automatischen Mitternachtslaeufe (siehe tools/nachtlauf_taskplaner.ps1).
Lies zuerst docs/NACHTBETRIEB.md vollstaendig - dort stehen die verbindlichen
Regeln fuer diesen Lauf.

Ablauf:

1. `python tools/nachtlauf.py starten` ausfuehren. Das legt den Nachtbranch
   nacht/YYYY-MM-DD an und checkt ihn aus. main bleibt dabei unberuehrt.

2. Die offenen Tickets aus docs/OFFENE_PUNKTE.md der Reihe nach mit dem Skill
   /mattpocock-skills:implement abarbeiten (TDD wo sinnvoll, Typecheck und
   Tests je Ticket). Vor jeder Aenderung an einer Datei pruefen, ob sie in
   nachtbetrieb/sperrliste.toml gesperrt ist:
   - `stufe = "sperre"`: NICHT anfassen, Ticket abbrechen und zum naechsten
     gehen (im Bericht vermerken, welches Ticket deswegen ausgelassen wurde).
   - `stufe = "warnung"`: weiterarbeiten, aber die Warnung im Morgenbericht
     vermerken.

3. Nach jedem abgeschlossenen Ticket GENAU EINEN Commit erzeugen mit
   `python tools/nachtlauf.py abschliessen <nummer> "<titel>" --tests "<kurzbefund>"`.
   Keine Handcommits, keine gesammelten Commits ueber mehrere Tickets.

4. Die Zeitgrenze je Ticket beachten (`python tools/nachtlauf.py grenze`).
   Wird sie ueberschritten, sauber abbrechen mit
   `python tools/nachtlauf.py steckengeblieben` und einer vollstaendigen
   Notiz: wo die Arbeit steht, was rot ist (mit der genauen Fehlermeldung),
   was schon versucht wurde, und die naechste Hypothese. Danach zum
   naechsten Ticket weitergehen - nicht an einem Ticket haengen bleiben.

5. Wenn ALLE Tickets aus docs/OFFENE_PUNKTE.md entweder committet, wegen
   Sperre uebersprungen oder als steckengeblieben dokumentiert sind:
   `/code-review` auf dem Nachtbranch gegen main laufen lassen. Kleine,
   sichere Funde selbst beheben und committen. Groessere oder riskante
   Funde nur im Morgenbericht vermerken, nicht selbst anfassen.

6. `python tools/nachtlauf.py bericht` ausfuehren, um den Morgenbericht zu
   erzeugen. Dies ist der letzte Schritt - erst danach ist der Lauf fertig.
   Danach die wichtigsten Punkte aus dem Bericht kurz zusammenfassen.

7. NICHT nach main mergen und NICHT pushen. Das bleibt ein manueller
   Schritt fuer den Nutzer (`python tools/nachtlauf.py stand`, dann gezielt
   mergen oder mit `python tools/nachtlauf.py zuruecknehmen` einzelne
   Tickets verwerfen).

Bearbeite so viele Tickets wie moeglich, ohne zwischendurch anzuhalten oder
eine vorzeitige Zusammenfassung zu geben, solange noch offene, bearbeitbare
Tickets uebrig sind oder Schritt 5/6 noch nicht erledigt sind. Wenn etwas
wirklich nicht automatisiert entscheidbar ist, notiere es ehrlich im
Morgenbericht statt zu raten oder die Sperrliste/NACHTBETRIEB.md-Regeln zu
umgehen.

## AFK-Loop: automatisch weiterlaufen, ohne dass der Nutzer "mach weiter" tippt

Ein einzelner Antwort-Durchlauf entscheidet irgendwann selbst, dass er
"fertig genug" ist, und haelt an - auch wenn Schritt 6 noch nicht erreicht
ist. Damit der Nutzer dafuer nicht am Bildschirm bleiben muss, uebernimm das
Weiterlaufen selbst, ueber ScheduleWakeup, nach genau diesem Muster:

**Zustand mitfuehren** in `nachtbetrieb/laufberichte/nightshift_zustand.json`
(dieser Ordner ist bereits per `.gitignore` von git ausgenommen - reiner
Arbeitsstand, kein Projektinhalt):
- Existiert die Datei nicht: bei Schritt 1 anlegen mit
  `{"start": "<ISO-Zeitstempel jetzt>", "durchlauf": 1, "letzter_commit": "<aktueller git rev-parse HEAD>", "ohne_fortschritt": 0}`.
- Existiert sie: einlesen statt neu zu starten (du fuehrst einen bereits
  laufenden AFK-Loop fort).

**Am Ende JEDER Antwort in diesem Ablauf**, bevor du die Antwort abschliesst:

1. Pruefen, ob `nachtbetrieb/laufberichte/morgenbericht.md` existiert UND
   neuer ist als `start` aus der Zustandsdatei. Wenn ja: fertig - Bericht
   kurz zusammenfassen, Zustandsdatei loeschen, Antwort normal beenden
   (KEIN ScheduleWakeup mehr aufrufen).
2. Sonst mit `git rev-parse HEAD` den aktuellen Commit vergleichen mit
   `letzter_commit` aus der Zustandsdatei:
   - gleich -> `ohne_fortschritt` um 1 erhoehen.
   - unterschiedlich -> `ohne_fortschritt` auf 0 zuruecksetzen,
     `letzter_commit` aktualisieren.
3. Abbruchbedingungen pruefen (alle drei wie beim automatischen
   Mitternachtslauf in `tools/nachtlauf_taskplaner.ps1`): `durchlauf >= 20`,
   oder mehr als 7 Stunden seit `start` vergangen, oder
   `ohne_fortschritt >= 2`. Trifft eine zu: aufhoeren, dem Nutzer ehrlich
   erklaeren warum (Zeitbudget/kein Fortschritt/zu viele Durchlaeufe),
   Zustandsdatei stehen lassen (fuer die Fehlersuche), KEIN ScheduleWakeup
   mehr aufrufen.
4. Sonst: `durchlauf` um 1 erhoehen, Zustandsdatei speichern, und
   `ScheduleWakeup` aufrufen mit `delaySeconds: 60`, `noop: false` (es wurde
   ja gearbeitet), `reason: "Nachtbetrieb laeuft weiter, naechstes Ticket"`
   und als `prompt` einen in sich verstaendlichen Text wie: "Setze den
   Nachtbetrieb fort: lies
   nachtbetrieb/laufberichte/nightshift_zustand.json fuer den bisherigen
   Stand, dann mach mit dem naechsten offenen Ticket aus
   docs/OFFENE_PUNKTE.md weiter bzw. mit Schritt 5/6, falls keine Tickets
   mehr offen sind. Wende dabei weiter die Regeln aus /nightshift an."

So bleibt die Sitzung eigenstaendig aktiv, bis der Morgenbericht steht oder
eine der drei Sicherheitsgrenzen greift - der Nutzer muss dafuer nicht am
Rechner bleiben.

$ARGUMENTS
