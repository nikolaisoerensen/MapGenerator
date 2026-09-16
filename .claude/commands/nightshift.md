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

$ARGUMENTS
