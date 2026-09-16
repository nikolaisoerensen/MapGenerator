# Path: tools/nachtlauf_taskplaner.ps1
#
# Wird von der Windows-Aufgabenplanung ("Task Scheduler") jede Nacht um
# 00:00 Uhr gestartet. Startet eine nicht-interaktive Claude-Code-Sitzung,
# die den kompletten Nachtbetrieb (siehe docs/NACHTBETRIEB.md) durchfuehrt:
# Nachtbranch anlegen, offene Tickets aus docs/OFFENE_PUNKTE.md abarbeiten,
# Code-Review, Morgenbericht. Merged NICHT nach main - das bleibt ein
# manueller Schritt fuer den Nutzer am Morgen (tools/nachtlauf.py stand).
#
# --permission-mode bypassPermissions ist bewusst gewaehlt: ohne jemanden,
# der nachts einzelne Tool-Aufrufe bestaetigt, wuerde die Sitzung sonst beim
# ersten Bash/git-Befehl haengen bleiben. Das ist eine bewusste Entscheidung
# des Nutzers (2026-09-16), keine Standardeinstellung.

$ErrorActionPreference = "Stop"

$projektpfad = "C:\Lokale Dateien\Projects\Python\MapGenerator"
$claudeExe = "C:\Users\soere\.local\bin\claude.exe"

$logVerzeichnis = Join-Path $projektpfad "nachtbetrieb\laufberichte"
New-Item -ItemType Directory -Force -Path $logVerzeichnis | Out-Null
$zeitstempel = Get-Date -Format "yyyy-MM-dd_HHmmss"
$logDatei = Join-Path $logVerzeichnis "taskplaner_$zeitstempel.log"

$prompt = @'
Du fuehrst heute Nacht den Nachtbetrieb dieses Projekts eigenstaendig durch.
Lies zuerst docs/NACHTBETRIEB.md vollstaendig - dort stehen die verbindlichen
Regeln fuer diesen Lauf. Es ist niemand da, der Rueckfragen beantwortet:
triff die noetigen Entscheidungen selbst und dokumentiere sie ehrlich im
Morgenbericht statt zu warten.

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

5. Wenn alle moeglichen Tickets bearbeitet sind (oder keine Zeit/kein
   Ticket mehr sinnvoll bearbeitbar ist): `/code-review` auf dem Nachtbranch
   gegen main laufen lassen. Kleine, sichere Funde selbst beheben und
   committen. Groessere oder riskante Funde nur im Morgenbericht vermerken,
   nicht selbst anfassen.

6. `python tools/nachtlauf.py bericht` ausfuehren, um den Morgenbericht zu
   erzeugen.

7. NICHT nach main mergen und NICHT pushen. Das bleibt ein manueller
   Morgenschritt fuer den Nutzer (`python tools/nachtlauf.py stand`,
   dann gezielt mergen oder mit `python tools/nachtlauf.py zuruecknehmen`
   einzelne Tickets verwerfen).

Arbeite die Nacht durch, ohne auf Antworten zu warten. Wenn etwas wirklich
nicht automatisiert entscheidbar ist, notiere es ehrlich im Morgenbericht
statt zu raten oder die Sperrliste/NACHTBETRIEB.md-Regeln zu umgehen.
'@

Set-Location $projektpfad

& $claudeExe -p $prompt --permission-mode bypassPermissions *> $logDatei

"Nachtlauf beendet: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $logDatei
