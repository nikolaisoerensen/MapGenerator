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
#
# Wieso eine Schleife um den claude-Aufruf: `claude -p "..."` ist EIN
# einzelner Durchlauf. Der Agent arbeitet darin zwar von sich aus mehrere
# Tickets ab, entscheidet aber selbst, wann er "fertig genug" ist und schliesst
# dann mit einer Zusammenfassung ab - das kann vor dem Ende der Ticketliste
# passieren. Es gibt keine CLI-Option, die eine feste Anzahl an Arbeitsschritten
# erzwingt. Deshalb wird hier von aussen wiederholt: nach jedem Aufruf wird
# geprueft, ob der Morgenbericht (das einzige verlaessliche "fertig"-Signal aus
# docs/NACHTBETRIEB.md, Schritt 6) neu geschrieben wurde. Wenn nicht, wird die
# GLEICHE Sitzung mit --continue fortgesetzt statt neu zu beginnen - so bleibt
# der Kontext (bereits erledigte Tickets, offener Nachtbranch) erhalten.

$ErrorActionPreference = "Stop"

$projektpfad = "C:\Lokale Dateien\Projects\Python\MapGenerator"
$claudeExe = "C:\Users\soere\.local\bin\claude.exe"

$logVerzeichnis = Join-Path $projektpfad "nachtbetrieb\laufberichte"
New-Item -ItemType Directory -Force -Path $logVerzeichnis | Out-Null
$zeitstempel = Get-Date -Format "yyyy-MM-dd_HHmmss"
$logDatei = Join-Path $logVerzeichnis "taskplaner_$zeitstempel.log"
$morgenberichtDatei = Join-Path $logVerzeichnis "morgenbericht.md"

# Sicherheitsgrenzen der AEUSSEREN Schleife (nicht zu verwechseln mit der
# Zeitgrenze JE TICKET aus tools/nachtlauf.py grenze):
#
# BEFUND 2026-09-22: drei Naechte in Folge (20./21./22.09.) brach die
# Schleife sofort ab, OHNE dass die 7 Stunden Zeitbudget je genutzt wurden.
# Grund: claude.exe meldet ein erschoepftes Nutzungskontingent (Sitzungs-
# oder Wochenlimit) und beendet sich sofort - zwei Durchlaeufe ohne
# Fortschritt waren damit innerhalb von SEKUNDEN erreicht (00:00:02 und
# 00:00:05 in taskplaner_2026-09-22_000002.log), lange bevor ein Reset um
# 00:10 (Sitzungslimit) oder 06:00 Europe/Berlin (Wochenlimit) ueberhaupt
# denkbar war. Der Task startet taeglich um 00:00 Uhr; bis zum Ende des
# Zeitbudgets um 07:00 Uhr waere fuer beide beobachteten Reset-Zeiten genug
# Luft gewesen - sie wurde nur nie abgewartet.
#
# Fix: zwischen Durchlaeufen OHNE Fortschritt eine echte Pause einlegen
# (siehe $verzoegerungOhneFortschrittSekunden) statt sofort erneut zu
# versuchen, und die Anzahl erlaubter Fehlversuche so hoch setzen, dass
# nicht sie, sondern weiterhin $maxDauerStunden die eigentliche Grenze ist.
# Ein echt haengender Agent (kein Limit, sondern ein Bug) ist dadurch nicht
# schutzlos: er kostet im schlimmsten Fall die vollen 7 Stunden statt vorher
# wenigen Sekunden - das ist nachts kein Schaden, nur ungenutzte Zeit.
$maxDurchlaeufe = 30
$maxDauerStunden = 7
$maxOhneFortschritt = 30
$verzoegerungOhneFortschrittSekunden = 900

$erstAufrufPrompt = @'
Du fuehrst heute Nacht den Nachtbetrieb dieses Projekts eigenstaendig durch.
Lies zuerst docs/NACHTBETRIEB.md vollstaendig - dort stehen die verbindlichen
Regeln fuer diesen Lauf. Es ist niemand da, der Rueckfragen beantwortet:
triff die noetigen Entscheidungen selbst und dokumentiere sie ehrlich im
Morgenbericht statt zu warten.

WICHTIG - wann diese Sitzung als fertig gilt: Diese Sitzung wird von einem
aeusseren Skript ggf. mehrfach mit --continue fortgesetzt. Fasse NICHT
zusammen und hoere NICHT auf, solange noch mindestens ein offenes,
bearbeitbares Ticket aus docs/OFFENE_PUNKTE.md uebrig ist ODER Schritt 5
(Code-Review) bzw. Schritt 6 (Morgenbericht) unten noch nicht erledigt sind.
Bearbeite in dieser Sitzung so viele Tickets wie in vernuenftiger Zeit
moeglich, dann schliesse deine Antwort einfach ab (kein "Zusammenfassung und
Ende") - das aeussere Skript prueft danach selbst, ob der Morgenbericht schon
existiert, und setzt sonst fort.

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
   erzeugen. Dies ist der letzte Schritt - erst danach gilt der Lauf als
   fertig.

7. NICHT nach main mergen und NICHT pushen. Das bleibt ein manueller
   Morgenschritt fuer den Nutzer (`python tools/nachtlauf.py stand`,
   dann gezielt mergen oder mit `python tools/nachtlauf.py zuruecknehmen`
   einzelne Tickets verwerfen).

Wenn etwas wirklich nicht automatisiert entscheidbar ist, notiere es ehrlich
im Morgenbericht statt zu raten oder die Sperrliste/NACHTBETRIEB.md-Regeln zu
umgehen.
'@

$weiterPrompt = @'
Mach direkt weiter, ohne Rueckfragen. Falls noch offene, bearbeitbare Tickets
aus docs/OFFENE_PUNKTE.md uebrig sind: das naechste davon bearbeiten (Sperrliste
und Zeitgrenze weiter beachten, ein Commit je Ticket ueber
tools/nachtlauf.py abschliessen). Wenn KEIN Ticket mehr offen/bearbeitbar ist,
aber Schritt 5 (/code-review gegen main) oder Schritt 6
(tools/nachtlauf.py bericht) aus der ersten Anweisung noch nicht erledigt sind:
genau damit weitermachen. Erst wenn tools/nachtlauf.py bericht bereits gelaufen
ist, ist nichts mehr zu tun - dann kurz bestaetigen, dass der Nachtbetrieb
abgeschlossen ist.
'@

Set-Location $projektpfad

$start = Get-Date
$commitVorher = ""
try {
    $commitVorher = (& git rev-parse HEAD 2>$null)
} catch {}
$ohneFortschritt = 0
$fertig = $false

for ($i = 1; $i -le $maxDurchlaeufe; $i++) {
    if (((Get-Date) - $start).TotalHours -ge $maxDauerStunden) {
        "Zeitbudget ($maxDauerStunden h) erschoepft nach $($i - 1) Durchlaeufen - Schleife beendet, ohne dass der Morgenbericht sicher fertig ist." | Add-Content $logDatei
        break
    }

    "=== Durchlauf $i / $maxDurchlaeufe  ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ===" | Add-Content $logDatei

    if ($i -eq 1) {
        & $claudeExe -p $erstAufrufPrompt --permission-mode bypassPermissions *>> $logDatei
    } else {
        & $claudeExe --continue -p $weiterPrompt --permission-mode bypassPermissions *>> $logDatei
    }

    if (Test-Path $morgenberichtDatei) {
        $berichtZeit = (Get-Item $morgenberichtDatei).LastWriteTime
        if ($berichtZeit -ge $start) {
            "Morgenbericht gefunden ($morgenberichtDatei, geschrieben $berichtZeit) - Nachtbetrieb fertig nach $i Durchlauf/Durchlaeufen." | Add-Content $logDatei
            $fertig = $true
            break
        }
    }

    # Fortschritts-Sicherung: kam kein neuer Commit dazu, zaehlt das als ein
    # Durchlauf ohne Fortschritt. Zwei davon in Folge brechen die Schleife ab -
    # sonst wuerde ein haengender Agent das ganze Zeitbudget sinnlos verbrauchen.
    $commitJetzt = ""
    try {
        $commitJetzt = (& git rev-parse HEAD 2>$null)
    } catch {}
    if ($commitJetzt -eq $commitVorher) {
        $ohneFortschritt++
        "Kein neuer Commit seit dem letzten Durchlauf (jetzt $ohneFortschritt von $maxOhneFortschritt ohne Fortschritt)." | Add-Content $logDatei
        if ($ohneFortschritt -ge $maxOhneFortschritt) {
            "Kein Fortschritt in $maxOhneFortschritt aufeinanderfolgenden Durchlaeufen - Schleife abgebrochen, damit sie sich nicht sinnlos wiederholt." | Add-Content $logDatei
            break
        }
        if (((Get-Date) - $start).TotalHours -ge $maxDauerStunden) {
            "Zeitbudget ($maxDauerStunden h) waere durch die Wartepause ueberschritten - Schleife beendet." | Add-Content $logDatei
            break
        }
        "Warte $verzoegerungOhneFortschrittSekunden Sekunden vor dem naechsten Versuch (haeufigster Grund fuer Fortschrittslosigkeit: ein erschoepftes Nutzungskontingent mit bekanntem Reset-Zeitpunkt, nicht ein haengender Agent)." | Add-Content $logDatei
        Start-Sleep -Seconds $verzoegerungOhneFortschrittSekunden
    } else {
        $ohneFortschritt = 0
    }
    $commitVorher = $commitJetzt
}

if (-not $fertig) {
    "Nachtbetrieb NICHT ueber den Morgenbericht abgeschlossen - siehe obige Zeilen fuer den Grund (Zeitbudget, kein Fortschritt, oder maximale Durchlaufzahl erreicht)." | Add-Content $logDatei
}

"Nachtlauf beendet: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $logDatei
