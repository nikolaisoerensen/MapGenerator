# Prüfliste am laufenden Programm — angelegt 2026-09-23

**Diese Datei ist die Gegenseite zur Marke `[!]`.** Jedes Ticket, das gebaut,
aber noch nicht am laufenden Programm gesehen wurde, trägt hier ein, *woran man
erkennt, dass es wirkt*. Ohne diesen Eintrag ist die Marke `[!]` nur ein
Vermerk ohne Auftrag.

Die Regel dazu steht in `CLAUDE.md`, Abschnitt „ein Ticket ist erst fertig,
wenn es geprüft wurde".

Start des Programms:

    .venv/Scripts/python.exe main.py

**Wenn eine Zeile bestätigt ist:** Zeile hier löschen, in
`docs/OFFENE_PUNKTE.md` die Marke von `[!]` auf `[x]` setzen (bzw. das
GitHub-Issue schließen) und das Ergebnis im Ticket-Text vermerken — auch ein
„sieht gut aus" ist ein Messwert.

Die Vorgängerin `docs/PRUEFLISTE_LIVE.md` liegt seit dem 2026-08-27 im Archiv
(`docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md`) und gilt nicht mehr; ihre Punkte
sind teils erledigt, teils überholt, ohne dass das nachgeführt wurde. Diese
Datei fängt deshalb leer an und wächst mit jedem abgeschlossenen Ticket.

---

## Teil A — lokale Tickets aus `docs/OFFENE_PUNKTE.md`

| Ticket | Was ansehen | Was richtig ist | Was schiefgehen kann |
|---|---|---|---|
| **5.10** Plot-Physik vereinfacht | Reiter *Siedlungen Regional*, eine Stadt mit Parzellen erzeugen | Parzellen liegen ruhig und gleichmäßig verteilt; keine zuckenden oder ineinander gefahrenen Knoten | Die drei abgeschalteten Federn (`enable_core_plotnode_spring`, `enable_plotnode_plotnode_spring`, `enable_pressure`) fehlen jetzt — wenn die Parzellen dadurch auseinanderfallen statt sich zu beruhigen, war die Abschaltung zu grob |
| **6.1** Regionenansicht | Reiter *Siedlungen Global*, Haken „Regionen" einschalten — **in 2D und in 3D** | Farbige Regionsflächen mit **weißen** Trennlinien, dezent (alpha 0.25), Städte und Wege bleiben davor sichtbar | Im 3D wird der Skin nur bei `renderMode==5` mit Alpha gemischt; sitzt er falsch, färbt er auch Wasserflächen ein. **Noch nicht gebaut:** derselbe Haken in *Siedlungen Regional* |
| **6.16** Adaptives 3D-Netz | Ein 1024er Gelände in 3D drehen, besonders an Klippen und Küsten | Gelände sieht **genauso aus wie vorher** — keine Risse, keine Löcher, keine flimmernden Kanten zwischen groben und feinen Dreiecken | Die erste Zusage („sieht ok aus") stammt vom 2026-08-12 und damit vom **alten** Gitter — die Vorbedingung war da noch falsch, das adaptive Netz lief gar nicht. Diese Prüfung ist die erste am echten neuen Netz |
| **6.21** Overlay-Textur-Cache | Kamera im 3D **dauerhaft drehen**, während *Küstentyp* und danach *Slope* angezeigt wird | Flüssiges Drehen ohne Ruckeln | Vorher wurden pro Bild 0.6 s gerechnet. Greift der Identitäts-Cache nicht, ruckelt es unverändert; greift er zu stark, ändert sich das Bild nach einer Neugenerierung **nicht** — beides ansehen |

## Teil B — GitHub-Tickets

Alle sechs sind die Umstellung der Reiter auf die Naht (Overlays anmelden statt
selbst zeichnen). Sie wurden am 2026-09-17 gebaut und geschlossen, **ohne dass
je jemand hingesehen hat**; am 2026-09-23 wieder geöffnet.

| Ticket | Was ansehen | Was richtig ist |
|---|---|---|
| **#5** Biome-Reiter in 3D | Reiter *Biome*, 3D-Ansicht | Flüsse **und** Siedlungen sind da — und es ist dasselbe Flusssystem wie in 2D, nicht ein zweites |
| **#8** Die Naht, Biome als erster Nutzer | Reiter *Biome*, jeden Haken einzeln an und aus, 2D und 3D | Jeder Haken schaltet in **beiden** Ansichten dieselbe Schicht |
| **#9** Siedlungs-Reiter | Reiter *Siedlungen Global*, alle Haken | Städte, Landmarken, Land- und Seewege in 2D und 3D, gleiche Farben |
| **#10** Regional-Reiter | Reiter *Siedlungen Regional* | Zeigt überhaupt etwas (5.15 war hier der Fehler); der entfernte Fehlerschlucker heißt: ein Problem erscheint jetzt als Meldung statt als leeres Bild |
| **#11** Fluss-Reiter | Reiter *Flussnetzwerk* | Flussnetz in 2D und 3D deckungsgleich |
| **#12** Aufräumen, alte Weichen weg | Alle umgestellten Reiter einmal durchklicken, **Konsole mitlesen** | Kein `AttributeError`, keine Zeile „nicht anwendbar" oder „Fallback" |
