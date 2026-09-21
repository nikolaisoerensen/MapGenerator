# 07: Regional-Reiter auf die Naht umgestellt — und der Fehlerschlucker raus

**What to build:** Die Overlays des Regional-Reiters — Regionen,
Regionsgitter, Siedlungen, Stadtgrenzkontur, Parzellengrenzen — wirken in
beiden Ansichten, soweit ein 3D-Weg dafuer besteht; wo keiner besteht, sagt
das Programm es laut.

**Der unangenehmste der drei Reiter, aus zwei Gruenden.**

Erstens greifen hier in der 3D-Ansicht **alle fuenf** Weichen nicht — der
Reiter zeichnet seine Overlays gegen die gerade aktive Anzeige, und in 3D
trifft keine einzige zu. Nur die Parzellengrenzen haben einen eigenen
3D-Weg daneben.

Zweitens umschliesst ein `except Exception` mit reiner Debug-Ausgabe den
**gesamten** Overlay-Block. Es hat dort nachweislich schon einmal monatelang
einen echten Fehler verschluckt; die Stelle traegt einen Kommentar darueber.
Solange dieser Schlucker steht, ist jeder weitere Ausfall hier garantiert
unsichtbar. Er faellt mit diesem Ticket weg — das ist die Projektlehre
*"jeder stille Rueckfall braucht eine laute Logzeile"*, angewandt auf die
Stelle, die sie am noetigsten hat.

Drei der fuenf Overlays (Regionsgitter, Stadtgrenzkontur, Parzellengrenzen)
stehen in der Schuldliste. Dieses Ticket **loest sie nicht ein** — es sorgt
dafuer, dass ihr Fehlen angemeldet und laut ist statt still. Ob
Parzellengrenzen in 3D ueberhaupt sinnvoll sind, bleibt offen: Tausende
Parzellen koennten Pixelmatsch werden.

**Blocked by:** 05. Nicht von 06 oder 08 — die drei Reiter koennen parallel
laufen.

**Status:** ready-for-agent

- [ ] Der Reiter enthaelt keine `hasattr`-Weiche auf eine Anzeige mehr.
- [ ] Das `except Exception` um den Overlay-Block ist weg. Was dort kuenftig
      schiefgeht, ist sichtbar.
- [ ] Overlays mit 3D-Weg wirken in beiden Ansichten.
- [ ] Overlays ohne 3D-Weg stehen angemeldet in der Schuldliste und schreiben
      eine WARNING, statt stillschweigend nichts zu tun.
- [ ] Die Schuldliste ist durch dieses Ticket nicht gewachsen.
- [ ] Der Reiter zeichnet weiterhin seine Basiskarte — der bestehende
      Sonderteil des Waechtertests dafuer bleibt gruen.
- [ ] Am laufenden Programm bestaetigt, Eintrag in `docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md`.
      Dieser Reiter hatte am 2026-08-13 schon einmal einen Fehler, den kein
      headless-Test gefunden haette.
