# 08: Fluss-Reiter auf die Naht umgestellt

**What to build:** Die Flussanzeige des Fluss-Reiters, samt dem Haken
"Baeche (Mikro)", laeuft ueber die Naht statt ueber eine eigene Weiche, und
wirkt in beiden Ansichten wie bisher.

Der leichteste der drei Reiter: sein Overlay existiert bereits auf **beiden**
Anzeigen, er hat also heute schon einen funktionierenden 3D-Weg. Er ist
ausserdem der einzige Reiter im Programm, der bei einer verfehlten Weiche eine
Warnung schreibt statt stillschweigend nichts zu tun — die einzige Stelle, an
der die Fehlerklasse sich selbst gemeldet haette.

Genau deshalb ist er wertvoll: er belegt, dass die Umstellung einen
**funktionierenden** Pfad nicht kaputt macht. Die anderen beiden Reiter
belegen, dass sie kaputte repariert.

Nebenbefund aus der Spezifikation, nicht Teil dieses Tickets, aber hier zu
pruefen und zu protokollieren: die Stufe "Baeche (Mikro)" gibt es im Menue,
aber laut Nutzerbericht nicht im Ergebnis. Falls sich das bestaetigt, gehoert
es als eigener Punkt notiert — es ist ein Datenproblem, kein Anzeigeproblem,
und hat in diesem Buendel nichts zu suchen.

**Blocked by:** 05. Nicht von 06 oder 07 — die drei Reiter koennen parallel
laufen.

**Status:** ready-for-agent

- [ ] Der Reiter enthaelt keine `hasattr`-Weiche auf eine Anzeige mehr.
- [ ] Die Flussanzeige wirkt in beiden Ansichten wie vor der Umstellung.
- [ ] Der Haken "Baeche (Mikro)" verhaelt sich in beiden Ansichten gleich.
- [ ] Abwaehlen raeumt die 3D-Textur ab.
- [ ] Die eigene Warnzeile des Reiters ist entweder ueberfluessig geworden
      (weil das Register die Rolle uebernimmt) oder bewusst behalten — nicht
      kommentarlos entfernt.
- [ ] Die Live-Vorschau (128 px) funktioniert weiter.
- [ ] Protokolliert, ob "Baeche (Mikro)" im Ergebnis tatsaechlich vorkommt.
      Falls nein: als eigener Punkt in `docs/OFFENE_PUNKTE.md`, nicht hier
      behoben.
- [ ] Am laufenden Programm bestaetigt, Eintrag in `docs/archiv/2026-08-27_PRUEFLISTE_LIVE.md`.
