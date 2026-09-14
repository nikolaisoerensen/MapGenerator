# 06: Siedlungs-Reiter auf die Naht umgestellt

**What to build:** Alle Haken des Siedlungs-Reiters — Settlements, Landmarks,
Roadsites, Roads, Regionen — wirken in beiden Ansichten, und der Reiter
enthaelt keine eigene Weiche mehr.

Der Reiter hat heute zwei voneinander unabhaengige Anzeigepfade, die dieselben
Daten getrennt beschaffen und dieselben Haken getrennt abfragen: einen fuer
2D, einen fuer 3D, zusammen rund 185 Zeilen. Sie koennen auseinanderlaufen,
und der 3D-Pfad deckt weniger ab als der 2D-Pfad. Nach diesem Ticket gibt es
einen Pfad.

Der 3D-Weg dieses Reiters gilt bisher als Vorbild fuer alle anderen. Das
bleibt richtig — nur wandert das Vorbild ins Register und wird dadurch
allgemein, statt in jedem Reiter neu abgeschrieben zu werden.

**Sonderfall Wege:** sie bleiben echte Bandgeometrie und werden **nicht** zur
Textur. Sie muessen beim Zoomen scharf und anklickbar bleiben. Das Register
kennt sie als eigenen 3D-Weg, nicht als Rasterweg — das ist der Beleg, dass
das Register mehr kann als eine Rasterfunktion aufzurufen.

**Blocked by:** 05. Nicht von 07 oder 08 — die drei Reiter koennen parallel
laufen.

**Status:** ready-for-agent

- [ ] Settlements, Landmarks, Roadsites, Roads und Regionen wirken je einzeln
      in beiden Ansichten.
- [ ] Mehrere gleichzeitig gesetzte Haken loeschen sich in 3D nicht
      gegenseitig aus.
- [ ] Wege sind in 3D weiterhin Bandgeometrie, scharf beim Zoomen und
      anklickbar — nicht durch eine Textur ersetzt.
- [ ] Der Reiter enthaelt keine `hasattr`-Weiche auf eine Anzeige mehr.
- [ ] Die beiden getrennten Anzeigepfade sind zu einem geworden; die Daten
      werden einmal beschafft, nicht zweimal.
- [ ] Die anklickbaren Objekte der 3D-Ansicht funktionieren weiter.
- [ ] Am laufenden Programm bestaetigt, Eintrag in `docs/PRUEFLISTE_LIVE.md`.
