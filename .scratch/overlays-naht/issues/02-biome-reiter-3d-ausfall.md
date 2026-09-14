# 02: Biome-Reiter zeigt Siedlungen und Fluesse wieder in 3D

**What to build:** Im Biome-Reiter den Haken "Settlements" oder "Flussnetz"
setzen, auf 3D umschalten — und beides ist da. Heute ist es weg, ohne
Fehlermeldung.

Ursache: `BiomeTab.apply_overlays()` steigt in der ersten Zeile aus, wenn die
Ansicht nicht 2D ist. Die 3D-Zweige darunter wurden am 2026-08-25
ausdruecklich als Behebung eingebaut und **koennen nie ausgefuehrt werden**.
Betroffen sind Siedlungen *und* Flussgenerationen — letztere obwohl
`overlay_river_generations` auf beiden Anzeigen existiert.

Das ist Vorfall 4 derselben Fehlerklasse und der dritte, der als behoben
verbucht wurde, ohne es zu sein. Der Kommentarblock ueber der Stelle
beschreibt eine Behebung, die nie gewirkt hat, und muss richtiggestellt
werden — sonst liest ihn der naechste als Beleg, dass hier alles in Ordnung
ist.

Kleinstes moegliches Ticket, haengt an nichts, am Programm sofort vorfuehrbar.
Es macht den Umbau nicht ueberfluessig: es behebt diesen einen Fall, nicht die
Fehlerklasse.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Haken "Settlements" im Biome-Reiter wirkt in 3D.
- [ ] Haken "Flussnetz" im Biome-Reiter wirkt in 3D.
- [ ] Beide Haken wirken in 2D unveraendert weiter.
- [ ] Ein in 2D gesetzter Haken wirkt nach dem Umschalten auf 3D sofort, ohne
      erneutes Generieren.
- [ ] Abwaehlen entfernt die 3D-Textur, statt sie liegen zu lassen.
- [ ] Der irrefuehrende Kommentarblock ist richtiggestellt: er benennt, dass
      die Behebung vom 2026-08-25 unerreichbar war, und warum.
- [ ] Ein Test deckt genau diesen Ausfall ab — er muss fehlschlagen, wenn der
      vorzeitige Ausstieg zurueckkommt. (Der allgemeine Waechter ist
      Ticket 04; hier genuegt der gezielte Fall.)
- [ ] Am laufenden Programm bestaetigt, mit Eintrag in
      `docs/PRUEFLISTE_LIVE.md`. Headless ist das nicht zu sehen.
