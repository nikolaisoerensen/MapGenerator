# 03: Uebersichts-Reiter — loeschen oder bauen

**What to build:** Eine Entscheidung ueber den Uebersichts-Reiter, umgesetzt.

Der Reiter ruft vier Anzeigemethoden auf, die es **nirgends im Programm
gibt** — nicht auf der 2D-Anzeige, nicht auf der 3D-Anzeige, nicht auf dem
Wrapper. Er ist damit vollstaendig wirkungslos, und weil jeder Aufruf hinter
einer `hasattr`-Weiche steht, faellt das niemandem auf.

Zwei Wege, beide vertretbar:

* **Loeschen**, wenn die Uebersicht nicht gebraucht wird. Billig, ehrlich.
* **Bauen**, wenn sie gebraucht wird — dann aber ueber die Naht aus
  Ticket 05, nicht als vierte Hand-Verdrahtung.

Gehoert nicht zur Overlay-Arbeit, steht aber davor: der Erreichbarkeitstest
aus Ticket 04 wird ueber diesen Reiter stolpern. Entweder ist er dann weg,
oder er ist begruendet ausgenommen.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Entschieden und im Sitzungslog festgehalten, welcher der beiden Wege
      gegangen wurde und warum.
- [ ] Bei "loeschen": der Reiter ist weg, kein toter Verweis bleibt stehen,
      die Navigation kennt ihn nicht mehr.
- [ ] Bei "bauen": der Reiter zeigt in **beiden** Ansichten etwas, gemaess
      der stehenden Regel aus `CLAUDE.md`.
- [ ] Bei "spaeter": der Reiter ist im Waechtertest namentlich und mit
      Begruendung ausgenommen — nicht stillschweigend uebersprungen.
- [ ] Kein Suchlauf findet danach noch einen Aufruf einer Anzeigemethode, die
      es nirgends gibt.
