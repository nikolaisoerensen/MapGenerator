# 01: Rasterfunktionen in ein eigenes Modul (Vorarbeit)

**What to build:** Die fuenf `rasterize_*_rgba()`-Funktionen wohnen nicht
laenger in der 2D-Anzeige. Heute holen sich drei Reiter sie **im 3D-Pfad per
Import aus dem 2D-Modul** — die 3D-Ansicht haengt damit am 2D-Modul, obwohl
sie es sonst nicht braucht. Nach diesem Ticket liegen sie in einem eigenen
Modul, das beide Anzeigen gleichberechtigt benutzen.

Reine Vorarbeit: *make the change easy, then make the easy change.* Es aendert
kein Verhalten und ist am Programm nicht zu sehen. Sein Zweck ist, dass das
Register aus Ticket 05 auf ein neutrales Modul zeigen kann statt auf die
2D-Anzeige.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Die fuenf Rasterfunktionen (Parzellengrenzen, Siedlungen, Regionen,
      Fluesse, Kuestenarchetypen) liegen in einem eigenen Modul.
- [ ] Kein Reiter importiert eine Rasterfunktion mehr aus dem 2D-Anzeigemodul.
- [ ] Die 2D-Anzeige importiert sie ihrerseits aus dem neuen Modul, falls sie
      sie selbst braucht — keine zweite Kopie.
- [ ] `tests/smoke_test_display_2d.py` bleibt gruen, unveraendert.
- [ ] Ein Suchlauf belegt, dass kein Aufrufer auf den alten Ort zeigt.
- [ ] Die Pfadlehre des Projekts ist beachtet: jede `__file__`-Kette im
      verschobenen Code ist nachgerechnet und zeigt noch ins Projekt.
