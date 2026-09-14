# 09: Aufraeumen — alte Weichen weg, Namensraeume ins Register

**What to build:** Nach diesem Ticket gibt es im Programm genau **einen** Ort,
an dem steht, wie ein Overlay in 2D und wie es in 3D gezeichnet wird. Die
Uebergangszeit ist zu Ende: die alte Verdrahtung existiert nicht mehr neben
der neuen.

Der Abschluss-Schritt. Drei Dinge fallen zusammen:

**Die Reste der Weichenlogik.** Was nach 05 bis 08 an
`hasattr`-Anzeigeabfragen uebrigbleibt, verschwindet oder ist begruendet.
Ausgangslage waren 43 in `gui/`.

**Die drei Namensraeume fuer denselben Layer.** Heute gibt es den
2D-Datenschluessel, den kurzen 3D-UI-Namen und die Farbbereichstabelle, an
drei verschiedenen Orten gepflegt. Ein eigener Test existiert nur deshalb,
weil diese handgehaltene Bruecke staendig driftet. Sie zieht ins Register.

**Die registerlosen 3D-Layernamen.** Zeichenketten wie `"uebersicht"` oder
`"wegbaender"` stehen heute blank in den Reitern und muessen zu Dicts in der
3D-Anzeige passen. Ein Tippfehler dort erzeugt **kein** `KeyError`, sondern
ein Nichts — dieselbe Fehlerklasse noch einmal, eine Ebene tiefer. Auch sie
ziehen ins Register.

Dazu die Begruendungen im Waechtertest: sie zeigen danach auf das Register,
nicht mehr auf Zeilen in Reitern. Das ist die Lehre aus dem heutigen Fall —
eine Ausnahme, die sich mit einer Reiterzeile begruendet, kann durch toten
Code wahr *scheinen*.

**Blocked by:** 06, 07, 08 (alle drei Reiterumstellungen).

**Status:** ready-for-agent

- [ ] Kein Reiter fragt eine Anzeige mehr per `hasattr` nach einer
      Overlay-Methode. Verbliebene Abfragen anderer Art sind gezaehlt und
      begruendet.
- [ ] Die Uebersetzung zwischen 2D-Datenschluessel und 3D-Layername steht an
      einer Stelle — dem Register.
- [ ] Die 3D-Layernamen der Vektor-Overlays stehen im Register, nicht blank
      in Reitern.
- [ ] Ein unbekannter Layername erzeugt einen Fehler mit Namen, kein Nichts.
- [ ] Der Paritaetstest der Namensraeume laeuft gegen das Register statt gegen
      die Tabelle in der Reiter-Basisklasse.
- [ ] Jede Begruendung im Waechtertest zeigt auf das Register oder auf eine
      Sacheigenschaft, nie auf eine Zeile in einem Reiter.
- [ ] Die Schuldliste ist gegenueber dem Stand vor 05 nicht gewachsen.
- [ ] `docs/SPEC_OVERLAYS.md` ist als umgesetzt markiert, `docs/SITZUNGSLOG.md`
      und `docs/OFFENE_PUNKTE.md` (Abschnitt 14) sind nachgezogen.
- [ ] Am laufenden Programm bestaetigt: ein Rundgang durch alle vier
      umgestellten Reiter, jeder Haken in beiden Ansichten.
