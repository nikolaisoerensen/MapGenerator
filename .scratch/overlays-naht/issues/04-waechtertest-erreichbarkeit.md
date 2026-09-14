# 04: Waechtertest prueft Erreichbarkeit statt Namensexistenz

**What to build:** Ein Anzeigezweig, der nie ausgefuehrt werden kann, laesst
den Test fehlschlagen.

Der heutige Waechter fragt: *existiert diese Methode auf einer der beiden
Anzeigeklassen?* Das ist die falsche Frage. Alle vier bisherigen Ausfaelle
haetten ihn passiert, weil die Methode jeweils existierte — sie wurde nur nie
gerufen. Die richtige Frage ist: *kann der Zweig, der sie ruft, ueberhaupt
laufen?*

Der Test hat dabei mehr geleistet, als er zugibt: seine Ausnahmenliste hat den
vierten Fall **aufbewahrt**. Die Begruendung fuer `overlay_settlements` zeigt
woertlich auf die unerreichbare Zeile im Biome-Reiter — der Test war gruen
*wegen* des toten Codes. Dass er seine Ausnahmen begruenden laesst, ist ein
Verdienst und bleibt; nur muss eine Begruendung kuenftig auf etwas zeigen, das
laeuft.

Ohne dieses Ticket bewacht waehrend der Umstellung (05 bis 08) niemand das
Ergebnis.

**Blocked by:** 02 (Biome-Reiter), 03 (Uebersichts-Reiter). Beide wuerden den
neuen Test sonst sofort rot faerben — 02 zu Recht, 03 aus einem Grund, der
mit Overlays nichts zu tun hat.

**Status:** ready-for-agent

- [ ] Ein vorzeitiges `return`, das alle 3D-Zweige einer Anzeigemethode
      abschneidet, laesst den Test fehlschlagen.
- [ ] Gegenprobe: der Zustand von **vor** Ticket 02 laesst den neuen Test
      fehlschlagen. Ohne diesen Nachweis ist nicht belegt, dass er die
      Fehlerklasse ueberhaupt sieht.
- [ ] Eine Begruendung in der Ausnahmenliste darf nicht auf eine Reiterzeile
      zeigen, die nie laeuft.
- [ ] Der Test erfasst auch Aufrufe, bei denen `hasattr` und der Methodenname
      auf verschiedenen Zeilen stehen — der heutige zeilenweise Suchausdruck
      uebersieht sie, und seine eigene Begruendung behauptet deshalb faelschlich,
      ein Aufruf sei verschwunden.
- [ ] Der Pruefbereich umfasst nicht nur die Reiter, sondern auch den
      Karteneditor und die Anzeige-Hilfswidgets — dort wird heute ungeprueft
      auf Reiter verteilt.
- [ ] Die Trennung zwischen "einseitig und richtig so" und "einseitig und eine
      Schuld" bleibt erhalten. Die Schuldliste darf weiterhin nur schrumpfen.
- [ ] Bestehende Anzeige-Tests bleiben gruen.
