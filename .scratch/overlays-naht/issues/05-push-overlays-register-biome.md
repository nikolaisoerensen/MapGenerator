# 05: Die Naht — Overlays anmelden statt zeichnen, mit dem Biome-Reiter als erstem Nutzer

**What to build:** Ein Reiter sagt, **was** gezeigt werden soll, und weiss
nicht mehr, **wie**. Der Biome-Reiter ist der erste, der so arbeitet, und
seine beiden Overlays erscheinen in beiden Ansichten, ohne dass der Reiter
eine einzige Weiche enthaelt.

Dies ist der Tracer-Bullet des Buendels: ein schmaler, aber vollstaendiger
Pfad durch alle Schichten — Wert, Register, beide Adapter, kopflose Tests,
sichtbares Ergebnis im Programm. Mechanismus und erster Nutzer stecken
bewusst in **einem** Ticket; der Mechanismus allein waere ein waagerechter
Schnitt und am Programm nicht vorfuehrbar.

**Die Naht ist die bestehende, nicht eine neue** (Nutzerentscheidung
2026-09-14). Die Basisklasse der Reiter verschickt heute schon Anzeigedaten
fuer Skalarlayer; sie bekommt ein Geschwister dafuer, das Vektor-Overlays
verschickt. Die Reiter lernen keine zweite Anlaufstelle.

Entwurfsvorlage, vom Nutzer so gewaehlt (haelt die Entscheidung genauer fest
als Prosa — kein fertiger Code):

```
class BaseMapTab:
    def _push_overlays(self, overlays: list[Overlay]) -> None:
        """Wie _push_data_to_current_display, aber fuer Vektor-Overlays.
        Laeuft IMMER gegen beide Anzeigen, nicht gegen current_view."""
```

```
self._push_overlays([
    Overlay("siedlungen", sichtbar=self.cb.isChecked(), daten=(orte, marken)),
    Overlay("fluesse",    sichtbar=self.rivers_cb.isChecked(), daten=gen_karte),
])
```

**Die zentrale Regel:** immer beide Anzeigen bedienen, nie die gerade
sichtbare fragen. Genau so macht es der Skalarpfad heute, mit
ausgeschriebener Begruendung — die 3D-Anzeige existiert je Reiter immer, auch
waehrend 2D sichtbar ist. Wer sie nur bei aktiver 3D-Ansicht fuellt, zeigt
beim Umschalten ein leeres Bild. Das Fragen nach der aktiven Ansicht ist die
Ursache aller vier bisherigen Ausfaelle.

Vollstaendiger Hintergrund: `docs/SPEC_OVERLAYS.md`.

**Blocked by:** 04 (Waechtertest). Ausserdem 01, falls die Vorarbeit
mitgenommen wird — ohne sie zeigt das Register auf die 2D-Anzeige statt auf
ein neutrales Modul, was geht, aber unordentlich ist.

**Status:** ready-for-agent

- [ ] Ein Overlay ist ein Wert mit Name, Sichtbarkeit und Daten — und traegt
      **nicht**, wie gezeichnet wird.
- [ ] `sichtbar=False` ist ein vollwertiger Zustand: er raeumt eine
      liegengebliebene 3D-Textur ab, statt das Overlay nur wegzulassen.
- [ ] Ein Register haelt je Overlay-Namen beide Wege: den 2D-Aufruf und den
      3D-Weg samt Bereichs- und Layernamen.
- [ ] Ein unbekannter Overlay-Name ist ein Fehler mit Namen, keine stille
      Auslassung.
- [ ] Ein angemeldetes Overlay ohne 3D-Weg, das **nicht** in der Schuldliste
      steht, ist ein Fehler beim Start. Steht es drin, laeuft es und schreibt
      eine WARNING je Anlass — nie stillschweigend.
- [ ] Die neue Implementierung fragt an **keiner** Stelle nach der aktiven
      Ansicht. Nachweisbar und nachgewiesen.
- [ ] Der Biome-Reiter enthaelt keine `hasattr`-Weiche auf eine Anzeige mehr.
- [ ] Beide Overlays des Biome-Reiters wirken in beiden Ansichten, Setzen wie
      Abwaehlen, auch ueber einen Ansichtswechsel hinweg.
- [ ] Die Zeichenentscheidung ist **ohne Qt und ohne OpenGL** pruefbar: zwei
      mitschreibende Attrappen-Anzeigen, je Overlay ein Eintrag in beiden
      Protokollen.
- [ ] Tests laufen gegen die echten Kartengroessen (128/256/512/1024), nicht
      gegen ausgedachte — die Lehre aus dem adaptiven Netz.
- [ ] Am laufenden Programm bestaetigt, Eintrag in `docs/PRUEFLISTE_LIVE.md`.
      Dass die Textur **richtig aussieht**, sieht kein Test.
