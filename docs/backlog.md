# Backlog

Offene Punkte aus dem Kanban-Board (Stand 2026-07-08, zuletzt aktualisiert
2026-07-08 nach GPU-Session). Diese Datei ist die Quelle der Wahrheit für
Claude, da das Kanban-Plugin selbst nicht auslesbar ist. Bitte beim
Abschließen eines Punktes hier durchstreichen/entfernen statt nur im
Kanban-Board zu verschieben.

## To Do

2. **Flüsse versiegen statt aus der Map zu fließen** — Flüsse, die entstehen,
   sammeln sich aktuell in einem einzelnen Pixel und versiegen dort. Ziel:
   Über die LOD-Stufen hinweg soll der Fluss sich immer einen Weg aus der Map
   heraus suchen und dabei Täler graben. Lösungsansatz z.B. über
   Lakefill-Algorithmen. *(Priority: high)*
   - Update 2026-07-08: beim GPU-Umbau von `water.lake_detection` einen
     direkt verwandten Bug gefunden+gefixt — die Lake-Volumen-Berechnung
     verglich Pixelhöhen gegen die Höhe am Seed (= per Definition der
     tiefste Punkt), wodurch das berechnete Volumen strukturell IMMER ~0
     war und der Volume-Threshold nie griff. Jetzt korrekt (Wasserspiegel =
     höchster Punkt im geflohnenen Becken). Das eigentliche
     "Fluss-gräbt-sich-keinen-Weg-raus"-Verhalten (Lakefill über LOD-Stufen)
     ist davon unberührt und weiterhin offen.

3. **Zu viele Fluss-Biom-Felder** — Es entstehen viel zu viele
   Fluss-Biom-Felder.
   a) Riverbank evtl. auf 2px begrenzen.
   b) Erkennung "was zählt als Fluss" grundsätzlich hinterfragen.
   c) Aktuell entstehen praktisch überall an Hängen Flüsse. *(Priority: high)*

4. **Settlements überarbeiten** — Es sollen echte Städte und Voronoi-Plots
   für die gesamte Karte entstehen, wie im Deskriptor/der Doku beschrieben.
   Vor Umsetzung mit dem User diskutieren. *(Priority: high)*
   - Update 2026-07-08: User hat eigene Änderungsvorschläge für Settlement,
     die zuerst besprochen werden müssen — deshalb auch die GPU-Anbindung
     für Settlement (siehe Ticket #40) bewusst zurückgestellt, um nicht
     an einem Algorithmus zu bauen, der ohnehin überarbeitet wird.

5. **Ladebalken kalibrieren** — Jeder Rechenschritt von Anfang bis Ende muss
   mit Kosten festgelegt sein. Das Ende ist NICHT von Mapsize Max abhängig,
   sondern vom "Final LOD" der gewählten Map Size. Beispiel: Map Size 333 →
   üblicherweise 32→64→128→256→333, also LOD5. Kann im Einzelfall abweichen.
   *(Priority: medium)*

6. **Settlement-Tab-UI aufräumen** — Zwischen System Status und den
   Parameter-Slidern steht aktuell zu viel, was da nicht hingehört. Vorbild:
   Geology-Tab. (Bei Terrain wirkt der System-Status-Bereich zu hoch, als ob
   die Höhe des Felds variabel wäre und noch Platz übrig gewesen wäre — das
   gehört mit repariert.) *(Priority: medium)*

7. **Breite ändert sich beim Klicken auf "Berechnen"** — Beim Klicken ändert
   sich die Breite der Slider (evtl. auch anderer Elemente). Möglicherweise
   ändert sich auch die Panel-Größe. Ursache unklar, muss untersucht werden.
   *(Priority: low)*

39. **Echte GLSL-Simplex-Noise für Terrain-GPU-Pfad schreiben** — Der
    GPU-Noise-Pfad für Terrain ist aktuell bewusst deaktiviert (siehe
    Ticket #37 unten), weil die Platzhalter-Formel kein echtes Noise ist.
    Läuft Stand jetzt als eigene Session im Hintergrund (vom User gestartet).
    *(Priority: low-medium)*

40. **GPU-Anbindung für Settlement** — Aktuell kein einziger
    `shader_manager`-Aufruf im gesamten Settlement-Code, obwohl Settlement
    (Pathfinding, Plot-Painting) der rechenintensivste Generator ist. Bewusst
    zurückgestellt bis Ticket #4 (Settlements überarbeiten) besprochen ist.
    *(Priority: high, aber blockiert)*

41. **GPU-Anbindung für Geology** — Wie Settlement bisher komplett ohne
    GPU-Anbindung, außerhalb des Water/Weather/Biome-Fokus der GPU-Session
    vom 2026-07-08. *(Priority: medium)*

## Erledigt (2026-07-08 — siehe docs/tickets.xlsx für Details)

1. ~~**GPU-Support prüfen**~~ — Vollständig untersucht und größtenteils
   behoben. Kernfund: `GenerationOrchestrator` hat NIE einen `ShaderManager`
   in irgendeinen Generator injiziert — GPU lief bei KEINEM der 6 Generatoren,
   unabhängig vom Shader-Code selbst. Nach dem Fix (dedizierter
   GPUWorker-Thread mit eigenem Offscreen-GL-Kontext): Water (alle 7
   Calculator-Knoten) und Weather (alle 4 Knoten) vollständig GPU-beschleunigt
   und in der echten Pipeline verifiziert (nicht nur isoliert getestet).
   Biome: 3 von 5 Knoten (die anderen 2 waren schon vektorisiert bzw. auf CPU
   nachweislich schneller). Terrain: Shadow-Raycast GPU-beschleunigt, Noise
   bewusst weiter CPU (siehe Ticket #37/#39). Settlement/Geology bleiben
   offen (siehe Ticket #40/#41).

Daneben mehrere kritische, unabhängig davon gefundene Bugs behoben (LOD-
Ceiling-Desync beim Auto-Start, Endlosschleife bei Regenerierung nach
Auto-Start, `impact_matrix`-Tippfehler "size"→"map_size", Pipeline-Status-
Panel jetzt granular über alle 34 Calculator-Knoten) — vollständige Liste
mit Root-Cause-Erklärungen in docs/tickets.xlsx, Nr. 27-38.
