"""
Path: tools/pipeline_kritischer_pfad.py

Was kostet die Ladezeit, und was davon liesse sich durch Parallelitaet
ueberhaupt einsparen?

Der Abhaengigkeitsgraph muss dafuer nicht erst erfasst werden - er steht
seit 2026-07-08 als `CALCULATOR_GRAPH` in managers/calculator_graph.py, mit
`depends_on` je Knoten. Was fehlte, war die VERBINDUNG von Graph und
Messwerten: das Pipeline-Log (2026-08-11) misst jeden Knoten einzeln, aber
niemand hat die Messwerte je gegen die Struktur gehalten.

Genau das macht dieses Werkzeug. Es rechnet drei Zahlen:

  seriell    - die Summe aller Knoten, also was heute passiert
  kritischer Pfad - die laengste Kette von Abhaengigkeiten. DAS ist die
                untere Schranke: schneller geht es mit BELIEBIG vielen
                Kernen nicht, weil diese Knoten aufeinander warten muessen.
  Ebenenzahl - wie viele Knoten gleichzeitig laufen koennten

Und je Knoten den SPIELRAUM (englisch "slack"): wie viel langsamer der
Knoten werden duerfte, ohne die Gesamtzeit zu erhoehen. Ein Knoten mit
grossem Spielraum zu optimieren bringt NICHTS - er wartet ohnehin. Nur
Knoten mit Spielraum 0 liegen auf dem kritischen Pfad.

Aufruf:
    .venv/Scripts/python.exe tools/pipeline_kritischer_pfad.py
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from managers.calculator_graph import CALCULATOR_GRAPH

# --------------------------------------------------------------------- #
# Messwerte aus dem Lauf vom 2026-08-22, 1024 px (LOD 6), Spalte "Dauer".
#
# NUR Rechenzeit je Knoten. Die [assemble]-Zeilen (Ablegen, Signal,
# Anzeige-Refresh) stehen separat unten, weil sie NICHT im Graph stehen und
# nicht parallelisierbar sind - sie laufen im Main-Thread.
MESSUNG_S = {
    "terrain.noise": 0.019,
    "terrain.redistribution": 61.117,
    "terrain.slope": 0.099,
    "terrain.shadow": 2.907,
    "geology.layer_thickness": 2.203,
    "geology.tectonic_displacement": 3.241,
    "geology.outcrop": 0.136,
    "geology.intrusions": 0.109,
    "geology.metamorphic_overprint": 0.020,
    "geology.sediment_overlay": 0.099,
    "geology.hardness": 0.020,
    "geology.rock_color": 0.066,
    "erosion.hydraulic": 0.009,
    "erosion.slope": 0.022,
    "weather.temperature": 13.489,
    "weather.wind": 0.001,
    "weather.humidity": 0.001,
    "weather.precipitation": 0.001,
    "water.lake_detection": 0.429,
    "water.flow_network": 2.264,
    "water.manning_flow": 0.861,
    "water.evaporation": 0.073,
    "water.soil_moisture": 0.105,
    "biome.preseed_hint": 0.071,
    "biome.climate_classification": 0.030,
    "biome.super_override": 1.054,
    "biome.base_classification": 3.644,
    "biome.integrate_layers": 0.002,
    "biome.supersampling": 0.052,
    "settlement.suitability": 0.443,
    "settlement.settlements": 1.582,
    "settlement.city_boundary": 1.332,
    "settlement.pathfinding": 60.283,
    "settlement.roadsites": 4.137,
    "settlement.civ_influence": 1.655,
    "settlement.landmarks": 1.109,
    "settlement.landmark_roads": 0.776,
    "settlement.plot_nodes": 20.248,
}

# Main-Thread-Anteile, nicht parallelisierbar (aus den [assemble]-Zeilen).
HAUPTFADEN_S = {
    "terrain": 5.833 + 0.796,
    "geology": 0.403 + 0.747,
    "erosion": 0.430 + 0.361,
    "weather": 0.019 + 0.702,
    "water": 0.018 + 0.686,
    "biome": 0.219 + 5.974,
    "settlement": 3.116,
}


def _dauer(cid):
    return MESSUNG_S.get(cid, 0.0)


def _kanten():
    """Nur Kanten zwischen Knoten, die es wirklich gibt."""
    vor = {cid: [d for d in spec.depends_on if d in CALCULATOR_GRAPH]
           for cid, spec in CALCULATOR_GRAPH.items()}
    return vor


def analyse():
    vor = _kanten()
    nach = defaultdict(list)
    for cid, deps in vor.items():
        for d in deps:
            nach[d].append(cid)

    # Topologische Ordnung. Rueckkopplungskanten (der Graph hat bewusst
    # einen Kreis, siehe _generators_after_feedback) werden dabei
    # uebersprungen - fuer die Zeitrechnung zaehlt der Vorwaertsdurchlauf.
    grad = {cid: len(vor[cid]) for cid in CALCULATOR_GRAPH}
    bereit = [c for c, g in grad.items() if g == 0]
    ordnung, ebene = [], {}
    warte = list(bereit)
    for c in warte:
        ebene[c] = 0
    while warte:
        c = warte.pop(0)
        ordnung.append(c)
        for n in nach[c]:
            grad[n] -= 1
            ebene[n] = max(ebene.get(n, 0), ebene[c] + 1)
            if grad[n] == 0:
                warte.append(n)

    im_kreis = [c for c in CALCULATOR_GRAPH if c not in ordnung]

    # Frueheste Start-/Endzeiten (unbegrenzt viele Kerne)
    start, ende = {}, {}
    for c in ordnung:
        start[c] = max((ende[d] for d in vor[c] if d in ende), default=0.0)
        ende[c] = start[c] + _dauer(c)
    gesamt = max(ende.values()) if ende else 0.0

    # Spaeteste Zeiten -> Spielraum
    spaet_ende = {c: gesamt for c in ordnung}
    for c in reversed(ordnung):
        nachf = [n for n in nach[c] if n in spaet_ende]
        if nachf:
            spaet_ende[c] = min(spaet_ende[n] - _dauer(n) for n in nachf)
    spielraum = {c: spaet_ende[c] - ende[c] for c in ordnung}

    seriell = sum(_dauer(c) for c in CALCULATOR_GRAPH)
    haupt = sum(HAUPTFADEN_S.values())

    print("=" * 72)
    print("PIPELINE 1024 px - Messung 2026-08-22")
    print("=" * 72)
    print(f"Knoten im Graph                    {len(CALCULATOR_GRAPH)}")
    print(f"davon gemessen                     {sum(1 for c in CALCULATOR_GRAPH if c in MESSUNG_S)}")
    if im_kreis:
        print(f"in der Rueckkopplungsschleife      {len(im_kreis)}: {im_kreis}")
    print()
    print(f"seriell  (heute)                   {seriell:7.1f} s Rechnen")
    print(f"kritischer Pfad (unendl. Kerne)    {gesamt:7.1f} s   <- UNTERE SCHRANKE")
    print(f"theoretische Ersparnis             {seriell - gesamt:7.1f} s "
          f"({(seriell - gesamt) / seriell:.0%})")
    print()
    print(f"Main-Thread obendrauf              {haupt:7.1f} s (nicht parallelisierbar)")
    print(f"=> bestenfalls erreichbar          {gesamt + haupt:7.1f} s "
          f"= {(gesamt + haupt) / 60:.1f} min")
    print(f"   heute                           {seriell + haupt:7.1f} s "
          f"= {(seriell + haupt) / 60:.1f} min")

    print()
    print("--- maximale Parallelitaet je Ebene ---")
    je_ebene = defaultdict(list)
    for c in ordnung:
        je_ebene[ebene[c]].append(c)
    breit = max(len(v) for v in je_ebene.values())
    print(f"Ebenen {len(je_ebene)}, breiteste Ebene {breit} Knoten")
    print(f"mittlere Breite {len(ordnung) / len(je_ebene):.1f} Knoten")

    print()
    print("--- DER KRITISCHE PFAD (Spielraum 0) ---")
    pfad = sorted((c for c in ordnung if spielraum[c] < 1e-6),
                  key=lambda c: start[c])
    for c in pfad:
        if _dauer(c) >= 0.05:
            print(f"  {start[c]:7.1f} -> {ende[c]:7.1f} s  "
                  f"{_dauer(c):7.3f} s  {c}")
    print(f"  {'':>7}    {'':>7}    {sum(_dauer(c) for c in pfad):7.3f} s  SUMME")

    print()
    print("--- groesster SPIELRAUM (Optimierung bringt hier nichts) ---")
    frei = sorted((c for c in ordnung if _dauer(c) >= 0.4),
                  key=lambda c: -spielraum[c])[:8]
    for c in frei:
        lage = "KRITISCH" if spielraum[c] < 1e-6 else f"{spielraum[c]:.1f} s frei"
        print(f"  {_dauer(c):7.3f} s  {c:<38} {lage}")

    _bild(ordnung, start, ende, spielraum, gesamt)
    return 0


def _bild(ordnung, start, ende, spielraum, gesamt):
    """
    Gantt statt Knotennetz.

    Ein Knotennetz mit 38 Knoten zeigt die Struktur, aber nicht das
    Problem - und das Problem ist ZEIT. Waagerecht die Sekunden, je Balken
    ein Knoten: dann sieht man auf einen Blick, dass vier Balken fast die
    ganze Breite ausmachen und der Rest Staub ist.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sicht = [c for c in ordnung if _dauer(c) >= 0.02]
    sicht.sort(key=lambda c: (start[c], -_dauer(c)))

    fig, ax = plt.subplots(figsize=(13, 9))
    for i, c in enumerate(sicht):
        kritisch = spielraum[c] < 1e-6
        ax.barh(i, _dauer(c), left=start[c], height=0.72,
                color="#c0392b" if kritisch else "#7f8c8d",
                edgecolor="none")
        if not kritisch and spielraum[c] > 0.5:
            ax.barh(i, spielraum[c], left=ende[c], height=0.72,
                    color="#7f8c8d", alpha=0.18, edgecolor="none")
        if _dauer(c) >= 2.0:
            ax.text(ende[c] + 1.5, i, f"{_dauer(c):.0f}s",
                    va="center", fontsize=8,
                    color="#c0392b" if kritisch else "#555")

    ax.set_yticks(range(len(sicht)))
    ax.set_yticklabels(sicht, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Sekunden ab Start (frueheste moegliche Lage)")
    ax.set_xlim(0, gesamt * 1.08)
    ax.axvline(120, color="#2980b9", ls="--", lw=1.6)
    ax.text(120, -1.4, " Ziel 2 min", color="#2980b9", fontsize=10,
            va="bottom", fontweight="bold")
    ax.set_title("Pipeline 1024 px: kritischer Pfad (rot) und Spielraum "
                 "(blass)\n"
                 f"kritischer Pfad {gesamt:.0f} s - "
                 "Parallelitaet zwischen Knoten spart nur 12 s",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    ziel = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "docs", "pipeline_kritischer_pfad.png")
    fig.savefig(ziel, dpi=110)
    print("\nBild: docs/pipeline_kritischer_pfad.png")


if __name__ == "__main__":
    sys.exit(analyse())
