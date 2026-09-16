"""
Funktionsweise: Haelt die fuenf `rasterize_*_rgba()`-Funktionen (Parzellen-
grenzen, Siedlungen, Regionen, Fluesse, Kuestenarchetypen) plus ihre
gemeinsame Bucketing-Logik und Farbtabellen.
Aufgabe: Neutrales Modul, das MapDisplay2D und MapDisplay3D gleichberechtigt
importieren, ohne dass die 3D-Ansicht dafuer an das 2D-Anzeigemodul haengt
(Ticket #4, Vorarbeit fuer das Overlay-Register aus Ticket #5/`docs/SPEC_OVERLAYS.md`).
"""
import numpy as np
from scipy.ndimage import zoom
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# Farbschema fuer PlotPhysicsSystem-Kerne/-Nodes (siehe
# [[project-settlement-plot-physics-rebuild]]) - geteilt zwischen der Live-
# Konvergenz-Vorschau (draw_plot_physics_snapshot in map_display_2d.py) und
# dem finalen "eingefrorenen" Ergebnis (overlay_plot_boundaries), damit beide
# Ansichten optisch konsistent bleiben.
PLOT_CORE_COLOR_BY_TYPE = {
    "standard_plot_node": "#3498db", "wilderness_core": "#2ecc71", "city_core": "#e74c3c",
}
PLOT_NODE_COLOR_BY_TYPE = {
    "standard_plot_node": "#bdc3c7", "wilderness_node": "#27ae60",
    "map_border_node": "#7f8c8d", "city_border_node": "#c0392b",
}

# Traffic-Gradient fuer path/road-Kanten (hell-orange -> dunkelrot, Nutzer-
# Vorgabe) statt der frueheren 2 Fixfarben je Tier - siehe
# _build_traffic_colored_segments() und [[project-settlement-physics-lab-parity]].
_TRAFFIC_GRADIENT_CMAP = LinearSegmentedColormap.from_list("traffic_avg", ["#ffcc80", "#8b0000"])


def _build_traffic_colored_segments(plot_edges, node_by_id):
    """
    Baut Liniensegmente fuer Voronoi-Kanten - "none"-Kanten (reine Parzellen-
    grenzen, keine Straßen) bleiben dimgray, "path"/"road"-Kanten bekommen
    einen kontinuierlichen Farbverlauf + Liniendicke nach PlotEdge.traffic_avg
    (Lauf-Durchschnitt ueber die gesamte Konvergenz-Simulation, siehe
    PlotPhysicsSystem._classify_road_tiers()) statt der frueheren 2 diskreten
    Tier-Farben. Gemeinsame Basis fuer overlay_plot_boundaries() (2D,
    interaktiv, in map_display_2d.py) und rasterize_plot_boundaries_rgba()
    (3D-Textur-Export) - beide hatten identische, duplizierte Bucketing-Logik.

    Normiert per 95.-Perzentil statt hartem Max (analog zur Potentialfeld-
    Normierung in tools/biome_lab/draw.py), damit ein einzelner Ausreißer-Wert
    nicht den gesamten Verlauf auf ein Extrem zusammenstaucht.

    Return: (none_segments, colored_segments, colors, linewidths) - colors/
    linewidths sind leere Listen, wenn keine path/road-Kanten existieren.
    """
    none_segments = []
    colored_segments = []
    traffic_values = []
    for edge in (plot_edges or {}).values():
        a = node_by_id.get(edge.node_a)
        b = node_by_id.get(edge.node_b)
        if a is None or b is None:
            continue
        seg = (a.node_location, b.node_location)
        if edge.classification in ("road", "path"):
            colored_segments.append(seg)
            traffic_values.append(max(0.0, float(getattr(edge, "traffic_avg", 0.0))))
        else:
            none_segments.append(seg)

    colors, linewidths = [], []
    if traffic_values:
        traffic_arr = np.asarray(traffic_values, dtype=np.float64)
        scale = float(np.percentile(traffic_arr, 95)) if traffic_arr.max() > 0 else 1.0
        normalized = np.clip(traffic_arr / max(scale, 1e-6), 0.0, 1.0)
        colors = [_TRAFFIC_GRADIENT_CMAP(t) for t in normalized]
        linewidths = [1.0 + 2.0 * t for t in normalized]

    return none_segments, colored_segments, colors, linewidths


def rasterize_plot_boundaries_rgba(plot_nodes, plot_edges, plot_cores, wilderness_polygons,
                                    map_size, resolution=512):
    """
    Funktionsweise: Rendert dieselbe Plot-Geometrie wie MapDisplay2D.
    overlay_plot_boundaries() (Kantennetz/Straßen-Tiers/Wildnisgrenzen/Kerne/
    Nodes), aber headless auf eine feste (resolution, resolution, 4)-RGBA-
    Textur statt in ein Qt-Canvas - der transparente Hintergrund (alpha=0)
    laesst das Terrain ueberall durchscheinen, wo nichts gezeichnet wurde.
    Aufgabe: Gemeinsame Rasterisierungs-Basis fuer den 3D-"Skin"-Textur-
    Upload (siehe map_display_3d.py's _render_settlement_plot_skin(),
    [[project-settlement-plot-physics-rebuild]] Teil 4) - "die 2D-Darstellung
    als Skin auf das Terrain legen" (Nutzer-Vorgabe) statt eigener 3D-
    Wireframe-/Marker-Geometrie.
    Parameter: plot_nodes/plot_edges/plot_cores/wilderness_polygons wie
    overlay_plot_boundaries(), map_size (int) - Kartengroesse in Pixeln
    (Koordinatensystem der node_location-Werte), resolution (int) -
    Ziel-Texturaufloesung.
    Return: (resolution, resolution, 4) uint8 RGBA-Array, Zeile 0 = y=0
    (row-index==y-Konvention wie heightmap/civ_map - siehe Docstring-
    Kommentar an der Aufrufstelle in map_display_3d.py).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not plot_nodes:
        return np.zeros((resolution, resolution, 4), dtype=np.uint8)

    dpi = 100
    fig = Figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    node_by_id = {n.node_id: n for n in plot_nodes}
    none_segments, colored_segments, colors, linewidths = _build_traffic_colored_segments(plot_edges, node_by_id)
    if none_segments:
        ax.add_collection(LineCollection(none_segments, colors='dimgray', linewidths=0.7, alpha=0.6, zorder=3))
    if colored_segments:
        ax.add_collection(LineCollection(colored_segments, colors=colors, linewidths=linewidths, alpha=0.9, zorder=3))

    outline_segments = []
    for poly_coords in (wilderness_polygons or []):
        coords = np.asarray(poly_coords, dtype=float)
        if len(coords) < 2:
            continue
        outline_segments.extend((tuple(coords[i]), tuple(coords[i + 1])) for i in range(len(coords) - 1))
    if outline_segments:
        ax.add_collection(LineCollection(outline_segments, colors='magenta', linewidths=1.8, alpha=0.85, zorder=3))

    xs = [n.node_location[0] for n in plot_nodes]
    ys = [n.node_location[1] for n in plot_nodes]
    colors = [PLOT_NODE_COLOR_BY_TYPE.get(n.node_type, "#bdc3c7") for n in plot_nodes]
    ax.scatter(xs, ys, c=colors, marker='.', s=8, alpha=0.7, zorder=4)

    for node_type, color in PLOT_CORE_COLOR_BY_TYPE.items():
        cxs = [c.node_location[0] for c in (plot_cores or []) if c.node_type == node_type]
        cys = [c.node_location[1] for c in (plot_cores or []) if c.node_type == node_type]
        if cxs:
            ax.scatter(cxs, cys, c=color, marker='o', s=45, edgecolors='white', linewidths=0.6, zorder=5)

    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    # buffer_rgba() liefert Zeile 0 = oberer Bildrand (screen-space) = y=map_size;
    # geflippt, damit Zeile 0 = y=0 gilt (row-index==y, wie heightmap/civ_map).
    return np.flipud(buffer)


def rasterize_settlements_rgba(settlement_list, landmark_list, roadsite_list,
                                roads, sea_roads, map_size, resolution=None):
    """
    Funktionsweise: Rendert dieselbe globale Siedlungsuebersicht wie
    MapDisplay2D.overlay_settlements()/overlay_roads() (Staedte/Landmarken/
    Roadsites als Punkte, Land- und Seewege als Linien), aber headless auf
    eine (resolution, resolution, 4)-RGBA-Textur - transparent ueberall dort,
    wo nichts gezeichnet wurde, damit das Terrain durchscheint.
    Aufgabe: 3D-Darstellung des globalen Siedlungsreiters (Nutzer-Vorgabe
    2026-08-13 nach der Sichtpruefung: "3D Settlements global sollte jetzt
    umgesetzt werden"), ueber denselben Alpha-Overlay-Pfad wie Regionen und
    Kuestentypen - KEIN neuer GLSL-Code, keine eigene 3D-Marker-Geometrie.

    Farben, Marker und Linienstile sind bewusst DIESELBEN wie auf der
    2D-Seite (rot/Kreis fuer Staedte, gold/Dreieck fuer Landmarken,
    saddlebrown/Quadrat fuer Roadsites, darkorange fuer Landwege,
    royalblue gestrichelt fuer Seewege) - eine zweite Farbwahl hier waere
    eine zweite Wahrheit, die beim naechsten Umfaerben auseinanderlaufen
    wuerde.

    Marker sind GROESSER als in 2D (s=40 -> s=110): das 2D-Bild wird auf ein
    Canvas von wenigen hundert Pixeln gezeichnet, die 3D-Textur dagegen auf
    map_size (bis 1024) - bei gleicher Punktgroesse waeren die Staedte auf
    dem 3D-Gelaende kaum zu finden.

    Return: (resolution, resolution, 4) uint8 RGBA, Zeile 0 = y=0
    (row-index==y wie heightmap, siehe rasterize_plot_boundaries_rgba()).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if resolution is None:
        resolution = int(map_size)

    hat_inhalt = bool(settlement_list or landmark_list or roadsite_list
                      or roads or sea_roads)
    if not hat_inhalt:
        return np.zeros((resolution, resolution, 4), dtype=np.uint8)

    dpi = 100
    fig = Figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    def _coords(items):
        xs, ys = [], []
        for item in items or []:
            x = getattr(item, 'x', None)
            y = getattr(item, 'y', None)
            if x is None and isinstance(item, (tuple, list)) and len(item) >= 2:
                x, y = item[0], item[1]
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        return xs, ys

    def _wege_zeichnen(pfade, farbe, linestyle, linewidth):
        segmente = []
        for pfad in pfade or []:
            if not pfad or len(pfad) < 2:
                continue
            punkte = [(p[0], p[1]) for p in pfad]
            segmente.extend((punkte[i], punkte[i + 1]) for i in range(len(punkte) - 1))
        if segmente:
            ax.add_collection(LineCollection(
                segmente, colors=farbe, linewidths=linewidth, alpha=0.9,
                linestyles=linestyle, zorder=3))

    _wege_zeichnen(roads, 'darkorange', '-', 3.0)
    _wege_zeichnen(sea_roads, 'royalblue', '--', 2.5)

    rs_x, rs_y = _coords(roadsite_list)
    if rs_x:
        ax.scatter(rs_x, rs_y, c='saddlebrown', marker='s', s=45,
                    edgecolors='black', linewidths=0.8, zorder=4)

    lm_x, lm_y = _coords(landmark_list)
    if lm_x:
        ax.scatter(lm_x, lm_y, c='gold', marker='^', s=85,
                    edgecolors='black', linewidths=1.0, zorder=5)

    st_x, st_y = _coords(settlement_list)
    if st_x:
        ax.scatter(st_x, st_y, c='red', marker='o', s=110,
                    edgecolors='white', linewidths=1.4, zorder=6)

    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    return np.flipud(buffer)


def rasterize_regions_rgba(region_map, heightmap, resolution=None, alpha=0.55, border_alpha=0.9):
    """
    (H,W,4) RGBA-Array: die neun Regionsfarben als Flaechenfuellung (nur auf
    Land) plus WEISSE Grenzlinien zwischen benachbarten Regionen - "Grenzen
    wie in einer Strategiespiel-Provinzkarte, farbige Regionen, weisse
    Trennung" (Nutzer-Vorgabe 2026-08-11, docs/OFFENE_PUNKTE.md 6.1).
    Transparent (alpha=0) ausserhalb von Land und ausserhalb jeder
    Regionsgrenze - laesst Terrain/Basis-Layer darunter durchscheinen, reine
    Zusatzeinfaerbung statt eigenem exklusivem Anzeigemodus (anders als
    MapDisplay2D._render_region_map(), die einen kompletten Modus ersetzt).

    Gemeinsame Rasterisierungsbasis fuer das 2D-Overlay
    (MapDisplay2D.overlay_regions()) und den 3D-Skin-Textur-Upload
    (map_display_3d.py._render_regions_overlay()), analog zu
    rasterize_plot_boundaries_rgba() fuer Plots.

    `alpha` regelt die Deckkraft der Flaechenfuellung - ein SUBTILER Wert
    (Nutzer-Vorgabe: "in subtilen Toenen bei Settlement-Vorschau") fuer die
    Siedlungs-Reiter (wo Staedte/Strassen im Vordergrund stehen sollen), der
    volle Wert im Terrain-Reiter.
    """
    from core.terrain_weltkarte import alle_regionen

    region_map = np.asarray(region_map)
    heightmap = np.asarray(heightmap, dtype=np.float32)
    if region_map.shape != heightmap.shape:
        faktor = heightmap.shape[0] / region_map.shape[0]
        region_map = zoom(region_map.astype(np.float32), faktor, order=0).astype(np.int16)
    if resolution is not None and heightmap.shape[0] != resolution:
        faktor = resolution / heightmap.shape[0]
        region_map = zoom(region_map.astype(np.float32), faktor, order=0).astype(np.int16)
        heightmap = zoom(heightmap, faktor, order=0)

    land = heightmap > 0.0
    rgba = np.zeros(region_map.shape + (4,), dtype=np.uint8)
    for i, (_z, _s, r) in enumerate(alle_regionen()):
        treffer = land & (region_map == i)
        if not treffer.any():
            continue
        rgb = to_rgba(r["farbe"])[:3]
        rgba[treffer, 0] = int(round(rgb[0] * 255))
        rgba[treffer, 1] = int(round(rgb[1] * 255))
        rgba[treffer, 2] = int(round(rgb[2] * 255))
        rgba[treffer, 3] = int(round(alpha * 255))

    grenzen = np.zeros(region_map.shape, dtype=bool)
    grenzen[:, :-1] |= region_map[:, :-1] != region_map[:, 1:]
    grenzen[:-1, :] |= region_map[:-1, :] != region_map[1:, :]
    grenzen &= land
    rgba[grenzen, 0] = 255
    rgba[grenzen, 1] = 255
    rgba[grenzen, 2] = 255
    rgba[grenzen, 3] = int(round(border_alpha * 255))

    return rgba


def rasterize_fluesse_rgba(generation_map, heightmap, zeige_mikro=False,
                           breite_px=1):
    """
    (H,W,4) RGBA-Array des Flussnetzes, nach GENERATION eingefaerbt.

    DIESELBE FARBLOGIK wie `MapDisplay2D.overlay_river_generations()`, aber
    als eigenstaendiges Array statt als Achsen-Zeichnung - damit die
    3D-Ansicht dasselbe Netz in denselben Farben zeigt und nicht eine
    zweite Wahrheit entsteht.

    ANLASS (Nutzerbefund 2026-08-24): *"dass man im 3D modus bei dem
    Flussnetzwerk keine fluesse sehn kann. ich will in den jeweiligen
    reitern die fluesse auf dem boden sehen."* Die 2D-Fassung gibt es seit
    dem 2026-08-06; im 3D fehlte sie ersatzlos. `river_tab` ruft
    `overlay_river_generations()` ueber ein `hasattr` auf - im 3D schlug
    das fehl, und zwar LAUTLOS.

    `generation_map` kommt aus terrain.redistribution/river_generation:
    3 = Makro (die Stroeme), 2 = Meso (Nebenfluesse), 1 = Mikro (Baeche),
    0 = kein Fluss.

    MIKRO BLEIBT NORMALERWEISE WEG - auf einer 21-km-Karte sind das
    Rinnsale von wenigen hundert Metern (Nutzer 2026-08-06: "die kleineren
    fluesse sind nicht zu sehen, zu insignifikant").

    `breite_px` verdickt die Laeufe. Im 3D ist das noetig: ein einzelnes
    Pixel verschwindet auf einer schraeg betrachteten Textur, waehrend die
    2D-Ansicht mit `scatter` ohnehin groessere Marker zeichnet.
    """
    from scipy import ndimage

    leer = np.zeros((1, 1, 4), dtype=np.uint8)
    if not isinstance(generation_map, np.ndarray) or generation_map.ndim != 2:
        return leer
    rgba = np.zeros(generation_map.shape + (4,), dtype=np.uint8)

    # Von FEIN nach GROB, damit ein Strom ueber seinem Nebenfluss liegt.
    stufen = [(1.0, (232, 192, 32))] if zeige_mikro else []
    stufen += [(2.0, (37, 160, 58)), (3.0, (224, 48, 48))]

    for wert, farbe in stufen:
        treffer = generation_map == wert
        if not treffer.any():
            continue
        if breite_px > 0:
            treffer = ndimage.binary_dilation(treffer, iterations=int(breite_px))
        rgba[treffer, 0] = farbe[0]
        rgba[treffer, 1] = farbe[1]
        rgba[treffer, 2] = farbe[2]
        rgba[treffer, 3] = 235

    # NUR AUF LAND. Ein Lauf reicht konstruktionsbedingt bis
    # MUENDUNGSTIEFE_M (-50 m) ins Meer hinein, damit die Muendungsrichtung
    # stimmt; gezeichnet wird nur der Teil ueber Wasser (siehe Modulkopf
    # von core/terrain_weltfluesse.py).
    if isinstance(heightmap, np.ndarray) and heightmap.shape == generation_map.shape:
        rgba[heightmap <= 0.0, 3] = 0
    return rgba


def rasterize_kuesten_archetypen_rgba(region_map, heightmap, archetyp, staerke=None):
    """
    (H,W,4) RGBA-Array der Kuesten-Archetypen - dieselbe Farblogik wie
    MapDisplay2D._render_kuesten_archetypen() (Regionsfarbe x Helligkeit nach
    Steilheit, Deckkraft nach `kuesten_staerke`, weisse Zonengrenzen), aber
    als eigenstaendiges Array statt als Achsen-Zeichnung - fuer den 3D-Skin-
    Textur-Upload (map_display_3d.py, Nutzer-Vorgabe 2026-08-13: "die 3D
    darstellung ALLER 2D maps, aber vor allem der Kuesten auf die 3D Terrains
    bekommen"), analog zu `rasterize_regions_rgba()`.

    Transparent (alpha=0) ausserhalb von Land und ausserhalb jeder Zone -
    laesst die Terrain-Basisfarbe darunter durchscheinen.
    """
    from core.terrain_weltkarte import alle_regionen, KUESTEN_ARCHETYPEN

    region_map = np.asarray(region_map)
    heightmap = np.asarray(heightmap, dtype=np.float32)
    archetyp = np.asarray(archetyp)
    staerke = np.asarray(staerke) if staerke is not None else np.ones_like(heightmap)
    land = heightmap > 0.0

    HOEHE_FAKTOR_MIN, HOEHE_FAKTOR_MAX = 0.2, 2.0
    rgba = np.zeros(region_map.shape + (4,), dtype=np.uint8)
    gebiete = [r for _z, _s, r in alle_regionen()]
    for i, region in enumerate(gebiete):
        archetypen = KUESTEN_ARCHETYPEN.get(region["name"])
        if not archetypen:
            continue
        basis_rgb = np.array(to_rgba(region["farbe"])[:3], dtype=np.float32)
        for lokal_index, typ in enumerate(archetypen):
            treffer = land & (region_map == i) & (archetyp == lokal_index)
            if not treffer.any():
                continue
            norm = np.clip(
                (typ["hoehe_faktor"] - HOEHE_FAKTOR_MIN) / (HOEHE_FAKTOR_MAX - HOEHE_FAKTOR_MIN),
                0.0, 1.0)
            helligkeit = 1.3 - norm * 0.8
            rgb = np.clip(basis_rgb * helligkeit, 0.0, 1.0)
            alpha = 0.25 + 0.55 * np.clip(staerke[treffer], 0.0, 1.0)
            rgba[treffer, 0] = np.round(rgb[0] * 255).astype(np.uint8)
            rgba[treffer, 1] = np.round(rgb[1] * 255).astype(np.uint8)
            rgba[treffer, 2] = np.round(rgb[2] * 255).astype(np.uint8)
            rgba[treffer, 3] = np.round(alpha * 255).astype(np.uint8)

    zonen_id = region_map.astype(np.int32) * 8 + np.where(archetyp >= 0, archetyp, 0)
    grenzen = np.zeros_like(land)
    grenzen[:, :-1] |= (zonen_id[:, :-1] != zonen_id[:, 1:]) & land[:, :-1] & land[:, 1:]
    grenzen[:-1, :] |= (zonen_id[:-1, :] != zonen_id[1:, :]) & land[:-1, :] & land[1:, :]
    rgba[grenzen, 0] = 255
    rgba[grenzen, 1] = 255
    rgba[grenzen, 2] = 255
    rgba[grenzen, 3] = 128

    return rgba
