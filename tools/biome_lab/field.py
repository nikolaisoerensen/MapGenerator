"""
Path: tools/biome_lab/field.py

Potentialfeld: ein vorberechnetes 2D-Vektorfeld (self.potential_field,
Shape (h, w, 2)), das pro Pixel die Summe aller globalen Kraefte enthaelt
(Civ-Gradient, Stadtgravitation, Wildnis-Huegelpotential, Stadtmauer-
Abstossung, Wildnis-Repulsion, Kartenrand-Abstossung). physics.py sampled
dieses Feld pro Tick bilinear an den aktuellen Node-/Kern-Positionen, statt
die Kraefte jedes Mal einzeln neu zu berechnen -- das haelt den Live-Tick
guenstig, da das Feld selbst nur bei Slider-Aenderungen oder einem Szenen-
Rebuild neu berechnet wird (siehe ui._add_log_slider/_add_slider).
"""
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


class FieldMixin:
    """Potentialfeld-Aufbau und bilineares Sampling. Wird von PlotPhysicsLab
    (app.py) zusammen mit den uebrigen Mixins eingebunden."""

    # ------------------------------------------------------ Potentialfeld --
    def _compute_potential_field(self):
        h, w = self.civ_map.shape
        field = np.zeros((h, w, 2), dtype=np.float64)
        yy, xx = np.mgrid[0:h, 0:w]

        # 1) Civ-Gradient: Saettigung statt harter Normalisierung, damit in
        #    flachen Bereichen kein Rauschen auf volle Staerke aufgeblasen wird.
        gy, gx = np.gradient(self.civ_map)
        grad_norm = np.sqrt(gx ** 2 + gy ** 2)
        GRAD_SATURATION = 0.05
        grad_strength = 1.0 - np.exp(-grad_norm / GRAD_SATURATION)
        mask = grad_norm > 1e-10
        gx_dir = np.zeros_like(gx)
        gy_dir = np.zeros_like(gy)
        gx_dir[mask] = gx[mask] / grad_norm[mask]
        gy_dir[mask] = gy[mask] / grad_norm[mask]
        civ_strength = np.clip(1.0 - self.civ_map, 0.0, 1.0) * 0.5
        field[:, :, 0] += gx_dir * grad_strength * civ_strength
        field[:, :, 1] += gy_dir * grad_strength * civ_strength

        # 2) Gravitation zur naechsten Stadt -- DOMINANTES Feld: hohe Basis-
        # Gewichtung (140 statt vormals 50), damit diese Kraft alle anderen
        # Terme (Civ-Gradient, Wildnis-Repulsion) im Normalfall uebertoent und
        # Nodes selbst tief in der Wildnis zuverlaessig staedtewaerts ziehen.
        if self.settlements:
            eps = self.SOFTENING
            for settlement in self.settlements:
                sx, sy = settlement.x, settlement.y
                dx = xx - sx
                dy = yy - sy
                dist = np.maximum(np.hypot(dx, dy), eps)
                weight = 140.0 / np.sqrt(dist)
                field[:, :, 0] -= (dx / dist) * weight * self.plot_gravity_strength
                field[:, :, 1] -= (dy / dist) * weight * self.plot_gravity_strength

        # 2b) Wildnis-Huegelpotential: sanfter Zug zu lokalen Terrain-Hoehen
        # ausserhalb der Zivilisation. Simuliert die Bevorzugung erhoehter,
        # trockener Lagen fuer Wegpunkte/Siedlungsplaetze in der Wildnis, ohne
        # die dominante Stadtgravitation zu ueberschreiben (kleine Staerke,
        # nur ausserhalb civ_mask aktiv).
        hill_fx, hill_fy = self._compute_wilderness_hill_term(xx, yy)
        field[:, :, 0] += hill_fx
        field[:, :, 1] += hill_fy

        # 3) Stadtmauer-Abstossung, an der Quelle gekappt (Explosion verhindern)
        PUSH_CAP = 3.0
        wall_spacing = 5.0
        for settlement in self.settlements:
            sx, sy = settlement.x, settlement.y
            dx = xx - sx
            dy = yy - sy
            dist = np.maximum(np.hypot(dx, dy), 1e-6)
            ratio = np.minimum(wall_spacing / dist, 6.0)
            push = np.minimum(self.plot_city_repulsion_strength * ratio ** 3, PUSH_CAP)
            field[:, :, 0] += (dx / dist) * push
            field[:, :, 1] += (dy / dist) * push

        # 4) Wildnis-Repulsion ueber signed distance field: wirkt auch INNERHALB
        #    der Wildnis Richtung Zivilisation, nicht nur an der Grenze.
        civ_mask = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        dist_out = distance_transform_edt(~civ_mask)
        dist_in = distance_transform_edt(civ_mask)
        signed_dist = np.where(civ_mask, dist_in, -dist_out)
        gy_s, gx_s = np.gradient(gaussian_filter(signed_dist, sigma=3.0))
        norm_s = np.sqrt(gx_s ** 2 + gy_s ** 2)
        mask_s = norm_s > 1e-10
        wgx = np.zeros_like(gx_s)
        wgy = np.zeros_like(gy_s)
        wgx[mask_s] = gx_s[mask_s] / norm_s[mask_s]
        wgy[mask_s] = gy_s[mask_s] / norm_s[mask_s]
        wild_scale = 25.0
        push_strength = np.where(
            signed_dist < 0,
            1.0 - np.exp(-np.abs(signed_dist) / wild_scale),
            np.exp(-np.maximum(signed_dist, 0) / wild_scale))
        field[:, :, 0] += wgx * push_strength * 0.6
        field[:, :, 1] += wgy * push_strength * 0.6

        # 5) Kartenrand-Abstossung
        BORDER_MARGIN = 25.0
        BORDER_STRENGTH = 0.4

        def _edge_push(dist_to_edge):
            t = np.clip(1.0 - dist_to_edge / BORDER_MARGIN, 0.0, 1.0)
            return t ** 2 * BORDER_STRENGTH

        field[:, :, 0] += _edge_push(xx.astype(float)) - _edge_push((w - 1 - xx).astype(float))
        field[:, :, 1] += _edge_push(yy.astype(float)) - _edge_push((h - 1 - yy).astype(float))

        # 6) Globaler Multiplikator ueber den Log-Regler "Potential-Staerke gesamt"
        field *= self.norm_potential_strength

        self.potential_field = field
        self._mark_overlay_dirty()

    def _compute_wilderness_hill_term(self, xx, yy):
        """Baut ein monoton steigendes Huegel-Zugpotential fuer die Wildnis:
        die Staerke waechst mit dem Abstand zur Civ-Grenze (signed distance),
        saettigt aber ab HILL_SATURATION_DIST, damit Nodes tief in der
        Wildnis nicht staerker gezogen werden als an einem sinnvollen Maximum.
        Richtung folgt weiterhin dem lokalen (geglaetteten) Terrain-Gradienten,
        damit Huegelkuppen als Zielpunkte wirken."""
        civ_mask_hill = self.civ_map >= self.WILDERNESS_CIV_THRESHOLD
        dist_out = distance_transform_edt(~civ_mask_hill)

        HILL_SATURATION_DIST = 40.0
        HILL_MAX_STRENGTH = 0.5
        monotonic_strength = HILL_MAX_STRENGTH * (
                1.0 - np.exp(-dist_out / HILL_SATURATION_DIST))
        monotonic_strength = np.where(civ_mask_hill, 0.0, monotonic_strength)

        hgy, hgx = np.gradient(gaussian_filter(self.heightmap, sigma=4.0))
        hnorm = np.sqrt(hgx ** 2 + hgy ** 2)
        hmask = hnorm > 1e-10
        hgx_dir = np.zeros_like(hgx)
        hgy_dir = np.zeros_like(hgy)
        hgx_dir[hmask] = hgx[hmask] / hnorm[hmask]
        hgy_dir[hmask] = hgy[hmask] / hnorm[hmask]

        return hgx_dir * monotonic_strength, hgy_dir * monotonic_strength

    # -------------------------------------------------------- Feld-Sampling --
    def _sample_field(self, x, y):
        """Bilineares Sampling des Potentialfelds an einer Einzelposition.
        Fuer viele Punkte gleichzeitig: _sample_field_batch (vektorisiert)."""
        if self.potential_field is None:
            return np.zeros(2, dtype=float)

        h, w = self.potential_field.shape[:2]
        x = float(np.clip(x, 0.5, w - 1.5))
        y = float(np.clip(y, 0.5, h - 1.5))

        ix = int(x)
        iy = int(y)
        fx = x - ix
        fy = y - iy

        dx = 1 if ix + 1 < w else 0
        dy = 1 if iy + 1 < h else 0

        return (
                self.potential_field[iy, ix] * (1.0 - fx) * (1.0 - fy)
                + self.potential_field[iy, ix + dx] * fx * (1.0 - fy)
                + self.potential_field[iy + dy, ix] * (1.0 - fx) * fy
                + self.potential_field[iy + dy, ix + dx] * fx * fy
        )

    def _sample_field_batch(self, positions):
        """Vektorisierte Variante von _sample_field fuer viele Punkte gleichzeitig."""
        if self.potential_field is None or len(positions) == 0:
            return np.zeros((len(positions), 2), dtype=float)
        h, w = self.potential_field.shape[:2]
        pos = np.asarray(positions, dtype=float)
        x = np.clip(pos[:, 0], 0.5, w - 1.5)
        y = np.clip(pos[:, 1], 0.5, h - 1.5)
        ix = x.astype(int)
        iy = y.astype(int)
        fx = x - ix
        fy = y - iy
        dx = np.where(ix + 1 < w, 1, 0)
        dy = np.where(iy + 1 < h, 1, 0)
        f00 = self.potential_field[iy, ix]
        f10 = self.potential_field[iy, ix + dx]
        f01 = self.potential_field[iy + dy, ix]
        f11 = self.potential_field[iy + dy, ix + dx]
        w00 = ((1 - fx) * (1 - fy))[:, None]
        w10 = (fx * (1 - fy))[:, None]
        w01 = ((1 - fx) * fy)[:, None]
        w11 = (fx * fy)[:, None]
        return f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11

    def _civ_at(self, pos):
        x, y = int(pos[0]), int(pos[1])
        h, w = self.civ_map.shape
        if 0 <= y < h and 0 <= x < w:
            return float(self.civ_map[y, x])
        return 0.0

    def _civ_at_continuous(self, pos):
        """Civ-Wert ohne Pixel-Spruenge bilinear abfragen.

        Die diskrete Abfrage in ``_civ_at`` ist fuer Zuordnungen ausreichend,
        erzeugt an einer harten Kollisionsgrenze aber ein treppenfoermiges
        Verhalten. Diese Variante wird deshalb fuer die Randkollision benutzt."""
        h, w = self.civ_map.shape

        x = float(np.clip(pos[0], 0.0, w - 1.0))
        y = float(np.clip(pos[1], 0.0, h - 1.0))

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)

        tx = x - x0
        ty = y - y0

        return float(
            self.civ_map[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + self.civ_map[y0, x1] * tx * (1.0 - ty)
            + self.civ_map[y1, x0] * (1.0 - tx) * ty
            + self.civ_map[y1, x1] * tx * ty
        )

    def _civ_at_continuous_batch(self, positions):
        """Vektorisierte Variante von _civ_at_continuous fuer viele Punkte gleichzeitig."""
        h, w = self.civ_map.shape
        pos = np.asarray(positions, dtype=float)
        x = np.clip(pos[:, 0], 0.0, w - 1.0)
        y = np.clip(pos[:, 1], 0.0, h - 1.0)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        tx = x - x0
        ty = y - y0
        return (self.civ_map[y0, x0] * (1 - tx) * (1 - ty)
                + self.civ_map[y0, x1] * tx * (1 - ty)
                + self.civ_map[y1, x0] * (1 - tx) * ty
                + self.civ_map[y1, x1] * tx * ty)
