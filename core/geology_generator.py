"""
Path: core/geology_generator.py
Date Changed: 01.09.2025

Funktionsweise: Erweiterte geologische Schichten und Gesteinstypen mit DataLODManager-Integration und 3-stufigem Fallback-System
- GeologySystemGenerator koordiniert geologische Simulation mit numerischem LOD-System
- Rock-Type Klassifizierung (sedimentary, metamorphic, igneous) basierend auf geologischer Simulation
- Mass-Conservation-System (R+G+B=255) für Erosions-/Sedimentations-Kompatibilität
- Hardness-Map-Generation für Water-Generator und nachfolgende Systeme

Parameter Input:
- sedimentary_hardness, igneous_hardness, metamorphic_hardness [0-100]
- ridge_warping, bevel_warping, metamorphic_foliation, metamorphic_folding, igneous_flowing [0.0-1.0]

Dependencies (über DataLODManager):
- heightmap_combined (von terrain_generator)
- slopemap (von terrain_generator)

Output:
- GeologyData-Objekt mit rock_map, hardness_map, validity_state und LOD-Metadaten
- DataLODManager-Storage für nachfolgende Generatoren (water, biome, settlement)
"""

import numpy as np
from opensimplex import OpenSimplex
from scipy.ndimage import gaussian_filter, zoom
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

from gui.OldManagers.calculator_graph import CalculatorRoundScheduler


@dataclass
class GeologyData:
    """
    Container für alle Geology-Daten mit Validity-System und Cache-Management
    """
    rock_map: np.ndarray  # 3D array (H,W,3) RGB für Sedimentary/Igneous/Metamorphic
    hardness_map: np.ndarray  # 2D array, Gesteinshärte [0-100]
    lod_level: int
    actual_size: Tuple[int, int]
    validity_state: Dict[str, bool]
    parameter_hash: str
    parameters: Dict[str, Any]
    # 2D array, kumulative Höhenänderung in Metern aus tektonischer Deformation
    # (ridge_warping/bevel_warping/metamorph_foliation+folding/igneous_flowing),
    # akkumuliert über LOD-Durchläufe wie WaterData.erosion_map/sedimentation_map.
    # Fließt additiv in DataLODManager.get_terrain_data_combined() ein.
    height_delta: Optional[np.ndarray] = None

    def is_valid(self) -> bool:
        """Prüft ob alle Geology-Daten gültig sind"""
        return all(self.validity_state.values())

    def invalidate(self):
        """Invalidiert alle Geology-Daten"""
        self.validity_state = {key: False for key in self.validity_state.keys()}

    def validate_against_parameters(self, new_parameters: Dict[str, Any]) -> bool:
        """Validiert gegen neue Parameter"""
        critical_params = ['sedimentary_hardness', 'igneous_hardness', 'metamorphic_hardness',
                          'ridge_warping', 'bevel_warping']

        for param in critical_params:
            if abs(self.parameters.get(param, 0) - new_parameters.get(param, 0)) > 0.01:
                return False
        return True

    def get_validity_summary(self) -> Dict[str, str]:
        """Gibt Validity-Status-Zusammenfassung zurück"""
        return {
            'overall_valid': str(self.is_valid()),
            'mass_conservation': str(self.validity_state.get('mass_conservation', False)),
            'hardness_range': str(self.validity_state.get('hardness_range', False)),
            'rock_distribution': str(self.validity_state.get('rock_distribution', False))
        }


class HardnessCalculator:
    """
    Spezialisierte Klasse für Hardness-Map-Berechnung aus Rock-Map und Hardness-Parametern
    """

    # Stufen-Multiplikatoren für die Härte-Tier-Maske (soft/medium/hard) - siehe
    # RockTypeClassifier.calculate_hardness_tier_mask(). Bewusst gestuft statt
    # kontinuierlich, damit benachbarte "Nieren" sich in klaren Sprüngen statt
    # beliebig fein unterscheiden.
    HARDNESS_TIER_FACTORS = (0.7, 1.0, 1.3)  # soft, medium, hard

    @staticmethod
    def calculate_hardness_map(rock_map: np.ndarray, sedimentary_hardness: float,
                             igneous_hardness: float, metamorphic_hardness: float,
                             heightmap_combined: Optional[np.ndarray] = None,
                             hardness_tier_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Berechnet Hardness-Map aus Rock-Map und Hardness-Parametern

        Args:
            rock_map: RGB array mit Gesteinsverteilung
            sedimentary_hardness: Härte für Sedimentgestein [0-100]
            igneous_hardness: Härte für Eruptivgestein [0-100]
            metamorphic_hardness: Härte für Metamorphgestein [0-100]
            heightmap_combined: Optional - wenn gesetzt, wird eine milde Höhen-Tendenz
                angewendet (siehe unten). Ohne Heightmap bleibt das Verhalten unverändert
                (reine Gesteinstyp-Gewichtung).
            hardness_tier_mask: Optional - int-array mit Werten 0/1/2 (soft/medium/hard)
                aus RockTypeClassifier.calculate_hardness_tier_mask(). Ergibt zusammen mit
                den 3 Gesteinstypen 9 Härte-Kombinationen (z.B. igneous soft/medium/hard)
                in großen, gestuften Flächen statt einer stufenlosen Gesteinstyp-Härte.

        Returns:
            2D array mit gewichteten Härte-Werten
        """
        # Härte-Array für Gewichtung
        hardness_values = np.array([sedimentary_hardness, igneous_hardness, metamorphic_hardness])

        # Vectorized calculation für Performance
        rock_ratios = rock_map.astype(np.float32) / 255.0
        hardness_map = np.sum(rock_ratios * hardness_values.reshape(1, 1, 3), axis=2)

        # 9-Stufen-System: Härte-Tier-Maske multipliziert die Gesteinstyp-Härte gestuft
        if hardness_tier_mask is not None:
            tier_factors = np.array(HardnessCalculator.HARDNESS_TIER_FACTORS, dtype=np.float32)
            hardness_map = hardness_map * tier_factors[hardness_tier_mask]

        # Milde Höhen-Tendenz: Gestein in tieferen Lagen (näher an Meereshöhe) etwas
        # weicher, höher gelegenes Gestein unverändert - gedeckelt auf max. 15%
        # Reduktion, keine dominante Regel sondern ein sanfter Trend on top der
        # eigentlichen Gesteinstyp-Härte.
        if heightmap_combined is not None:
            h_min = float(np.min(heightmap_combined))
            h_max = float(np.max(heightmap_combined))
            h_range = h_max - h_min if h_max > h_min else 1.0
            norm_height = np.clip((heightmap_combined - h_min) / h_range, 0.0, 1.0)
            elevation_factor = 0.85 + 0.15 * norm_height
            hardness_map = hardness_map * elevation_factor

        return np.clip(hardness_map, 0.0, 100.0)

    @staticmethod
    def validate_hardness_ranges(hardness_map: np.ndarray) -> bool:
        """
        Validiert Hardness-Map auf gültige Werte-Bereiche

        Args:
            hardness_map: Zu validierende Hardness-Map

        Returns:
            True wenn alle Werte in [0-100] liegen
        """
        return np.all((hardness_map >= 0) & (hardness_map <= 100)) and not np.any(np.isnan(hardness_map))


class TectonicDeformationProcessor:
    """
    Spezialisierte Klasse für tektonische Verformung auf geologische Verteilung
    """

    def __init__(self, map_seed: int = 42):
        """
        Initialisiert Deformation-Processor mit separaten Noise-Generatoren

        Args:
            map_seed: Seed für reproduzierbare Deformation-Patterns
        """
        self.ridge_noise = OpenSimplex(seed=map_seed + 3000)
        self.bevel_noise = OpenSimplex(seed=map_seed + 4000)
        self.foliation_noise = OpenSimplex(seed=map_seed + 5000)
        self.flowing_noise = OpenSimplex(seed=map_seed + 6000)

    def apply_ridge_warping(self, rock_map: np.ndarray, ridge_warping: float,
                           height_range: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wendet Ridge-Warping auf geologische Verteilung UND Heightmap an: Ridge-Zonen
        bekommen mehr Igneous/Metamorphic-Anteil UND werden angehoben (echte, sichtbare
        Grate statt nur eine Gesteinsfarben-Verschiebung).

        Args:
            rock_map: Aktuelle Gesteinsverteilung [0.0-1.0]
            ridge_warping: Stärke der Ridge-Verformung [0.0-2.0]
            height_range: Spannweite (max-min) der aktuellen Heightmap, als Skala für
                         die Höhen-Anhebung

        Returns:
            (Verformte Rock-Map, Höhen-Delta in Metern)
        """
        height, width = rock_map.shape[:2]
        height_delta = np.zeros((height, width), dtype=np.float32)

        if ridge_warping <= 0.0:
            return rock_map, height_delta

        norm_x = np.arange(width, dtype=np.float64) / width
        norm_y = np.arange(height, dtype=np.float64) / height
        ridge_field = self.ridge_noise.noise2array(norm_x * 6, norm_y * 6)
        ridge_factor = (ridge_field + 1) * 0.5 * ridge_warping  # (height, width)

        mask = ridge_factor > 0.3  # Nur signifikante Ridge-Bereiche
        deformed_map = rock_map.astype(np.float32).copy()

        current_sed = deformed_map[:, :, 0]
        reduction = np.where(mask, current_sed * ridge_factor * 0.4, 0.0)

        deformed_map[:, :, 0] -= reduction
        deformed_map[:, :, 1] += reduction * 0.6  # Mehr Igneous
        deformed_map[:, :, 2] += reduction * 0.4  # Mehr Metamorphic

        # Ridge-Zonen heben die Landschaft an - bis zu 15% der Heightmap-Spannweite,
        # normiert auf den Bereich oberhalb der Signifikanz-Schwelle 0.3
        normalized_ridge = np.where(mask, (ridge_factor - 0.3) / 0.7, 0.0)
        height_delta = (normalized_ridge * height_range * 0.15).astype(np.float32)

        return deformed_map, height_delta

    def apply_bevel_warping(self, rock_map: np.ndarray, bevel_warping: float,
                           heightmap_combined: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wendet Bevel-Warping auf geologische Verteilung UND Heightmap an: Bevel-Zonen
        bekommen eine ausgeglichenere Gesteinsverteilung UND werden lokal geglättet
        (angeglichen an eine weichgezeichnete Version der Heightmap - "abgeschliffene"
        Kanten statt nur Gesteinsfarben-Mittelung).

        Args:
            rock_map: Aktuelle Gesteinsverteilung [0.0-1.0]
            bevel_warping: Stärke der Bevel-Verformung [0.0-2.0]
            heightmap_combined: Aktuelle Heightmap, als Basis für die lokale Glättung

        Returns:
            (Verformte Rock-Map, Höhen-Delta in Metern)
        """
        height, width = rock_map.shape[:2]
        height_delta = np.zeros((height, width), dtype=np.float32)

        if bevel_warping <= 0.0:
            return rock_map, height_delta

        norm_x = np.arange(width, dtype=np.float64) / width
        norm_y = np.arange(height, dtype=np.float64) / height
        bevel_field = self.bevel_noise.noise2array(norm_x * 8, norm_y * 8)
        bevel_factor = (bevel_field + 1) * 0.5 * bevel_warping  # (height, width)

        mask = bevel_factor > 0.4  # Bevel-Bereiche
        smoothing_factor = np.where(mask, bevel_factor * 0.2, 0.0)

        current_values = rock_map.astype(np.float32)
        # Tendenz zu ausgeglichenerer Verteilung, GLEICHE Skala wie current_values ([0,1])
        target_distribution = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        deformed_map = (current_values * (1 - smoothing_factor[:, :, None]) +
                       target_distribution * smoothing_factor[:, :, None])

        # Bevel-Zonen glätten die Landschaft lokal: Höhe wandert Richtung eines
        # weichgezeichneten Nachbarschafts-Mittels, proportional zur Bevel-Stärke
        smoothed_heightmap = gaussian_filter(heightmap_combined.astype(np.float32), sigma=2.0)
        height_delta = ((smoothed_heightmap - heightmap_combined) * smoothing_factor).astype(np.float32)

        return deformed_map, height_delta

    def process_metamorphic_effects(self, rock_map: np.ndarray, metamorphic_foliation: float,
                                  metamorphic_folding: float, height_range: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Verarbeitet Metamorphic-Foliation und -Folding-Effekte auf Gesteinsverteilung UND
        Heightmap: feine, wellenförmige Schichtungs-Textur (Foliation = lineare Streifen,
        Folding = wellige Falten), nicht als eigenes Gebirge, sondern als Textur-Layer.

        Args:
            rock_map: Aktuelle Gesteinsverteilung [0.0-1.0]
            metamorphic_foliation: Stärke der Foliation [0.0-1.0]
            metamorphic_folding: Stärke der Folding [0.0-1.0]
            height_range: Spannweite (max-min) der aktuellen Heightmap, als Skala für
                         die Höhen-Textur

        Returns:
            (Rock-Map mit Metamorphic-Effekten, Höhen-Delta in Metern)
        """
        height, width = rock_map.shape[:2]
        height_delta = np.zeros((height, width), dtype=np.float32)

        if metamorphic_foliation <= 0.0 and metamorphic_folding <= 0.0:
            return rock_map, height_delta

        norm_x = (np.arange(width, dtype=np.float64) / width)[None, :]
        norm_y = (np.arange(height, dtype=np.float64) / height)[:, None]

        # Foliation - lineare Strukturen, Folding - wellenförmige Strukturen
        foliation_detail = metamorphic_foliation * np.sin(norm_x * 20) * 0.1
        folding_detail = metamorphic_folding * np.cos(norm_y * 15) * 0.1
        foliation_detail, folding_detail = np.broadcast_arrays(foliation_detail, folding_detail)

        total_metamorphic_influence = np.abs(foliation_detail) + np.abs(folding_detail)
        mask = total_metamorphic_influence > 0.05

        # Erhöhe Metamorphic-Anteil in signifikanten Zonen - Skala an das [0,1]-Format
        # von rock_map angeglichen (vorher *30, sprengte den Wertebereich massiv)
        enhancement = np.where(mask, total_metamorphic_influence * 0.6, 0.0)

        enhanced_map = rock_map.astype(np.float32).copy()
        enhanced_map[:, :, 0] -= enhancement * 0.5  # Reduziere Sedimentary
        enhanced_map[:, :, 1] -= enhancement * 0.3  # Reduziere Igneous
        enhanced_map[:, :, 2] += enhancement * 0.8  # Erhöhe Metamorphic

        # Wellenförmige Höhen-Textur (signiert, keine Schwelle - Foliation/Folding
        # sollen als durchgehende feine Schichtung wirken, nicht nur an Extrempunkten)
        height_delta = ((foliation_detail + folding_detail) * height_range * 0.03).astype(np.float32)

        return enhanced_map, height_delta

    def process_igneous_flowing(self, rock_map: np.ndarray, igneous_flowing: float,
                               height_range: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Verarbeitet Igneous-Flowing-Effekte auf Gesteinsverteilung UND Heightmap: feine
        Strömungsmuster-Textur wie erstarrte Lavaflüsse.

        Args:
            rock_map: Aktuelle Gesteinsverteilung [0.0-1.0]
            igneous_flowing: Stärke des Igneous-Flowing [0.0-1.0]
            height_range: Spannweite (max-min) der aktuellen Heightmap, als Skala für
                         die Höhen-Textur

        Returns:
            (Rock-Map mit Igneous-Flowing-Effekten, Höhen-Delta in Metern)
        """
        height, width = rock_map.shape[:2]
        height_delta = np.zeros((height, width), dtype=np.float32)

        if igneous_flowing <= 0.0:
            return rock_map, height_delta

        norm_x = (np.arange(width, dtype=np.float64) / width)[None, :]
        norm_y = (np.arange(height, dtype=np.float64) / height)[:, None]

        # Flow-Pattern - komplexe Strömungsmuster
        flow_pattern = igneous_flowing * np.sin(norm_x * 12) * np.cos(norm_y * 8) * 0.1
        flow_pattern = np.broadcast_to(flow_pattern, (height, width))

        mask = np.abs(flow_pattern) > 0.03
        # Skala an das [0,1]-Format von rock_map angeglichen (vorher *40)
        flow_strength = np.where(mask, np.abs(flow_pattern) * 0.6, 0.0)

        flowing_map = rock_map.astype(np.float32).copy()
        flowing_map[:, :, 0] -= flow_strength * 0.4  # Reduziere Sedimentary
        flowing_map[:, :, 1] += flow_strength * 0.7  # Erhöhe Igneous stark
        flowing_map[:, :, 2] -= flow_strength * 0.3  # Reduziere Metamorphic

        # Strömungsmuster als feine Höhen-Textur (signiert, keine Schwelle)
        height_delta = (flow_pattern * height_range * 0.03).astype(np.float32)

        return flowing_map, height_delta


class RockTypeClassifier:
    """
    Erweiterte Klasse für Gesteinstyp-Klassifizierung mit drei separaten geologischen Zonen
    """

    def __init__(self, map_seed: int = 42):
        """
        Initialisiert Rock-Classifier mit separaten Noise-Generatoren für geologische Zonen

        Args:
            map_seed: Seed für reproduzierbare Gesteinsverteilung
        """
        # Drei unabhängige Simplex-Layers für geologische Zonen
        self.sedimentary_noise = OpenSimplex(seed=map_seed)
        self.metamorphic_noise = OpenSimplex(seed=map_seed + 1000)
        self.igneous_noise = OpenSimplex(seed=map_seed + 2000)

        # Deformation-Noise für Basis-Verteilung
        self.height_distortion_noise = OpenSimplex(seed=map_seed + 7000)

        # Niederfrequentes Noise für die Härte-Stufen-Maske (siehe calculate_hardness_tier_mask)
        # - großräumige, "nierenförmige" Flächen, unabhängig vom Gesteinstyp
        self.hardness_tier_noise = OpenSimplex(seed=map_seed + 9000)

        # Domain-Warp-Noise für gezackte Gesteinsgrenzen (siehe apply_faceted_boundaries)
        self.boundary_warp_noise_x = OpenSimplex(seed=map_seed + 8000)
        self.boundary_warp_noise_y = OpenSimplex(seed=map_seed + 8001)

    def classify_by_elevation(self, heightmap_combined: np.ndarray) -> np.ndarray:
        """
        Erste Klassifizierung basierend auf Höhen mit verzerrter Noise-Funktion

        Args:
            heightmap_combined: Höhendaten (eventuell post-erosion)

        Returns:
            3D array mit Basis-Gesteinsverteilung [0.0-1.0]
        """
        height, width = heightmap_combined.shape
        rock_map = np.zeros((height, width, 3), dtype=np.float32)

        # Höhen normalisieren
        min_height = np.min(heightmap_combined)
        max_height = np.max(heightmap_combined)
        height_range = max_height - min_height if max_height != min_height else 1.0

        for y in range(height):
            for x in range(width):
                # Normalisierte Höhe [0, 1]
                norm_height = (heightmap_combined[y, x] - min_height) / height_range

                # Koordinaten für Noise
                noise_x = x / width * 4.0
                noise_y = y / height * 4.0

                # Verzerrte Noise-Funktion mit Höhe multipliziert
                height_distortion = self.height_distortion_noise.noise2(noise_x * 0.5, noise_y * 0.5) * 0.3
                distorted_height = np.clip(norm_height + height_distortion, 0.0, 1.0)

                # Basis-Verteilung durch verzerrte Höhe
                if distorted_height < 0.3:
                    # Niedrige Bereiche: hauptsächlich sedimentary
                    rock_map[y, x, 0] = 0.7  # Sedimentary (R)
                    rock_map[y, x, 1] = 0.2  # Igneous (G)
                    rock_map[y, x, 2] = 0.1  # Metamorphic (B)
                elif distorted_height < 0.7:
                    # Mittlere Bereiche: gemischt
                    rock_map[y, x, 0] = 0.4  # Sedimentary (R)
                    rock_map[y, x, 1] = 0.3  # Igneous (G)
                    rock_map[y, x, 2] = 0.3  # Metamorphic (B)
                else:
                    # Hohe Bereiche: hauptsächlich igneous/metamorphic
                    rock_map[y, x, 0] = 0.1  # Sedimentary (R)
                    rock_map[y, x, 1] = 0.5  # Igneous (G)
                    rock_map[y, x, 2] = 0.4  # Metamorphic (B)

        return rock_map

    def apply_slope_hardening(self, rock_map: np.ndarray, slopemap: np.ndarray) -> np.ndarray:
        """
        Mischt härtere Gesteine in steile Hänge ein

        Args:
            rock_map: Aktuelle Gesteinsverteilung
            slopemap: Slope-Daten (dz/dx, dz/dy)

        Returns:
            Rock-Map mit Slope-Härtung
        """
        height, width = rock_map.shape[:2]

        for y in range(height):
            for x in range(width):
                # Slope-Magnitude berechnen
                dz_dx = slopemap[y, x, 0]
                dz_dy = slopemap[y, x, 1]
                slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)

                # Normalisierung der Slope (angenommen max slope ~2.0)
                slope_factor = min(1.0, slope_magnitude / 2.0)

                # Bei steilen Hängen: mehr igneous und metamorphic
                if slope_factor > 0.5:
                    hardening_factor = (slope_factor - 0.5) * 2.0  # [0, 1]

                    # Sedimentary reduzieren, igneous/metamorphic erhöhen
                    current_sed = rock_map[y, x, 0]
                    reduction = current_sed * hardening_factor * 0.5

                    rock_map[y, x, 0] -= reduction  # Sedimentary reduzieren
                    rock_map[y, x, 1] += reduction * 0.6  # Igneous erhöhen
                    rock_map[y, x, 2] += reduction * 0.4  # Metamorphic erhöhen

        return rock_map

    def blend_geological_zones(self, rock_map: np.ndarray) -> np.ndarray:
        """
        Addiert/subtrahiert geologische Zonen basierend auf drei unabhängigen Simplex-Funktionen

        Geological-Zones: Drei Simplex-Funktionen für Sedimentary (>0.2), Metamorphic (>0.6), Igneous (>0.8)

        Args:
            rock_map: Aktuelle Gesteinsverteilung

        Returns:
            Rock-Map mit geologischen Zonen
        """
        height, width = rock_map.shape[:2]

        for y in range(height):
            for x in range(width):
                # Normalisierte Koordinaten für Noise
                norm_x = x / width
                norm_y = y / height

                # Drei unabhängige geologische Zonen-Noise [0, 1] normalisiert
                sedimentary_zone = (self.sedimentary_noise.noise2(norm_x * 3, norm_y * 3) + 1) * 0.5
                metamorphic_zone = (self.metamorphic_noise.noise2(norm_x * 2, norm_y * 2) + 1) * 0.5
                igneous_zone = (self.igneous_noise.noise2(norm_x * 4, norm_y * 4) + 1) * 0.5

                # Schwellwerte anwenden und Gewichtungen berechnen
                sedimentary_influence = max(0, sedimentary_zone - 0.2) / 0.8 if sedimentary_zone > 0.2 else 0.0
                metamorphic_influence = max(0, metamorphic_zone - 0.6) / 0.4 if metamorphic_zone > 0.6 else 0.0
                igneous_influence = max(0, igneous_zone - 0.8) / 0.2 if igneous_zone > 0.8 else 0.0

                # Aktuelle Gesteinsverteilung
                current_sed = rock_map[y, x, 0]
                current_ign = rock_map[y, x, 1]
                current_met = rock_map[y, x, 2]

                # Ziel-Verteilungen für geologische Zonen
                target_sed_distribution = np.array([0.8, 0.1, 0.1])  # Sedimentary-dominiert
                target_ign_distribution = np.array([0.1, 0.8, 0.1])  # Igneous-dominiert
                target_met_distribution = np.array([0.1, 0.1, 0.8])  # Metamorphic-dominiert

                # Basis-Gewichtung (behält aktuelle Verteilung bei)
                total_influence = sedimentary_influence * 0.4 + metamorphic_influence * 0.4 + igneous_influence * 0.4
                base_weight = max(0.3, 1.0 - total_influence)  # Mindestens 30% der ursprünglichen Verteilung

                # Gewichtete Mischung der verschiedenen Einflüsse
                current_distribution = np.array([current_sed, current_ign, current_met])

                new_distribution = (current_distribution * base_weight +
                                  target_sed_distribution * sedimentary_influence * 0.4 +
                                  target_ign_distribution * igneous_influence * 0.4 +
                                  target_met_distribution * metamorphic_influence * 0.4)

                # Normalisierung auf Summe = 1.0 (Massenerhaltung)
                total_new = np.sum(new_distribution)
                if total_new > 0:
                    rock_map[y, x, :] = new_distribution / total_new
                else:
                    # Fallback bei numerischen Problemen
                    rock_map[y, x, :] = [0.33, 0.33, 0.34]

        return rock_map

    def apply_faceted_boundaries(self, rock_map: np.ndarray) -> np.ndarray:
        """
        Wandelt die weich geblendete Gesteinsverteilung in scharf abgegrenzte, gezackte
        Flächen um - wie auf einer echten geologischen Karte (Referenzbild vom Nutzer),
        statt eines glatten Farbverlaufs.

        Vorgehen: pro Pixel wird die dominante Gesteinsart bestimmt (statt der bisherigen
        fraktionalen Mischung aus classify_by_elevation()/blend_geological_zones()), und
        die Sampling-Koordinate wird vorher mit zwei überlagerten Noise-Oktaven verzerrt
        (Domain-Warping - eine grobe für große Buchten/Landzungen, eine feine für kleine
        Zacken), damit die Grenzen wie Küstenlinien/Bruchkanten aussehen statt wie glatte
        Höhenlinien. Läuft als letzter Formungsschritt vor der Mass-Conservation, damit
        auch die tektonischen Effekte (Ridge/Bevel/Foliation/Folding/Flowing) mit
        facettiert werden statt nur die Basis-Zonen.

        Args:
            rock_map: Weich geblendete Gesteinsverteilung [0.0-1.0] (H,W,3)

        Returns:
            Facettierte Gesteinsverteilung [0.0-1.0] (H,W,3)
        """
        height, width = rock_map.shape[:2]

        norm_x = np.arange(width, dtype=np.float64) / width
        norm_y = np.arange(height, dtype=np.float64) / height

        # Zwei überlagerte Warp-Oktaven: grob (große Formen) + fein (Zacken)
        warp_x = (self.boundary_warp_noise_x.noise2array(norm_x * 8, norm_y * 8) * 0.03 +
                 self.boundary_warp_noise_x.noise2array(norm_x * 30, norm_y * 30) * 0.015)
        warp_y = (self.boundary_warp_noise_y.noise2array(norm_x * 8, norm_y * 8) * 0.03 +
                 self.boundary_warp_noise_y.noise2array(norm_x * 30, norm_y * 30) * 0.015)

        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        warped_x = np.clip(x_grid + warp_x * width, 0, width - 1).astype(np.int32)
        warped_y = np.clip(y_grid + warp_y * height, 0, height - 1).astype(np.int32)

        warped_rock_map = rock_map[warped_y, warped_x]
        dominant_channel = np.argmax(warped_rock_map, axis=2)  # (H,W), Werte 0/1/2

        # Dominante Gesteinsart stark hervorheben statt fraktional zu mischen - erzeugt
        # satte, klar abgegrenzte Flächen statt eines Verlaufs. Keine reinen 100%-Flächen
        # (sonst wirken Übergänge zwischen Facetten hart-kantig ohne jede Textur).
        dominant_share = 0.85
        remainder_share = (1.0 - dominant_share) / 2.0

        faceted = np.full(rock_map.shape, remainder_share, dtype=np.float32)
        for channel in range(3):
            faceted[:, :, channel] = np.where(dominant_channel == channel, dominant_share, remainder_share)

        return faceted

    def calculate_hardness_tier_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Berechnet eine großräumige, gestufte Härte-Maske (soft/medium/hard) unabhängig
        vom Gesteinstyp - erzeugt zusammen mit den 3 Gesteinstypen die vom Nutzer
        gewünschten 9 Härte-Kombinationen (z.B. igneous soft / igneous medium / igneous
        hard). Niederfrequentes Noise statt der feinen Zonen-Noise aus
        blend_geological_zones(), damit große, "nierenförmige" zusammenhängende
        Flächen entstehen statt kleinteiliger Flecken - und in 3 diskrete Stufen
        geschwellt statt eines glatten Verlaufs, damit benachbarte Härten sich in
        klaren Sprüngen statt beliebig fein unterscheiden.

        Args:
            shape: (height, width) der Ziel-Maske

        Returns:
            2D int-array mit Werten 0 (soft), 1 (medium), 2 (hard)
        """
        height, width = shape

        norm_x = np.arange(width, dtype=np.float64) / width
        norm_y = np.arange(height, dtype=np.float64) / height

        tier_field = (self.hardness_tier_noise.noise2array(norm_x * 2.5, norm_y * 2.5) + 1) * 0.5  # [0,1]

        tier_mask = np.zeros((height, width), dtype=np.int32)
        tier_mask[tier_field >= 1 / 3] = 1
        tier_mask[tier_field >= 2 / 3] = 2

        return tier_mask


class MassConservationManager:
    """
    Erweiterte Klasse für Mass-Conservation mit verbesserter Normalisierung und Validation
    """

    @staticmethod
    def normalize_rock_masses(rock_map: np.ndarray) -> np.ndarray:
        """
        Normalisiert RGB-Werte so dass R+G+B=255 für jeden Pixel

        Args:
            rock_map: RGB-Array mit Gesteinsverteilung [0.0-1.0] oder [0-255]

        Returns:
            Normalisierte RGB-Map mit garantierter Summe 255
        """
        # Sicherstellen dass Input im richtigen Format ist
        if rock_map.dtype == np.float32 and np.max(rock_map) <= 1.0:
            rock_map = (rock_map * 255).astype(np.uint8)
        else:
            rock_map = rock_map.astype(np.uint8)

        height, width, channels = rock_map.shape
        normalized_map = np.zeros_like(rock_map, dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                r, g, b = rock_map[y, x, :]
                total = int(r) + int(g) + int(b)  # int für Präzision

                if total > 0:
                    # Proportionale Normalisierung auf 255
                    norm_r = int((r / total) * 255)
                    norm_g = int((g / total) * 255)
                    norm_b = int((b / total) * 255)

                    # Rundungsfehler korrigieren - Summe muss exakt 255 sein
                    current_sum = norm_r + norm_g + norm_b
                    diff = 255 - current_sum

                    # Differenz auf größten Kanal verteilen
                    if diff != 0:
                        max_channel = np.argmax([norm_r, norm_g, norm_b])
                        if max_channel == 0:
                            norm_r = max(0, min(255, norm_r + diff))
                        elif max_channel == 1:
                            norm_g = max(0, min(255, norm_g + diff))
                        else:
                            norm_b = max(0, min(255, norm_b + diff))

                    normalized_map[y, x, :] = [norm_r, norm_g, norm_b]
                else:
                    # Gleichverteilung bei total=0
                    normalized_map[y, x, :] = [85, 85, 85]  # 85*3 = 255

        return normalized_map

    @staticmethod
    def validate_conservation(rock_map: np.ndarray) -> bool:
        """
        Validiert dass alle Pixel R+G+B=255 erfüllen

        Args:
            rock_map: RGB-Array zum Validieren

        Returns:
            True wenn alle Pixel R+G+B=255 erfüllen
        """
        if rock_map.dtype != np.uint8:
            return False

        sums = np.sum(rock_map.astype(np.int32), axis=2)
        return np.all(sums == 255)

    @staticmethod
    def get_conservation_statistics(rock_map: np.ndarray) -> Dict[str, Any]:
        """
        Gibt detaillierte Statistiken zur Mass-Conservation zurück

        Args:
            rock_map: RGB-Array für Statistiken

        Returns:
            Dictionary mit Conservation-Statistiken
        """
        sums = np.sum(rock_map.astype(np.int32), axis=2)

        return {
            'all_pixels_valid': bool(np.all(sums == 255)),
            'min_sum': int(np.min(sums)),
            'max_sum': int(np.max(sums)),
            'mean_sum': float(np.mean(sums)),
            'pixels_with_errors': int(np.sum(sums != 255)),
            'total_pixels': int(sums.size),
            'conservation_percentage': float(np.sum(sums == 255) / sums.size * 100)
        }


class GeologySystemGenerator:
    """
    Hauptklasse für geologische Schichten und Gesteinstyp-Verteilung mit vollständiger LOD-Integration
    """

    def __init__(self, map_seed: int = 42):
        """
        Initialisiert Geology-System-Generator mit allen Sub-Komponenten

        Args:
            map_seed: Globaler Seed für reproduzierbare Geologie
        """
        self.map_seed = map_seed
        self.logger = logging.getLogger(__name__)

        # Sub-Komponenten initialisieren
        self.rock_classifier = RockTypeClassifier(map_seed)
        self.hardness_calculator = HardnessCalculator()
        self.deformation_processor = TectonicDeformationProcessor(map_seed)
        self.mass_manager = MassConservationManager()

        # Standard-Parameter
        self.default_parameters = self._load_default_parameters()

        # Progress-Callback für UI-Integration
        self.progress_callback = None

    def _load_default_parameters(self) -> Dict[str, Any]:
        """
        Lädt Standard-Parameter aus value_default.py

        Returns:
            Dictionary mit allen Standard-Parametern
        """
        try:
            from gui.config.value_default import GEOLOGY

            return {
                'sedimentary_hardness': GEOLOGY.SEDIMENTARY_HARDNESS["default"],
                'igneous_hardness': GEOLOGY.IGNEOUS_HARDNESS["default"],
                'metamorphic_hardness': GEOLOGY.METAMORPHIC_HARDNESS["default"],
                'ridge_warping': GEOLOGY.RIDGE_WARPING["default"],
                'bevel_warping': GEOLOGY.BEVEL_WARPING["default"],
                'metamorphic_foliation': GEOLOGY.METAMORPH_FOLIATION["default"],
                'metamorphic_folding': GEOLOGY.METAMORPH_FOLDING["default"],
                'igneous_flowing': GEOLOGY.IGNEOUS_FLOWING["default"]
            }
        except ImportError:
            # Fallback-Parameter wenn value_default nicht verfügbar
            self.logger.warning("Could not load parameters from value_default.py, using fallback values")
            return {
                'sedimentary_hardness': 30.0,
                'igneous_hardness': 70.0,
                'metamorphic_hardness': 60.0,
                'ridge_warping': 0.3,
                'bevel_warping': 0.2,
                'metamorphic_foliation': 0.4,
                'metamorphic_folding': 0.3,
                'igneous_flowing': 0.5
            }

    def set_progress_callback(self, callback):
        """
        Setzt Progress-Callback für UI-Updates

        Args:
            callback: Funktion(phase, progress, message)
        """
        self.progress_callback = callback

    def _update_progress(self, phase: str, progress: int, message: str):
        """
        Sendet Progress-Update an UI

        Args:
            phase: Aktuelle Generation-Phase
            progress: Fortschritt in Prozent [0-100]
            message: Status-Message für UI
        """
        if self.progress_callback:
            try:
                self.progress_callback(phase, progress, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def calculate_geology(self, heightmap_combined: np.ndarray, slopemap: np.ndarray,
                         parameters: Dict[str, Any], lod_level: int,
                         previous_height_delta: Optional[np.ndarray] = None) -> GeologyData:
        """
        Hauptmethode für Geology-Generierung mit vollständiger LOD-Integration

        Args:
            heightmap_combined: Post-Erosion Heightmap vom DataLODManager
            slopemap: Slope-Daten (dz/dx, dz/dy) vom Terrain-Generator
            parameters: Parameter-Dictionary mit allen Geology-Einstellungen
            lod_level: Numerisches LOD-Level (1-7+)
            previous_height_delta: Kumuliertes Tektonik-Höhen-Delta aus dem letzten
                abgeschlossenen Geology-LOD (None beim allerersten Lauf) - die tektonische
                Deformation dieser Passage kommt additiv dazu, analog zu
                WaterData.erosion_map/sedimentation_map

        Returns:
            GeologyData-Objekt mit allen Geology-Outputs und Validity-State
        """
        try:
            self.logger.info(f"Starting geology generation - LOD {lod_level}, Size: {heightmap_combined.shape}")

            # Parameter mit Defaults mergen
            merged_params = {**self.default_parameters, **parameters}

            # Input-Validation
            self._validate_inputs(heightmap_combined, slopemap, merged_params)

            # Läuft intern über CalculatorRoundScheduler statt einer flachen Aufruf-
            # Sequenz (siehe gui/OldManagers/calculator_graph.py - Geology-Calculator-
            # Knoten #5-#10 + faceted_boundaries aus docs/generation_pipeline_
            # dependencies.md). Phase des LOD-Lockstep-Umbaus (Tracker #16): nur
            # Geology-intern zerlegt, externes Verhalten (calculate_geology()
            # Signatur/Rückgabe inkl. previous_height_delta-Akkumulation) bleibt
            # unverändert.
            context = {
                "heightmap_combined": heightmap_combined,
                "slopemap": slopemap,
                "parameters": merged_params,
                "lod_level": lod_level,
                "previous_height_delta": previous_height_delta,
            }

            scheduler = CalculatorRoundScheduler(
                calculator_ids=[
                    "geology.classify_elevation", "geology.slope_hardening", "geology.blend_zones",
                    "geology.tectonic_deformation", "geology.faceted_boundaries",
                    "geology.mass_conservation", "geology.hardness",
                ],
                executors={
                    "geology.classify_elevation": self._calc_classify_elevation,
                    "geology.slope_hardening": self._calc_slope_hardening,
                    "geology.blend_zones": self._calc_blend_zones,
                    "geology.tectonic_deformation": self._calc_tectonic_deformation,
                    "geology.faceted_boundaries": self._calc_faceted_boundaries,
                    "geology.mass_conservation": self._calc_mass_conservation,
                    "geology.hardness": self._calc_hardness,
                },
            )
            scheduler.run_round(target_lod=lod_level, context=context)

            # GeologyData-Objekt erstellen
            geology_data = self._create_geology_data(
                context["rock_map_uint8"], context["hardness_map"], lod_level, merged_params,
                context["height_delta"]
            )

            self._update_progress("Generation Complete", 100, "Geology generation completed successfully")
            self.logger.info(f"Geology generation completed - LOD {lod_level}")

            return geology_data

        except Exception as e:
            self.logger.error(f"Geology generation failed: {e}")
            # Fallback zu Default-Rock-Distribution
            return self._create_fallback_geology_data(heightmap_combined, merged_params, lod_level)

    def _calc_classify_elevation(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'geology.classify_elevation' (#5)"""
        self._update_progress("Rock Classification", 20, "Classifying rocks by elevation...")
        context["rock_map"] = self.rock_classifier.classify_by_elevation(context["heightmap_combined"])

    def _calc_slope_hardening(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'geology.slope_hardening' (#6)"""
        self._update_progress("Rock Classification", 35, "Applying slope hardening...")
        context["rock_map"] = self.rock_classifier.apply_slope_hardening(
            context["rock_map"], context["slopemap"])

    def _calc_blend_zones(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'geology.blend_zones' (#7)"""
        self._update_progress("Geological Zones", 50, "Blending geological zones...")
        context["rock_map"] = self.rock_classifier.blend_geological_zones(context["rock_map"])

    def _calc_tectonic_deformation(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'geology.tectonic_deformation' (#8)"""
        self._update_progress("Tectonic Deformation", 65, "Applying tectonic deformation...")
        rock_map, height_delta = self._apply_tectonic_deformation(
            context["rock_map"], context["heightmap_combined"], context["parameters"],
            context["lod_level"], context["previous_height_delta"]
        )
        context["rock_map"] = rock_map
        context["height_delta"] = height_delta

    def _calc_faceted_boundaries(self, context: Dict[str, Any]) -> None:
        """
        Calculator-Node 'geology.faceted_boundaries' - scharfe, gezackte
        Gesteinsgrenzen statt glattem Verlauf (wie auf einer echten geologischen
        Karte); nicht in docs/generation_pipeline_dependencies.md erfasst.
        """
        self._update_progress("Faceting", 72, "Sharpening rock boundaries...")
        context["rock_map"] = self.rock_classifier.apply_faceted_boundaries(context["rock_map"])

    def _calc_mass_conservation(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'geology.mass_conservation' (#9)"""
        self._update_progress("Mass Conservation", 80, "Normalizing rock masses...")
        context["rock_map_uint8"] = self._ensure_mass_conservation(context["rock_map"])

    def _calc_hardness(self, context: Dict[str, Any]) -> None:
        """Calculator-Node 'geology.hardness' (#10)"""
        self._update_progress("Hardness Calculation", 95, "Calculating hardness map...")
        hardness_tier_mask = self.rock_classifier.calculate_hardness_tier_mask(
            context["heightmap_combined"].shape)
        context["hardness_map"] = self.hardness_calculator.calculate_hardness_map(
            context["rock_map_uint8"],
            context["parameters"]['sedimentary_hardness'],
            context["parameters"]['igneous_hardness'],
            context["parameters"]['metamorphic_hardness'],
            context["heightmap_combined"],
            hardness_tier_mask
        )

    def _validate_inputs(self, heightmap_combined: np.ndarray, slopemap: np.ndarray,
                        parameters: Dict[str, Any]):
        """
        Validiert Input-Daten und Parameter

        Args:
            heightmap_combined: Heightmap zum Validieren
            slopemap: Slopemap zum Validieren
            parameters: Parameter zum Validieren

        Raises:
            ValueError: Bei ungültigen Inputs
        """
        # Heightmap-Validation
        if heightmap_combined is None or heightmap_combined.size == 0:
            raise ValueError("Invalid heightmap_combined - empty or None")

        if len(heightmap_combined.shape) != 2:
            raise ValueError("Heightmap must be 2D array")

        if np.any(np.isnan(heightmap_combined)):
            self.logger.warning("Heightmap contains NaN values, will be replaced with zeros")
            heightmap_combined[np.isnan(heightmap_combined)] = 0.0

        # Slopemap-Validation
        if slopemap is None or slopemap.size == 0:
            raise ValueError("Invalid slopemap - empty or None")

        if slopemap.shape[:2] != heightmap_combined.shape:
            raise ValueError("Slopemap and heightmap must have same dimensions")

        if len(slopemap.shape) != 3 or slopemap.shape[2] != 2:
            raise ValueError("Slopemap must be 3D array with 2 channels (dz/dx, dz/dy)")

        # Parameter-Validation
        required_params = ['sedimentary_hardness', 'igneous_hardness', 'metamorphic_hardness']
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")

            value = parameters[param]
            if not (0 <= value <= 100):
                raise ValueError(f"Parameter {param} must be in range [0-100], got {value}")

    def _apply_tectonic_deformation(self, rock_map: np.ndarray, heightmap_combined: np.ndarray,
                                   parameters: Dict[str, Any], lod_level: int,
                                   previous_height_delta: Optional[np.ndarray] = None
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wendet tektonische Deformation mit LOD-spezifischen Details auf Gesteinsverteilung
        UND Heightmap an. Das Höhen-Delta akkumuliert über LOD-Durchläufe (wie
        WaterData.erosion_map/sedimentation_map): previous_height_delta kommt vom letzten
        abgeschlossenen Geology-LOD, die hier neu berechneten Deltas kommen addiert dazu.

        Args:
            rock_map: Aktuelle Gesteinsverteilung
            heightmap_combined: Aktuelle Heightmap (Referenz-Skala + Basis für Bevel-Glättung)
            parameters: Deformation-Parameter
            lod_level: LOD-Level für Detail-Skalierung
            previous_height_delta: Kumuliertes Höhen-Delta aus dem letzten Geology-LOD

        Returns:
            (Deformierte Rock-Map, kumuliertes Höhen-Delta in Metern)
        """
        # LOD-spezifische Deformation-Detail-Skalierung
        lod_detail_factor = min(1.0, lod_level / 5.0)  # Volldetail ab LOD 5

        height_range = float(np.max(heightmap_combined) - np.min(heightmap_combined))
        height_range = height_range if height_range > 0 else 1.0

        height_delta = np.zeros(heightmap_combined.shape, dtype=np.float32)

        # Ridge-Warping
        if parameters.get('ridge_warping', 0.0) > 0.0:
            scaled_ridge = parameters['ridge_warping'] * lod_detail_factor
            rock_map, ridge_delta = self.deformation_processor.apply_ridge_warping(
                rock_map, scaled_ridge, height_range)
            height_delta += ridge_delta

        # Bevel-Warping
        if parameters.get('bevel_warping', 0.0) > 0.0:
            scaled_bevel = parameters['bevel_warping'] * lod_detail_factor
            rock_map, bevel_delta = self.deformation_processor.apply_bevel_warping(
                rock_map, scaled_bevel, heightmap_combined)
            height_delta += bevel_delta

        # Metamorphic-Effekte
        metamorphic_foliation = parameters.get('metamorph_foliation', 0.0) * lod_detail_factor
        metamorphic_folding = parameters.get('metamorph_folding', 0.0) * lod_detail_factor

        if metamorphic_foliation > 0.0 or metamorphic_folding > 0.0:
            rock_map, metamorphic_delta = self.deformation_processor.process_metamorphic_effects(
                rock_map, metamorphic_foliation, metamorphic_folding, height_range
            )
            height_delta += metamorphic_delta

        # Igneous-Flowing
        if parameters.get('igneous_flowing', 0.0) > 0.0:
            scaled_flowing = parameters['igneous_flowing'] * lod_detail_factor
            rock_map, flowing_delta = self.deformation_processor.process_igneous_flowing(
                rock_map, scaled_flowing, height_range)
            height_delta += flowing_delta

        # Kumulation über LOD-Durchläufe (auf Zielgröße gebracht falls nötig)
        if previous_height_delta is not None:
            if previous_height_delta.shape != height_delta.shape:
                previous_height_delta = zoom(
                    previous_height_delta,
                    height_delta.shape[0] / previous_height_delta.shape[0],
                    order=1
                )
            height_delta = height_delta + previous_height_delta

        return rock_map, height_delta.astype(np.float32)

    def _ensure_mass_conservation(self, rock_map: np.ndarray) -> np.ndarray:
        """
        Stellt Mass-Conservation sicher und validiert Ergebnis

        Args:
            rock_map: Gesteinsverteilung [0.0-1.0]

        Returns:
            Normalisierte Rock-Map mit R+G+B=255
        """
        # Zu uint8 konvertieren für Mass-Conservation
        if rock_map.dtype == np.float32:
            rock_map_uint8 = (rock_map * 255).astype(np.uint8)
        else:
            rock_map_uint8 = rock_map.astype(np.uint8)

        # Mass-Conservation anwenden
        normalized_map = self.mass_manager.normalize_rock_masses(rock_map_uint8)

        # Validation
        if not self.mass_manager.validate_conservation(normalized_map):
            self.logger.warning("Mass conservation validation failed - applying correction")
            # Sekundäre Normalisierung
            normalized_map = self._force_mass_conservation(normalized_map)

        return normalized_map

    def _force_mass_conservation(self, rock_map: np.ndarray) -> np.ndarray:
        """
        Forciert Mass-Conservation als Fallback-Mechanismus

        Args:
            rock_map: Problematische Rock-Map

        Returns:
            Korrigierte Rock-Map mit garantierter Mass-Conservation
        """
        height, width = rock_map.shape[:2]
        corrected_map = np.zeros_like(rock_map, dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                pixel_sum = int(np.sum(rock_map[y, x, :]))

                if pixel_sum != 255:
                    if pixel_sum > 0:
                        # Proportionale Korrektur
                        factor = 255.0 / pixel_sum
                        corrected_map[y, x, :] = np.round(rock_map[y, x, :] * factor).astype(np.uint8)

                        # Finale Summen-Korrektur
                        final_sum = int(np.sum(corrected_map[y, x, :]))
                        if final_sum != 255:
                            diff = 255 - final_sum
                            max_channel = np.argmax(corrected_map[y, x, :])
                            corrected_map[y, x, max_channel] += diff
                    else:
                        # Gleichverteilung bei Null-Summe
                        corrected_map[y, x, :] = [85, 85, 85]
                else:
                    corrected_map[y, x, :] = rock_map[y, x, :]

        return corrected_map

    def _create_geology_data(self, rock_map: np.ndarray, hardness_map: np.ndarray,
                           lod_level: int, parameters: Dict[str, Any],
                           height_delta: Optional[np.ndarray] = None) -> GeologyData:
        """
        Erstellt GeologyData-Objekt mit vollständiger Validation

        Args:
            rock_map: Finale Rock-Map
            hardness_map: Finale Hardness-Map
            lod_level: Aktuelles LOD-Level
            parameters: Verwendete Parameter
            height_delta: Kumuliertes Tektonik-Höhen-Delta aus _apply_tectonic_deformation

        Returns:
            Vollständig validiertes GeologyData-Objekt
        """
        # Validity-State berechnen
        validity_state = {
            'mass_conservation': self.mass_manager.validate_conservation(rock_map),
            'hardness_range': self.hardness_calculator.validate_hardness_ranges(hardness_map),
            'rock_distribution': self._validate_rock_distribution(rock_map),
            'no_nan_values': not (np.any(np.isnan(rock_map)) or np.any(np.isnan(hardness_map))),
            'correct_dimensions': rock_map.shape[:2] == hardness_map.shape
        }

        # Parameter-Hash für Cache-Invalidation
        parameter_hash = self._calculate_parameter_hash(parameters)

        return GeologyData(
            rock_map=rock_map,
            hardness_map=hardness_map,
            lod_level=lod_level,
            actual_size=rock_map.shape[:2],
            validity_state=validity_state,
            parameter_hash=parameter_hash,
            parameters=parameters.copy(),
            height_delta=height_delta
        )

    def _validate_rock_distribution(self, rock_map: np.ndarray) -> bool:
        """
        Validiert Rock-Distribution auf realistische Werte

        Args:
            rock_map: Rock-Map zum Validieren

        Returns:
            True wenn Distribution realistisch ist
        """
        try:
            # Durchschnittliche Verteilung berechnen
            avg_distribution = np.mean(rock_map, axis=(0, 1)) / 255.0

            # Realistische Grenzen prüfen (keine Gesteinsart sollte <5% oder >90% sein)
            min_realistic = 0.05
            max_realistic = 0.90

            for i, avg_value in enumerate(avg_distribution):
                if avg_value < min_realistic or avg_value > max_realistic:
                    self.logger.warning(f"Unrealistic rock distribution: Type {i} = {avg_value:.3f}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Rock distribution validation failed: {e}")
            return False

    def _calculate_parameter_hash(self, parameters: Dict[str, Any]) -> str:
        """
        Berechnet Hash der Parameter für Cache-Invalidation

        Args:
            parameters: Parameter-Dictionary

        Returns:
            MD5-Hash der Parameter
        """
        import hashlib
        import json

        # Nur relevante Parameter für Hash verwenden
        relevant_params = {
            k: v for k, v in parameters.items()
            if k in ['sedimentary_hardness', 'igneous_hardness', 'metamorphic_hardness',
                    'ridge_warping', 'bevel_warping', 'metamorphic_foliation',
                    'metamorphic_folding', 'igneous_flowing']
        }

        param_string = json.dumps(relevant_params, sort_keys=True)
        return hashlib.md5(param_string.encode()).hexdigest()

    def _create_fallback_geology_data(self, heightmap_combined: np.ndarray,
                                     parameters: Dict[str, Any], lod_level: int) -> GeologyData:
        """
        Erstellt Fallback-Geology-Data bei kritischen Fehlern

        Args:
            heightmap_combined: Heightmap für Fallback-Generation
            parameters: Parameter für Fallback
            lod_level: Aktuelles LOD-Level

        Returns:
            Minimale aber funktionale GeologyData
        """
        self.logger.warning("Creating fallback geology data due to generation failure")

        height, width = heightmap_combined.shape

        # Einfache Height-basierte Rock-Distribution
        rock_map = np.zeros((height, width, 3), dtype=np.uint8)

        # Normalisierte Höhen
        norm_heights = (heightmap_combined - np.min(heightmap_combined)) / \
                      (np.max(heightmap_combined) - np.min(heightmap_combined) + 1e-6)

        for y in range(height):
            for x in range(width):
                h = norm_heights[y, x]

                if h < 0.3:
                    rock_map[y, x, :] = [180, 50, 25]  # Sedimentary-dominiert
                elif h < 0.7:
                    rock_map[y, x, :] = [85, 85, 85]   # Gleichverteilung
                else:
                    rock_map[y, x, :] = [40, 140, 75]  # Igneous-dominiert

        # Hardness-Map
        hardness_map = np.full((height, width), 50.0, dtype=np.float32)  # Durchschnittshärte

        # Validity-State (partial)
        validity_state = {
            'mass_conservation': True,  # Manuell sichergestellt
            'hardness_range': True,
            'rock_distribution': True,
            'no_nan_values': True,
            'correct_dimensions': True
        }

        return GeologyData(
            rock_map=rock_map,
            hardness_map=hardness_map,
            lod_level=lod_level,
            actual_size=(height, width),
            validity_state=validity_state,
            parameter_hash=self._calculate_parameter_hash(parameters),
            parameters=parameters.copy()
        )

    def update_seed(self, new_seed: int):
        """
        Aktualisiert Seed für alle Geology-Komponenten

        Args:
            new_seed: Neuer Seed für reproduzierbare Generation
        """
        if new_seed != self.map_seed:
            self.map_seed = new_seed

            # Alle Sub-Komponenten mit neuem Seed re-initialisieren
            self.rock_classifier = RockTypeClassifier(new_seed)
            self.deformation_processor = TectonicDeformationProcessor(new_seed)

            self.logger.info(f"Geology generator seed updated to {new_seed}")

    def get_generation_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über den Generator zurück

        Returns:
            Dictionary mit Generator-Informationen
        """
        return {
            'generator_type': 'geology',
            'version': '2.0.0',
            'map_seed': self.map_seed,
            'supported_lod_levels': list(range(1, 8)),
            'required_dependencies': ['heightmap_combined', 'slopemap'],
            'output_data': ['rock_map', 'hardness_map'],
            'fallback_levels': ['gpu', 'cpu', 'simple'],
            'mass_conservation': True
        }

    # ===== LEGACY-KOMPATIBILITÄT =====
    # Alle alten Methoden für Rückwärts-Kompatibilität

    def generate_rock_distribution(self, heightmap, slopemap, ridge_warping, bevel_warping,
                                   metamorph_foliation, metamorph_folding, igneous_flowing):
        """Legacy-Methode für Rock-Distribution"""
        self.logger.warning("Using deprecated generate_rock_distribution method")

        parameters = {
            'ridge_warping': ridge_warping,
            'bevel_warping': bevel_warping,
            'metamorphic_foliation': metamorph_foliation,
            'metamorphic_folding': metamorph_folding,
            'igneous_flowing': igneous_flowing,
            **self.default_parameters
        }

        geology_data = self.calculate_geology(heightmap, slopemap, parameters, lod_level=3)
        return geology_data.rock_map

    def calculate_hardness_map(self, rock_map, sedimentary_hardness, igneous_hardness, metamorphic_hardness):
        """Legacy-Methode für Hardness-Calculation"""
        return self.hardness_calculator.calculate_hardness_map(
            rock_map, sedimentary_hardness, igneous_hardness, metamorphic_hardness
        )

    def apply_geological_zones(self, rock_map, zone_parameters):
        """Legacy-Methode für geologische Zonen"""
        return self.rock_classifier.blend_geological_zones(rock_map)

    def generate_complete_geology(self, heightmap, slopemap, rock_types, hardness_values,
                                  ridge_warping, bevel_warping, metamorph_foliation,
                                  metamorph_folding, igneous_flowing):
        """Legacy-Methode für komplette Geology-Generierung"""
        self.logger.warning("Using deprecated generate_complete_geology method")

        parameters = {
            'sedimentary_hardness': hardness_values.get('sedimentary', 30.0),
            'igneous_hardness': hardness_values.get('igneous', 70.0),
            'metamorphic_hardness': hardness_values.get('metamorphic', 60.0),
            'ridge_warping': ridge_warping,
            'bevel_warping': bevel_warping,
            'metamorphic_foliation': metamorph_foliation,
            'metamorphic_folding': metamorph_folding,
            'igneous_flowing': igneous_flowing
        }

        geology_data = self.calculate_geology(heightmap, slopemap, parameters, lod_level=3)
        return geology_data.rock_map, geology_data.hardness_map

    def validate_rock_map(self, rock_map):
        """Legacy-Validierung mit erweiterten Statistiken"""
        mass_stats = self.mass_manager.get_conservation_statistics(rock_map)

        return {
            'mass_conservation': mass_stats['all_pixels_valid'],
            'value_range_valid': np.all((rock_map >= 0) & (rock_map <= 255)),
            'no_zero_pixels': np.all(np.sum(rock_map, axis=2) > 0),
            'average_distribution': np.mean(rock_map, axis=(0, 1)),
            'conservation_stats': mass_stats
        }


# ===== FACTORY-FUNCTION FÜR EINFACHE INTEGRATION =====

def create_geology_generator(map_seed: int = 42, progress_callback=None) -> GeologySystemGenerator:
    """
    Factory-Funktion für GeologySystemGenerator

    Args:
        map_seed: Seed für reproduzierbare Generation
        progress_callback: Optional Callback für Progress-Updates

    Returns:
        Konfigurierter GeologySystemGenerator
    """
    generator = GeologySystemGenerator(map_seed)
    if progress_callback:
        generator.set_progress_callback(progress_callback)
    return generator


# ===== UTILITY-FUNKTIONEN FÜR EXTERNE INTEGRATION =====

def validate_geology_parameters(parameters: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validiert Geology-Parameter extern

    Args:
        parameters: Parameter zum Validieren

    Returns:
        (is_valid, error_message)
    """
    try:
        required_params = ['sedimentary_hardness', 'igneous_hardness', 'metamorphic_hardness']

        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

            value = parameters[param]
            if not isinstance(value, (int, float)):
                return False, f"Parameter {param} must be numeric"

            if not (0 <= value <= 100):
                return False, f"Parameter {param} must be in range [0-100]"

        optional_params = ['ridge_warping', 'bevel_warping', 'metamorphic_foliation',
                          'metamorphic_folding', 'igneous_flowing']

        for param in optional_params:
            if param in parameters:
                value = parameters[param]
                if not isinstance(value, (int, float)):
                    return False, f"Parameter {param} must be numeric"

                if not (0.0 <= value <= 1.0):
                    return False, f"Parameter {param} must be in range [0.0-1.0]"

        return True, "All parameters valid"

    except Exception as e:
        return False, f"Parameter validation error: {e}"


def get_geology_parameter_info() -> Dict[str, Dict[str, Any]]:
    """
    Gibt Informationen über alle Geology-Parameter zurück

    Returns:
        Dictionary mit Parameter-Informationen
    """
    return {
        'hardness_parameters': {
            'sedimentary_hardness': {'range': [0, 100], 'default': 30.0, 'unit': 'hardness'},
            'igneous_hardness': {'range': [0, 100], 'default': 70.0, 'unit': 'hardness'},
            'metamorphic_hardness': {'range': [0, 100], 'default': 60.0, 'unit': 'hardness'}
        },
        'deformation_parameters': {
            'ridge_warping': {'range': [0.0, 1.0], 'default': 0.3, 'unit': 'factor'},
            'bevel_warping': {'range': [0.0, 1.0], 'default': 0.2, 'unit': 'factor'},
            'metamorphic_foliation': {'range': [0.0, 1.0], 'default': 0.4, 'unit': 'factor'},
            'metamorphic_folding': {'range': [0.0, 1.0], 'default': 0.3, 'unit': 'factor'},
            'igneous_flowing': {'range': [0.0, 1.0], 'default': 0.5, 'unit': 'factor'}
        }
    }