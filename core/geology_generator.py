"""
Path: core/geology_generator.py

Funktionsweise: Geologische Schichten und Gesteinstypen
- Rock-Type Klassifizierung (sedimentary, metamorphic, igneous) basierend auf geologischer Simulation mit folgenden Techniken:
    - die höhen werden genommen und dann mit einer neuen verzerrten noisefunktion multipliziert, so dass die höhen auch variieren für eine erste verteilung (weiche übergänge)
    - in besonders steile hänge werden härtere steine mit reingemischt.
    - geologische zonen werden in bestimmte zonen addiert und subtrahiert. d.h. je eine simplex funktion von 0 bis 1 bei der nur werte über 0.2 für sedimentär, eine andere simplex über 0.6 für metamorph und eine dritte simplex mit über 0.8 für igneous.
    - alle drei maps werden zu einer RGB-Karte gewichtet addiert und die Summe der einzelnen R, G und B ergeben zusammen IMMER 255, so dass die Massenerhaltung bestehen bleibt, aber die Verhältnisse sich jeweils ändern können (später bei Erosion und Sedimentation durch Wasser (water_generator.py)).
    - aus den Gesteinstypen wird entsprechend dem Eingabeparameter rock_type"_hardness" (mit sedimentary, igneous, metamorphic für rock_type) eine hardness_map(x,y) erstellt mit den jeweiligen Verhältnissen aus rock_map

Parameter Input:
- rock_types, hardness_values, ridge_warping, bevel_warping, metamorph_foliation, metamorph_folding, igneous_flowing

data_manager Input:
- heightmap

Output:
- rock_map RGB array
- hardness_map array

Klassen:
GeologyGenerator
    Funktionsweise: Hauptklasse für geologische Schichten und Gesteinstyp-Verteilung
    Aufgabe: Koordiniert Gesteinsverteilung und Härte-Berechnung
    Methoden: generate_rock_distribution(), calculate_hardness_map(), apply_geological_zones()

RockTypeClassifier
    Funktionsweise: Klassifiziert Gesteinstypen (sedimentary, metamorphic, igneous) basierend auf Höhe und Steigung
    Aufgabe: Erstellt rock_map mit RGB-Kanälen für drei Gesteinstypen
    Methoden: classify_by_elevation(), apply_slope_hardening(), blend_geological_zones()

MassConservationManager
    Funktionsweise: Stellt sicher dass R+G+B immer 255 ergibt für Massenerhaltung
    Aufgabe: Verwaltet Gesteins-Massenverteilung für spätere Erosion
    Methoden: normalize_rock_masses(), validate_conservation(), redistribute_masses()
"""

import numpy as np
from opensimplex import OpenSimplex
from core.base_generator import BaseGenerator


class MassConservationManager:
    """
    Funktionsweise: Stellt sicher dass R+G+B immer 255 ergibt für Massenerhaltung
    Aufgabe: Verwaltet Gesteins-Massenverteilung für spätere Erosion
    """

    @staticmethod
    def normalize_rock_masses(rock_map):
        """
        Funktionsweise: Normalisiert RGB-Werte so dass R+G+B=255 für jeden Pixel
        Aufgabe: Gewährleistet Massenerhaltung bei Gesteinsverteilung
        Parameter: rock_map (numpy.ndarray) - RGB-Array mit Gesteinsverteilung
        Returns: numpy.ndarray - Normalisierte RGB-Map mit garantierter Summe 255
        """
        height, width, channels = rock_map.shape
        normalized_map = np.zeros_like(rock_map, dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                r, g, b = rock_map[y, x, :]
                total = float(r + g + b)  # Float für Präzision

                if total > 0:
                    # Normalisierung auf 255 mit Clamping
                    norm_r = max(0, min(255, int((r / total) * 255)))
                    norm_g = max(0, min(255, int((g / total) * 255)))
                    norm_b = max(0, min(255, int((b / total) * 255)))

                    normalized_map[y, x, 0] = norm_r
                    normalized_map[y, x, 1] = norm_g
                    normalized_map[y, x, 2] = norm_b

                    # Rundungsfehler korrigieren - Summe muss exakt 255 sein
                    current_sum = norm_r + norm_g + norm_b
                    diff = 255 - current_sum

                    # Differenz auf größten Kanal verteilen
                    if diff != 0:
                        max_channel = np.argmax([norm_r, norm_g, norm_b])
                        current_value = int(normalized_map[y, x, max_channel])  # Explizit zu int konvertieren
                        new_value = max(0, min(255, current_value + diff))
                        normalized_map[y, x, max_channel] = new_value
                else:
                    # Gleichverteilung bei total=0
                    normalized_map[y, x, :] = [85, 85, 85]  # 85*3 = 255

        return normalized_map

    @staticmethod
    def validate_conservation(rock_map):
        """
        Funktionsweise: Validiert dass alle Pixel R+G+B=255 erfüllen
        Aufgabe: Überprüfung der Massenerhaltung für Debugging
        Parameter: rock_map (numpy.ndarray) - RGB-Array zum Validieren
        Returns: bool - True wenn alle Pixel R+G+B=255 erfüllen
        """
        sums = np.sum(rock_map, axis=2)
        return np.all(sums == 255)

    @staticmethod
    def redistribute_masses(rock_map, erosion_deltas):
        """
        Funktionsweise: Redistributiert Gesteinsmassen nach Erosion unter Erhaltung der Gesamtmasse
        Aufgabe: Massenerhaltung bei Erosions-/Sedimentationsprozessen
        Parameter: rock_map, erosion_deltas - Aktuelle Verteilung und Änderungen
        Returns: numpy.ndarray - Redistributierte Rock-Map
        """
        new_rock_map = rock_map.astype(np.float32) + erosion_deltas

        # Negative Werte auf 0 setzen
        new_rock_map = np.maximum(new_rock_map, 0)

        # Re-normalisierung
        return MassConservationManager.normalize_rock_masses(new_rock_map.astype(np.uint8))


class RockTypeClassifier:
    """
    Funktionsweise: Klassifiziert Gesteinstypen (sedimentary, metamorphic, igneous) basierend auf Höhe und Steigung
    Aufgabe: Erstellt rock_map mit RGB-Kanälen für drei Gesteinstypen
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Rock-Classifier mit Noise-Generatoren für geologische Zonen
        Aufgabe: Setup der Simplex-Noise-Generatoren für verschiedene Gesteinstypen
        Parameter: map_seed (int) - Seed für reproduzierbare Gesteinsverteilung
        """
        # Separate Noise-Generatoren für verschiedene Gesteinstypen
        self.sedimentary_noise = OpenSimplex(seed=map_seed)
        self.metamorphic_noise = OpenSimplex(seed=map_seed + 1000)
        self.igneous_noise = OpenSimplex(seed=map_seed + 2000)

        # Deformation-Noise für ridge/bevel warping
        self.ridge_noise = OpenSimplex(seed=map_seed + 3000)
        self.bevel_noise = OpenSimplex(seed=map_seed + 4000)

    def classify_by_elevation(self, heightmap):
        """
        Funktionsweise: Erste Klassifizierung basierend auf Höhen mit verzerrter Noise-Funktion
        Aufgabe: Basis-Verteilung der Gesteinstypen durch Höhen-basierte Zuordnung
        Parameter: heightmap (numpy.ndarray) - Höhendaten für Klassifizierung
        Returns: numpy.ndarray - RGB-Array mit Basis-Gesteinsverteilung
        """
        height, width = heightmap.shape
        rock_map = np.zeros((height, width, 3), dtype=np.float32)

        # Höhen normalisieren
        min_height = np.min(heightmap)
        max_height = np.max(heightmap)
        height_range = max_height - min_height

        if height_range == 0:
            height_range = 1

        for y in range(height):
            for x in range(width):
                # Normalisierte Höhe [0, 1]
                norm_height = (heightmap[y, x] - min_height) / height_range

                # Koordinaten für Noise
                noise_x = x / width * 4.0  # 4x Frequency für Details
                noise_y = y / height * 4.0

                # Verzerrte Noise-Funktion mit Höhe multipliziert
                height_distortion = self.ridge_noise.noise2(noise_x * 0.5, noise_y * 0.5) * 0.3
                distorted_height = norm_height + height_distortion
                distorted_height = max(0, min(1, distorted_height))  # Clamp [0,1]

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

    def apply_slope_hardening(self, rock_map, slopemap):
        """
        Funktionsweise: Mischt härtere Gesteine in steile Hänge ein
        Aufgabe: Realistische Gesteinsverteilung durch Slope-basierte Härtung
        Parameter: rock_map, slopemap - Aktuelle Gesteinsverteilung und Slope-Daten
        Returns: numpy.ndarray - Rock-Map mit Slope-Härtung
        """
        height, width = rock_map.shape[:2]

        for y in range(height):
            for x in range(width):
                # Slope-Magnitude berechnen
                dz_dx = slopemap[y, x, 0]
                dz_dy = slopemap[y, x, 1]
                slope_magnitude = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

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

    def blend_geological_zones(self, rock_map, ridge_warping, bevel_warping, metamorph_foliation, metamorph_folding,
                               igneous_flowing):
        """
        Funktionsweise: Addiert/subtrahiert geologische Zonen basierend auf Simplex-Funktionen
        Aufgabe: Erstellt geologische Zonen mit spezifischen Schwellwerten für jeden Gesteinstyp
        Parameter: rock_map, ridge_warping, bevel_warping, metamorph_foliation, metamorph_folding, igneous_flowing
        Returns: numpy.ndarray - Rock-Map mit geologischen Zonen
        """
        height, width = rock_map.shape[:2]

        for y in range(height):
            for x in range(width):
                # Normalisierte Koordinaten für Noise
                norm_x = x / width
                norm_y = y / height

                # Geologische Zonen-Noise [0, 1] normalisiert
                sedimentary_zone = (self.sedimentary_noise.noise2(norm_x * 3, norm_y * 3) + 1) * 0.5
                metamorphic_zone = (self.metamorphic_noise.noise2(norm_x * 2, norm_y * 2) + 1) * 0.5
                igneous_zone = (self.igneous_noise.noise2(norm_x * 4, norm_y * 4) + 1) * 0.5

                # Schwellwerte anwenden und Gewichtungen berechnen
                sedimentary_influence = 0.0
                metamorphic_influence = 0.0
                igneous_influence = 0.0

                if sedimentary_zone > 0.2:
                    sedimentary_influence = (sedimentary_zone - 0.2) / 0.8  # [0, 1]

                if metamorphic_zone > 0.6:
                    metamorphic_influence = (metamorphic_zone - 0.6) / 0.4  # [0, 1]

                if igneous_zone > 0.8:
                    igneous_influence = (igneous_zone - 0.8) / 0.2  # [0, 1]

                # Deformation-Faktoren berechnen
                ridge_factor = (self.ridge_noise.noise2(norm_x * 6, norm_y * 6) + 1) * 0.5 * ridge_warping
                bevel_factor = (self.bevel_noise.noise2(norm_x * 8, norm_y * 8) + 1) * 0.5 * bevel_warping
                total_deformation = (ridge_factor + bevel_factor) * 0.3  # Reduzierter Einfluss

                # Metamorphic Foliation/Folding
                metamorph_detail = metamorph_foliation * np.sin(norm_x * 20) + metamorph_folding * np.cos(norm_y * 15)
                metamorphic_influence += abs(metamorph_detail) * 0.1  # abs() für positive Werte

                # Igneous Flowing
                flow_pattern = igneous_flowing * np.sin(norm_x * 12) * np.cos(norm_y * 8)
                igneous_influence += abs(flow_pattern) * 0.1  # abs() für positive Werte

                # Aktuelle Gesteinsverteilung
                current_sed = rock_map[y, x, 0]
                current_ign = rock_map[y, x, 1]
                current_met = rock_map[y, x, 2]

                # Ziel-Verteilungen für geologische Zonen
                target_sed_distribution = [0.8, 0.1, 0.1]  # Sedimentary-dominiert
                target_ign_distribution = [0.1, 0.8, 0.1]  # Igneous-dominiert
                target_met_distribution = [0.1, 0.1, 0.8]  # Metamorphic-dominiert

                # Deformation-Ziel (härtere Gesteine)
                target_deform_distribution = [0.2, 0.5, 0.3]  # Igneous/Metamorphic-betont

                # Basis-Gewichtung (behält aktuelle Verteilung bei)
                base_weight = 1.0 - (sedimentary_influence * 0.4 + metamorphic_influence * 0.4 +
                                     igneous_influence * 0.4 + total_deformation)
                base_weight = max(0.3, base_weight)  # Mindestens 30% der ursprünglichen Verteilung

                # Gewichtete Mischung der verschiedenen Einflüsse
                new_sed = (current_sed * base_weight +
                           target_sed_distribution[0] * sedimentary_influence * 0.4 +
                           target_ign_distribution[0] * igneous_influence * 0.4 +
                           target_met_distribution[0] * metamorphic_influence * 0.4 +
                           target_deform_distribution[0] * total_deformation)

                new_ign = (current_ign * base_weight +
                           target_sed_distribution[1] * sedimentary_influence * 0.4 +
                           target_ign_distribution[1] * igneous_influence * 0.4 +
                           target_met_distribution[1] * metamorphic_influence * 0.4 +
                           target_deform_distribution[1] * total_deformation)

                new_met = (current_met * base_weight +
                           target_sed_distribution[2] * sedimentary_influence * 0.4 +
                           target_ign_distribution[2] * metamorphic_influence * 0.4 +
                           target_met_distribution[2] * metamorphic_influence * 0.4 +
                           target_deform_distribution[2] * total_deformation)

                # Normalisierung auf Summe = 1.0 (Massenerhaltung)
                total_new = new_sed + new_ign + new_met
                if total_new > 0:
                    rock_map[y, x, 0] = new_sed / total_new
                    rock_map[y, x, 1] = new_ign / total_new
                    rock_map[y, x, 2] = new_met / total_new
                else:
                    # Fallback bei numerischen Problemen
                    rock_map[y, x, 0] = 0.33
                    rock_map[y, x, 1] = 0.33
                    rock_map[y, x, 2] = 0.34

        return rock_map


class GeologyGenerator(BaseGenerator):
    """
    Funktionsweise: Hauptklasse für geologische Schichten und Gesteinstyp-Verteilung
    Aufgabe: Koordiniert Gesteinsverteilung und Härte-Berechnung mit universeller Generator-API
    """

    def __init__(self, map_seed=42):
        """
        Funktionsweise: Initialisiert Geology-Generator mit BaseGenerator und Sub-Komponenten
        Aufgabe: Setup von Rock-Classifier und Mass-Conservation-Manager
        Parameter: map_seed (int) - Globaler Seed für reproduzierbare Geologie
        """
        super().__init__(map_seed)
        self.rock_classifier = RockTypeClassifier(map_seed)
        self.mass_manager = MassConservationManager()

        # Standard-Härte-Werte als Fallback
        self.default_hardness = {
            'sedimentary': 30.0,
            'igneous': 70.0,
            'metamorphic': 60.0
        }

    def _load_default_parameters(self):
        """
        Funktionsweise: Lädt GEOLOGY-Parameter aus value_default.py
        Aufgabe: Standard-Parameter für Geology-Generierung
        Returns: dict - Alle Standard-Parameter für Geology
        """
        from gui.config.value_default import GEOLOGY

        return {
            'sedimentary_hardness': GEOLOGY.SEDIMENTARY_HARDNESS["default"],
            'igneous_hardness': GEOLOGY.IGNEOUS_HARDNESS["default"],
            'metamorphic_hardness': GEOLOGY.METAMORPHIC_HARDNESS["default"],
            'ridge_warping': GEOLOGY.RIDGE_WARPING["default"],
            'bevel_warping': GEOLOGY.BEVEL_WARPING["default"],
            'metamorph_foliation': GEOLOGY.METAMORPH_FOLIATION["default"],
            'metamorph_folding': GEOLOGY.METAMORPH_FOLDING["default"],
            'igneous_flowing': GEOLOGY.IGNEOUS_FLOWING["default"]
        }

    def _get_dependencies(self, data_manager):
        """
        Funktionsweise: Holt heightmap und slopemap aus DataManager
        Aufgabe: Dependency-Resolution für Geology-Generierung
        Parameter: data_manager - DataManager-Instanz
        Returns: dict - Benötigte Terrain-Daten
        """
        if not data_manager:
            raise Exception("DataManager required for Geology generation")

        heightmap = data_manager.get_terrain_data("heightmap")
        slopemap = data_manager.get_terrain_data("slopemap")

        if heightmap is None:
            raise Exception("Heightmap dependency not available in DataManager")
        if slopemap is None:
            raise Exception("Slopemap dependency not available in DataManager")

        self.logger.debug(f"Dependencies loaded - heightmap: {heightmap.shape}, slopemap: {slopemap.shape}")

        return {
            'heightmap': heightmap,
            'slopemap': slopemap
        }

    def _execute_generation(self, lod, dependencies, parameters):
        """
        Funktionsweise: Führt Geology-Generierung mit Progress-Updates aus
        Aufgabe: Kernlogik der Geology-Generierung mit allen geologischen Prozessen
        Parameter: lod, dependencies, parameters
        Returns: dict mit rock_map und hardness_map
        """
        heightmap = dependencies['heightmap']
        slopemap = dependencies['slopemap']

        # Schritt 1: Rock Distribution - Elevation-basierte Klassifizierung (20%)
        self._update_progress("Rock Distribution", 20, "Classifying rocks by elevation...")
        rock_map = self.rock_classifier.classify_by_elevation(heightmap)

        # Schritt 2: Rock Distribution - Slope-Härtung (35%)
        self._update_progress("Rock Distribution", 35, "Applying slope hardening...")
        rock_map = self.rock_classifier.apply_slope_hardening(rock_map, slopemap)

        # Schritt 3: Rock Distribution - Geologische Zonen (60%)
        self._update_progress("Rock Distribution", 60, "Blending geological zones...")
        rock_map = self.rock_classifier.blend_geological_zones(
            rock_map,
            parameters['ridge_warping'],
            parameters['bevel_warping'],
            parameters['metamorph_foliation'],
            parameters['metamorph_folding'],
            parameters['igneous_flowing']
        )

        # Schritt 4: Mass Conservation - RGB-Normalisierung (75%)
        self._update_progress("Mass Conservation", 75, "Normalizing rock masses...")
        rock_map_uint8 = (rock_map * 255).astype(np.uint8)
        normalized_rock_map = self.mass_manager.normalize_rock_masses(rock_map_uint8)

        # Validierung der Massenerhaltung
        if not self.mass_manager.validate_conservation(normalized_rock_map):
            self.logger.warning("Mass conservation validation failed")

        # Schritt 5: Hardness Calculation - Gewichtete Härte-Berechnung (90%)
        self._update_progress("Hardness Calculation", 90, "Calculating hardness map...")
        hardness_map = self.calculate_hardness_map(
            normalized_rock_map,
            parameters['sedimentary_hardness'],
            parameters['igneous_hardness'],
            parameters['metamorphic_hardness']
        )

        self.logger.debug(
            f"Generation complete - rock_map: {normalized_rock_map.shape}, hardness_map: {hardness_map.shape}")

        return {
            'rock_map': normalized_rock_map,
            'hardness_map': hardness_map
        }

    def _save_to_data_manager(self, data_manager, result, parameters):
        """
        Funktionsweise: Speichert Geology-Ergebnisse im DataManager
        Aufgabe: Automatische Speicherung aller Geology-Outputs mit Parameter-Tracking
        Parameter: data_manager, result, parameters
        """
        data_manager.set_geology_data("rock_map", result['rock_map'], parameters)
        data_manager.set_geology_data("hardness_map", result['hardness_map'], parameters)

        self.logger.debug("Geology results saved to DataManager")

    def update_seed(self, new_seed):
        """
        Funktionsweise: Aktualisiert Seed für alle Geology-Komponenten
        Aufgabe: Seed-Update mit Re-Initialisierung der Sub-Komponenten
        Parameter: new_seed (int) - Neuer Seed
        """
        if new_seed != self.map_seed:
            super().update_seed(new_seed)
            # Rock-Classifier mit neuem Seed re-initialisieren
            self.rock_classifier = RockTypeClassifier(new_seed)

    # ===== LEGACY-KOMPATIBILITÄT =====
    # Alle alten Methoden bleiben für Rückwärts-Kompatibilität erhalten

    def generate_rock_distribution(self, heightmap, slopemap, ridge_warping, bevel_warping,
                                   metamorph_foliation, metamorph_folding, igneous_flowing):
        """Legacy-Methode für Rock-Distribution"""
        rock_map = self.rock_classifier.classify_by_elevation(heightmap)
        rock_map = self.rock_classifier.apply_slope_hardening(rock_map, slopemap)
        rock_map = self.rock_classifier.blend_geological_zones(
            rock_map, ridge_warping, bevel_warping, metamorph_foliation, metamorph_folding, igneous_flowing
        )
        rock_map_uint8 = (rock_map * 255).astype(np.uint8)
        return self.mass_manager.normalize_rock_masses(rock_map_uint8)

    def calculate_hardness_map(self, rock_map, sedimentary_hardness, igneous_hardness, metamorphic_hardness):
        """
        Funktionsweise: Berechnet Härte-Map basierend auf Gesteinstyp-Verhältnissen und Härte-Parametern
        Aufgabe: Erstellt Hardness-Map für Erosions-/Weathering-Systeme
        Parameter: rock_map, sedimentary_hardness, igneous_hardness, metamorphic_hardness
        Returns: numpy.ndarray - Hardness-Map mit gewichteten Härte-Werten
        """
        height, width = rock_map.shape[:2]
        hardness_map = np.zeros((height, width), dtype=np.float32)

        # Härte-Array für Gewichtung
        hardness_values = np.array([sedimentary_hardness, igneous_hardness, metamorphic_hardness])

        for y in range(height):
            for x in range(width):
                # Gesteinsanteile (normalisiert auf [0, 1])
                rock_ratios = rock_map[y, x, :].astype(np.float32) / 255.0

                # Gewichtete Härte berechnen
                weighted_hardness = np.sum(rock_ratios * hardness_values)
                hardness_map[y, x] = weighted_hardness

        return hardness_map

    def apply_geological_zones(self, rock_map, zone_parameters):
        """Legacy-Methode für geologische Zonen"""
        return rock_map

    def generate_complete_geology(self, heightmap, slopemap, rock_types, hardness_values,
                                  ridge_warping, bevel_warping, metamorph_foliation,
                                  metamorph_folding, igneous_flowing):
        """
        Funktionsweise: Legacy-Methode für komplette Geology-Generierung
        Aufgabe: Rückwärts-Kompatibilität mit alter API
        """
        # Konvertiert alte API zur neuen API
        dependencies = {'heightmap': heightmap, 'slopemap': slopemap}
        parameters = {
            'sedimentary_hardness': hardness_values.get('sedimentary', self.default_hardness['sedimentary']),
            'igneous_hardness': hardness_values.get('igneous', self.default_hardness['igneous']),
            'metamorphic_hardness': hardness_values.get('metamorphic', self.default_hardness['metamorphic']),
            'ridge_warping': ridge_warping,
            'bevel_warping': bevel_warping,
            'metamorph_foliation': metamorph_foliation,
            'metamorph_folding': metamorph_folding,
            'igneous_flowing': igneous_flowing
        }

        result = self._execute_generation("LOD64", dependencies, parameters)
        return result['rock_map'], result['hardness_map']

    def validate_rock_map(self, rock_map):
        """
        Funktionsweise: Validiert Rock-Map auf Massenerhaltung und korrekte Werte
        Aufgabe: Debugging-Funktion für Rock-Map-Validierung
        Parameter: rock_map (numpy.ndarray) - Rock-Map zum Validieren
        Returns: dict - Validierungs-Ergebnisse
        """
        results = {
            'mass_conservation': self.mass_manager.validate_conservation(rock_map),
            'value_range_valid': np.all((rock_map >= 0) & (rock_map <= 255)),
            'no_zero_pixels': np.all(np.sum(rock_map, axis=2) > 0),
            'average_distribution': np.mean(rock_map, axis=(0, 1))
        }

        return results