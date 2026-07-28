"""
Throwaway headless smoke test for the Weather climatology + water-balance
changes (core/weather_generator.py). Not part of the test suite - run
manually via the shared venv, see CLAUDE.md.
"""
import sys
import traceback

import numpy as np

sys.path.insert(0, r"C:\Lokale Dateien\Projects\Python\MapGenerator")

from core.weather_generator import WeatherSystemGenerator
from gui.OldManagers.data_lod_manager import DataLODManager
from gui.config.value_default import WEATHER


def run_thermal_pressure_toggle():
    """Weather-Rework Punkt A: thermisch gekoppeltes Druckfeld mit Rückweg.
    Bei einem Terrain mit echter räumlicher Temperaturvariation (Nord-Sued-
    Ridge, wie beim Solar-Heating-Test) müssen sich EIN/AUS im Windfeld
    unterscheiden (Toggle hat einen echten Effekt), beide Pfade müssen aber
    weiterhin gültige, endliche Ergebnisse liefern (kein Absturz/NaN in
    keinem der beiden Zustände)."""
    from core.terrain_generator import ShadowCalculator, generate_seasonal_sun_angles

    size = 48
    y = np.arange(size)[:, None] * np.ones((1, size))
    heightmap = (400.0 - np.abs(y - size / 2) * 8.0).astype(np.float32)

    calc = ShadowCalculator()
    sun_angles = generate_seasonal_sun_angles(1, 48.0, 0.0)
    shadowmap = calc.calculate_shadows(heightmap, lod_level=3, sun_angles_override=sun_angles)

    base_params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 48.0,
        'map_longitude': 0.0,
    }

    ok = True
    results = {}
    for coupling in (True, False):
        dlm = DataLODManager()
        dlm.set_map_distance_km(20.0)
        geo = WeatherSystemGenerator(map_seed=7, data_lod_manager=dlm)
        params = dict(base_params, thermal_pressure_coupling=coupling)
        try:
            result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
        except Exception as e:
            print(f"[FAIL] calculate_weather_system raised (coupling={coupling}): {e}")
            traceback.print_exc()
            return False
        ok &= check(f"coupling={coupling}: gueltiges Ergebnis", result.is_valid())
        ok &= check(f"coupling={coupling}: wind_map endlich", bool(np.all(np.isfinite(result.wind_map))))
        results[coupling] = result

    wind_diff = np.abs(results[True].wind_map - results[False].wind_map).max()
    ok &= check(f"Toggle hat einen messbaren Effekt (max |delta_wind|={wind_diff:.4f} > 0)",
                wind_diff > 1e-6)
    return ok


def run_slope_aspect_solar_heating():
    """Weather-Rework Punkt B: Nord-Sued-Ridge, Nordhalbkugel - Suedhang muss
    im finalen temp_map waermer sein als Nordhang (weder frueher Flach-
    Mittel ueber die 7 Sonnenwinkel-Kanaele noch reiner Wind-Term reichten
    dafuer, siehe _weighted_solar_exposure() + direkte theta-Kopplung in
    _run_coupled_atmosphere_simulation())."""
    from core.terrain_generator import ShadowCalculator, generate_seasonal_sun_angles

    size = 48
    y = np.arange(size)[:, None] * np.ones((1, size))
    heightmap = (400.0 - np.abs(y - size / 2) * 8.0).astype(np.float32)

    calc = ShadowCalculator()
    sun_angles = generate_seasonal_sun_angles(1, 48.0, 0.0)
    shadowmap = calc.calculate_shadows(heightmap, lod_level=3, sun_angles_override=sun_angles)

    dlm = DataLODManager()
    dlm.set_map_distance_km(20.0)
    geo = WeatherSystemGenerator(map_seed=7, data_lod_manager=dlm)
    params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 48.0,
        'map_longitude': 0.0,
    }
    try:
        result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised: {e}")
        traceback.print_exc()
        return False

    south_temp = float(result.temp_map[size // 4, size // 2])
    north_temp = float(result.temp_map[3 * size // 4, size // 2])
    gap = south_temp - north_temp
    ok = check(f"Suedhang ({south_temp:.1f}C) waermer als Nordhang ({north_temp:.1f}C)",
               south_temp > north_temp)
    ok &= check(f"Sued-Nord-Gap ({gap:.1f}C) in plausiblem Fenster (0 < gap < 40)",
                0.0 < gap < 40.0)
    return ok


def check(label, condition):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def run_edge_padding_sanity():
    """Rand-Puffer (Weather-Rework Punkt C, _compute_edge_padding_px()) -
    0 bei map_distance_km<=0 (Guard gegen Division durch 0/negative Werte),
    ansonsten positiv und auf max. 20% der Gittergröße geklemmt."""
    dlm_zero = DataLODManager()
    dlm_zero.set_map_distance_km(0.0)
    geo_zero = WeatherSystemGenerator(map_seed=1, data_lod_manager=dlm_zero)
    ok = check("map_distance_km=0 -> pad=0", geo_zero._compute_edge_padding_px(64) == 0)

    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    geo = WeatherSystemGenerator(map_seed=1, data_lod_manager=dlm)
    pad_32 = geo._compute_edge_padding_px(32)   # km_per_px=0.3125 -> 2/0.3125=6.4 -> 6, max_pad=32//5=6
    pad_512 = geo._compute_edge_padding_px(512)  # km_per_px=0.0195 -> weit über max_pad -> geklemmt
    ok &= check(f"32px-Gitter, 10km Karte: pad={pad_32} (erwartet 6, geklemmt an max_pad)", pad_32 == 6)
    ok &= check(f"512px-Gitter, 10km Karte: pad={pad_512} > 0 und <= 512//5",
                0 < pad_512 <= 512 // 5)
    return ok


def run_climatology_sanity():
    """Äquator muss ganzjährig wärmer sein als der Pol, Südhalbkugel-Sommer
    (Januar) muss wärmer sein als Südhalbkugel-Winter (Juli)."""
    geo = WeatherSystemGenerator(map_seed=1)
    equator_july = geo._climate_baseline(0.0, 3)[0]   # Monatsindex 3 ~ Juli-Periode
    pole_july = geo._climate_baseline(85.0, 3)[0]
    pole_jan = geo._climate_baseline(85.0, 0)[0]
    south_pole_jan = geo._climate_baseline(-85.0, 0)[0]
    south_pole_july = geo._climate_baseline(-85.0, 3)[0]

    ok = check(f"Äquator (Juli={equator_july:.1f}) waermer als Nordpol (Juli={pole_july:.1f})",
               equator_july > pole_july)
    ok &= check(f"Nordpol Juli ({pole_july:.1f}) waermer als Nordpol Januar ({pole_jan:.1f})",
                pole_july > pole_jan)
    ok &= check(f"Suedpol Januar ({south_pole_jan:.1f}, Sommer) waermer als Suedpol Juli "
                f"({south_pole_july:.1f}, Winter) - Hemisphaeren-Phasenumkehr",
                south_pole_jan > south_pole_july)

    humid_equator = geo._climate_baseline(0.0, 0)[1]
    humid_pole = geo._climate_baseline(85.0, 0)[1]
    ok &= check(f"Aequator feuchter ({humid_equator:.2f}) als Pol ({humid_pole:.2f})",
                humid_equator > humid_pole)
    return ok


def run_baroclinic_wind_scaling():
    """Nutzer-Bug-Report 2026-07-23: gemessene Windstaerken lagen selbst bei
    Extremwerten nur bei ~2-11 m/s, unabhaengig von Breitengrad/Jahreszeit -
    der thermisch gekoppelte Druckterm reagiert nur auf LOKALE Gradienten,
    eine gleichmaessige Klimatologie-Verschiebung der ganzen Karte hat keinen
    Effekt. _baroclinic_wind_factor() skaliert das fertige Windfeld direkt
    (zwei Versuche, stattdessen nur interne Druck-/Beschleunigungs-Terme zu
    skalieren, blieben wirkungslos - siehe Docstring). Hohe Breite im Winter
    muss jetzt spuerbar staerkeren Wind zeigen als der Aequator."""
    size = 48
    rng = np.random.RandomState(6)
    heightmap = (500 + 200 * rng.randn(size, size)).astype(np.float32)
    shadowmap = np.clip(0.5 + 0.2 * rng.randn(size, size, 7), 0.0, 1.0).astype(np.float32)

    dlm = DataLODManager()
    dlm.set_map_distance_km(15.0)
    geo = WeatherSystemGenerator(map_seed=8, data_lod_manager=dlm)
    base_params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_longitude': 0.0,
    }

    equator_factor = geo._baroclinic_wind_factor(5.0, 0)
    polar_winter_factor = geo._baroclinic_wind_factor(80.0, 0)
    polar_summer_factor = geo._baroclinic_wind_factor(80.0, 3)
    ok = check(f"Aequator-Faktor ({equator_factor:.2f}) < Pol-Winter-Faktor ({polar_winter_factor:.2f})",
               equator_factor < polar_winter_factor)
    ok &= check(f"Pol-Sommer-Faktor ({polar_summer_factor:.2f}) < Pol-Winter-Faktor ({polar_winter_factor:.2f})",
                polar_summer_factor < polar_winter_factor)

    try:
        result_equator = geo.calculate_weather_system(
            heightmap, shadowmap, dict(base_params, map_latitude=5.0), lod_level=3)
        result_polar_winter = geo.calculate_weather_system(
            heightmap, shadowmap, dict(base_params, map_latitude=80.0), lod_level=3)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised: {e}")
        traceback.print_exc()
        return False

    equator_wind_max = float(np.hypot(result_equator.wind_map[:, :, 0],
                                       result_equator.wind_map[:, :, 1]).max())
    polar_wind_max = float(np.hypot(result_polar_winter.wind_map[:, :, 0],
                                     result_polar_winter.wind_map[:, :, 1]).max())
    ok &= check(f"Simuliertes Windfeld: Aequator-Max ({equator_wind_max:.2f} m/s) "
                f"< hohe-Breite-Max ({polar_wind_max:.2f} m/s)",
                equator_wind_max < polar_wind_max)
    ok &= check("Beide Ergebnisse gueltig/endlich",
                result_equator.is_valid() and result_polar_winter.is_valid()
                and bool(np.all(np.isfinite(result_equator.wind_map)))
                and bool(np.all(np.isfinite(result_polar_winter.wind_map))))
    return ok


def run_precip_annual_rescale():
    """Nutzer-Abstimmung 2026-07-24 (Revision der Kalibrierung vom
    2026-07-23): precip_map ist eine Perioden-Akkumulation (keine
    Jahresmenge - "mm" ist als Wassertiefe ohnehin bereits pro m² definiert,
    unabhaengig von der Pixelgroesse), kalibriert auf ~50 als typischen
    Maximalwert, seltene Ausreisser erlaubt. Verifiziert: (1) der End-
    Skalierungsfaktor selbst ist korrekt (0.5x), (2) ein feucht-tropisches
    Szenario (Aequator, hohe Luftfeuchte) liegt in einer plausiblen,
    aber nicht extremen Groessenordnung (spuerbar > trockene Vergleichsfaelle,
    aber nicht im drei- oder vierstelligen Bereich)."""
    from core.weather_generator import PRECIP_ANNUAL_SCALE_FACTOR
    ok = check(f"PRECIP_ANNUAL_SCALE_FACTOR == 0.5 (aktuell {PRECIP_ANNUAL_SCALE_FACTOR})",
               abs(PRECIP_ANNUAL_SCALE_FACTOR - 0.5) < 1e-9)

    size = 48
    rng = np.random.RandomState(7)
    heightmap = (300 + 100 * rng.randn(size, size)).astype(np.float32)
    shadowmap = np.full((size, size, 7), 0.7, dtype=np.float32)

    dlm = DataLODManager()
    dlm.set_map_distance_km(15.0)
    geo = WeatherSystemGenerator(map_seed=21, data_lod_manager=dlm)
    params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': 40.0,
        'map_latitude': 3.0,
        'map_longitude': 0.0,
    }
    try:
        result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised: {e}")
        traceback.print_exc()
        return False

    precip_mean = float(result.precip_map.mean())
    ok &= check(f"Feucht-tropisches Szenario: mittlerer Niederschlag ({precip_mean:.1f}mm) "
                f"plausibel (erwartet zwischen 3 und 60, NICHT im Jahres-mm-Bereich)",
                3.0 < precip_mean < 60.0)
    ok &= check("precip_map endlich und >= 0",
                bool(np.all(np.isfinite(result.precip_map)) and np.all(result.precip_map >= 0)))
    return ok


def run_climatology_reference_table_match():
    """Nutzer-Abstimmung 2026-07-23: _climate_baseline() muss an den 10x6
    Stuetzstellen von _TEMP_CLIMATOLOGY_TABLE exakt den abgestimmten Wert
    liefern (keine Interpolation an den Stuetzstellen selbst), UND die
    Altitude-Kuehlung (bestehender 6C/km-Lapse-Rate-Parameter) muss die vom
    Nutzer vorgegebenen Referenz-Deltas fuer 500/1500/3000/4500m (ggue.
    100m) reproduzieren."""
    geo = WeatherSystemGenerator(map_seed=1)
    ok = True
    for row, lat in enumerate(range(0, 91, 10)):
        for col in range(6):
            expected = float(geo._TEMP_CLIMATOLOGY_TABLE[row, col])
            actual = geo._climate_baseline(float(lat), col)[0]
            ok &= check(f"lat={lat} col={col}: {actual:.1f} == {expected:.1f}",
                        abs(actual - expected) < 1e-6)

    # Altitude-Lapse-Rate: 6.0 C/km ab dem 100m-Referenzpunkt (siehe
    # ALTITUDE_COOLING-Default) - reine Arithmetik, kein eigener
    # Klimatologie-Code, hier nur als Dokumentation/Regressionsschutz der
    # vom Nutzer bestaetigten Referenz-Deltas.
    lapse_c_per_km = WEATHER.ALTITUDE_COOLING["default"] / 1000.0
    for altitude_m, expected_delta in ((500, -2.4), (1500, -8.4), (3000, -17.4), (4500, -26.4)):
        delta = -lapse_c_per_km * (altitude_m - 100)
        ok &= check(f"Altitude-Delta bei {altitude_m}m: {delta:.1f}C (erwartet {expected_delta:.1f}C)",
                    abs(delta - expected_delta) < 0.1)
    return ok


def run_lod_inheritance_path_dependence():
    """Weather-Rework Punkt F: LOD-Vererbung. calculate_weather_system()
    ruft _calc_temperature() direkt auf (siehe dessen Docstring, "Standalone-
    Convenience-Entry-Point... der Effekt ist identisch" zur echten GUI-
    Pipeline) - ein schrittweiser LOD-1-dann-LOD-2-Aufbau auf demselben
    DataLODManager nimmt daher automatisch den echten Vererbungspfad
    (bikubisch hochskalierter Endzustand von LOD 1 als CFD-Startbedingung
    für LOD 2), während ein direkter LOD-2-Sprung auf einem frischen
    DataLODManager (keine LOD-1-Vorstufe vorhanden) auf das alte, rein
    Noise-basierte Seeding zurückfällt. Beide Pfade müssen gültig bleiben,
    sich aber messbar unterscheiden (Pfadabhängigkeit ist hier - anders als
    bei Geology - ausdrücklich gewollt, siehe Plan-Verifikationsabschnitt)."""
    size = 48
    y = np.arange(size)[:, None] * np.ones((1, size))
    heightmap = (400.0 - np.abs(y - size / 2) * 8.0).astype(np.float32)

    from core.terrain_generator import ShadowCalculator, generate_seasonal_sun_angles
    calc = ShadowCalculator()
    sun_angles = generate_seasonal_sun_angles(1, 48.0, 0.0)
    shadowmap = calc.calculate_shadows(heightmap, lod_level=2, sun_angles_override=sun_angles)

    params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 48.0,
        'map_longitude': 0.0,
    }

    # Pfad 1: schrittweiser Aufbau LOD 1 -> LOD 2 auf demselben Manager
    # (echte Vererbung greift für LOD 2).
    dlm_inherited = DataLODManager()
    dlm_inherited.set_map_distance_km(20.0)
    geo_inherited = WeatherSystemGenerator(map_seed=7, data_lod_manager=dlm_inherited)
    try:
        result_lod1 = geo_inherited.calculate_weather_system(heightmap, shadowmap, params, lod_level=1)
        result_inherited = geo_inherited.calculate_weather_system(heightmap, shadowmap, params, lod_level=2)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised (inherited path): {e}")
        traceback.print_exc()
        return False

    # Pfad 2: direkter LOD-2-Sprung auf frischem Manager (keine Vorstufe,
    # noise-basiertes Seeding wie vor dem Rework).
    dlm_fresh = DataLODManager()
    dlm_fresh.set_map_distance_km(20.0)
    geo_fresh = WeatherSystemGenerator(map_seed=7, data_lod_manager=dlm_fresh)
    try:
        result_fresh = geo_fresh.calculate_weather_system(heightmap, shadowmap, params, lod_level=2)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised (fresh path): {e}")
        traceback.print_exc()
        return False

    ok = check("LOD 1 gueltiges Ergebnis", result_lod1.is_valid())
    ok &= check("Vererbter LOD-2-Pfad gueltiges Ergebnis", result_inherited.is_valid())
    ok &= check("Frischer LOD-2-Pfad gueltiges Ergebnis", result_fresh.is_valid())
    ok &= check("Vererbter Pfad: temp_map endlich", bool(np.all(np.isfinite(result_inherited.temp_map))))
    ok &= check("Frischer Pfad: temp_map endlich", bool(np.all(np.isfinite(result_fresh.temp_map))))

    temp_diff = np.abs(result_inherited.temp_map - result_fresh.temp_map).max()
    wind_diff = np.abs(result_inherited.wind_map - result_fresh.wind_map).max()
    ok &= check(f"Vererbter vs. frischer LOD-2-Pfad unterscheidet sich messbar "
                f"(max |delta_temp|={temp_diff:.4f}, max |delta_wind|={wind_diff:.4f}, "
                f"erwartet > 0 - Pfadabhaengigkeit ist hier gewollt)",
                temp_diff > 1e-6 or wind_diff > 1e-6)
    return ok


def run_ground_temp_climatology_tracking():
    """Ground-Temperature-Modell (Plan "Weather: Bodentemperatur-Modell +
    konvektiver Waermeuebergang") Punkt (a): bei flachem Terrain und
    gleichmaessiger (0.5) Sonnenexposition soll der aus der Klimatologie
    kalibrierte Boden (T_boden ~ climate_temp bei ground_temp_offset=0) die
    resultierende Jahres-Durchschnittstemperatur grob in Richtung des
    _climate_baseline()-Jahresmittels ziehen - grosszuegige Toleranz wegen
    Wind/Advektion/Latentwaerme/Konvektions-Aequilibrierung. Toleranz auf
    15C angehoben (Nutzer-Feedback 2026-07-25: LATENT_HEAT_COEFFICIENT
    0.1->1.7 hochskaliert, am feucht-heissen Aequator-Szenario mit
    Standard-Feuchte betraegt der reine Kondensations-Waermebeitrag jetzt
    bis zu ~12C - siehe scratch_ground_heat_humidity_isolation.py)."""
    size = 48
    heightmap = np.full((size, size), 100.0, dtype=np.float32)
    shadowmap = np.full((size, size, 7), 0.5, dtype=np.float32)

    ok = True
    for latitude in (0.0, 60.0):
        params = {
            'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
            'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
            'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
            'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
            'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
            'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
            'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
            'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
            'map_latitude': latitude,
            'map_longitude': 0.0,
        }
        dlm = DataLODManager()
        dlm.set_map_distance_km(20.0)
        geo = WeatherSystemGenerator(map_seed=13, data_lod_manager=dlm)
        expected = float(np.mean([geo._climate_baseline(latitude, m)[0] for m in range(6)]))
        try:
            result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
        except Exception as e:
            print(f"[FAIL] calculate_weather_system raised (lat={latitude}): {e}")
            traceback.print_exc()
            return False
        actual = float(result.temp_map.mean())
        ok &= check(f"lat={latitude}: mittlere Temp ({actual:.1f}C) nahe Klimatologie-Jahresmittel "
                    f"({expected:.1f}C), Toleranz 15C", abs(actual - expected) < 15.0)
    return ok


def run_wind_dependent_ground_equilibration():
    """Ground-Temperature-Modell Punkt (c): der neue Boden-Luft-Waermeuebergang
    (Paquet-Formel + Massenerhaltung, siehe GROUND_HEAT_TIME_SCALE_S) muss bei
    unterschiedlicher Windgeschwindigkeit ein MESSBAR unterschiedliches
    Ergebnis liefern (nicht dass eine Seite "richtiger" ist, sondern dass der
    Wind-Term ueberhaupt wirkt - faengt eine zu gross gewaehlte Zeitskala ab,
    die beide Faelle in Saettigung laufen laesst)."""
    size = 48
    heightmap = np.full((size, size), 100.0, dtype=np.float32)
    shadowmap = np.full((size, size, 7), 1.0, dtype=np.float32)

    results = {}
    for label, wsf in (("low_wind", WEATHER.WIND_SPEED_FACTOR["min"]),
                        ("high_wind", WEATHER.WIND_SPEED_FACTOR["max"])):
        params = {
            'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
            'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
            'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
            'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
            'wind_speed_factor': wsf,
            'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
            'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
            'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
            'map_latitude': 20.0,
            'map_longitude': 0.0,
        }
        dlm = DataLODManager()
        dlm.set_map_distance_km(20.0)
        geo = WeatherSystemGenerator(map_seed=17, data_lod_manager=dlm)
        try:
            results[label] = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
        except Exception as e:
            print(f"[FAIL] calculate_weather_system raised ({label}): {e}")
            traceback.print_exc()
            return False

    ok = check("low_wind gueltiges Ergebnis", results["low_wind"].is_valid())
    ok &= check("high_wind gueltiges Ergebnis", results["high_wind"].is_valid())
    diff = float(np.abs(results["low_wind"].temp_map - results["high_wind"].temp_map).mean())
    ok &= check(f"Windgeschwindigkeit hat messbaren Effekt auf temp_map (mean |delta|={diff:.3f}C > 0.05C)",
                diff > 0.05)
    return ok


def run_airmass_dampening():
    """Ground-Temperature-Modell Punkt (d): _weighted_solar_exposure()
    gewichtet Kanaele UNTEREINANDER nach Airmass - ein Test mit einer über
    alle Kanaele KONSTANTEN shadowmap kann das nicht zeigen (ein gewichteter
    Mittelwert einer Konstanten bleibt immer diese Konstante, unabhaengig von
    den Gewichten - erste Version dieses Tests hatte genau diesen Fehler).
    Stattdessen: zwei Kanaele mit UNTERSCHIEDLICHEM Shadow-Wert (Daemmerung
    dunkel=0.2, Mittag hell=1.0, realistische Form), Daemmerungs-Winkel FEST
    niedrig gehalten, nur der Mittags-Winkel steigt von "schwacher" zu
    "hoher" Sonne - die Airmass-Gewichtung muss den hellen Mittags-Kanal bei
    hoher Mittagssonne STAERKER relativ zur dunklen Daemmerung gewichten als
    bei schwacher Mittagssonne, der kombinierte Wert steigt entsprechend."""
    dlm = DataLODManager()
    geo = WeatherSystemGenerator(map_seed=19, data_lod_manager=dlm)

    shadowmap = np.array([[[0.2, 1.0]]], dtype=np.float32)  # (1,1,2): Daemmerung dunkel, Mittag hell
    weak_noon = [(5.0, 90.0), (20.0, 180.0)]
    strong_noon = [(5.0, 90.0), (70.0, 180.0)]

    low_exposure = float(geo._weighted_solar_exposure(shadowmap, sun_angles=weak_noon).mean())
    high_exposure = float(geo._weighted_solar_exposure(shadowmap, sun_angles=strong_noon).mean())
    return check(f"Schwache Mittagssonne (20 Grad, exposure={low_exposure:.3f}) gewichtet den hellen "
                 f"Mittags-Kanal schwaecher als starke Mittagssonne (70 Grad, exposure={high_exposure:.3f})",
                 low_exposure < high_exposure - 0.01)


def run_biome_solar_absorption():
    """Ground-Temperature-Modell Punkt (e): _get_solar_absorption_factor()
    liest den biom-abhaengigen Absorptionsfaktor der VORHERIGEN LOD-Stufe
    (analog _get_roughness_damping) - direkter Test mit synthetischer
    biome_map (Haelfte Wueste=Index 6, Haelfte Regenwald=Index 8, siehe
    core/biome_generator.py BaseBiomeClassifier.biome_definitions), PLUS
    Regressionsschutz: ohne vorherige Biome-Anfrage (Standard-Testpfad)
    muss der Faktor None bleiben (kein Verhaltensunterschied ggue. vor
    dieser Aenderung)."""
    from core.weather_generator import _BIOME_SOLAR_ABSORPTION

    dlm = DataLODManager()
    geo = WeatherSystemGenerator(map_seed=23, data_lod_manager=dlm)

    biome_map = np.zeros((16, 16), dtype=np.int32)
    biome_map[:, 8:] = 6   # desert
    biome_map[:, :8] = 8   # tropical_rainforest
    dlm.set_calculator_output("biome.integrate_layers", 1, {"biome_map": biome_map})

    factor = geo._get_solar_absorption_factor((16, 16), lod_level=1)
    ok = check("Absorptionsfaktor nicht None bei vorhandener Biome-Karte", factor is not None)
    if factor is not None:
        ok &= check(f"Wueste-Haelfte traegt erwarteten Wert ({_BIOME_SOLAR_ABSORPTION[6]:.2f})",
                    bool(np.allclose(factor[:, 8:], _BIOME_SOLAR_ABSORPTION[6])))
        ok &= check(f"Regenwald-Haelfte traegt erwarteten Wert ({_BIOME_SOLAR_ABSORPTION[8]:.2f})",
                    bool(np.allclose(factor[:, :8], _BIOME_SOLAR_ABSORPTION[8])))

    dlm_empty = DataLODManager()
    geo_empty = WeatherSystemGenerator(map_seed=23, data_lod_manager=dlm_empty)
    factor_empty = geo_empty._get_solar_absorption_factor((16, 16), lod_level=1)
    ok &= check("Ohne Biome-Daten liefert _get_solar_absorption_factor None (kein Verhaltensunterschied)",
                factor_empty is None)
    return ok


def run_soil_moisture_coupling():
    """Weather-Rework Punkt G: Bodenfeuchte-/Wasserflaechen-Kopplung ueber
    vorheriges LOD. _calc_temperature() liest water.soil_moisture der
    VORHERIGEN LOD-Stufe und ersetzt damit den alten pauschalen 50%-
    Platzhalter in der Verdunstungs-Berechnung (core/weather_generator.py,
    _run_coupled_atmosphere_simulation, evap_rate0/evap_rate). Ein LOD-1-
    Bodenfeuchte-Eintrag mit durchgehend hoher Feuchte (Sumpf/See-aehnlich,
    95%) muss bei sonst identischen Parametern zu spuerbar mehr Luftfeuchte
    fuehren als ein LOD-1-Eintrag mit durchgehend niedriger Feuchte (Wueste-
    aehnlich, 10%) - Wasser kann nur verdunsten, was der Boden hergibt."""
    size = 48
    heightmap = np.full((size, size), 400.0, dtype=np.float32)
    shadowmap = np.full((size, size, 7), 0.7, dtype=np.float32)

    params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 20.0,
        'map_longitude': 0.0,
    }

    results = {}
    for label, soil_value in (("swamp", 95.0), ("desert", 10.0)):
        dlm = DataLODManager()
        dlm.set_map_distance_km(20.0)
        # Simuliert einen bereits abgeschlossenen Water-Durchlauf bei LOD 1 -
        # _calc_temperature holt diesen Output direkt über get_calculator_output
        # ("water.soil_moisture", "soil_moist_map", lod_level-1), unabhaengig
        # davon, ob Water in diesem Test tatsaechlich lief.
        soil_moist_map = np.full((32, 32), soil_value, dtype=np.float32)
        dlm.set_calculator_output("water.soil_moisture", 1, {"soil_moist_map": soil_moist_map})

        geo = WeatherSystemGenerator(map_seed=11, data_lod_manager=dlm)
        try:
            results[label] = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=2)
        except Exception as e:
            print(f"[FAIL] calculate_weather_system raised ({label}): {e}")
            traceback.print_exc()
            return False

    ok = check("swamp-Szenario gueltiges Ergebnis", results["swamp"].is_valid())
    ok &= check("desert-Szenario gueltiges Ergebnis", results["desert"].is_valid())
    ok &= check("swamp humid_map endlich", bool(np.all(np.isfinite(results["swamp"].humid_map))))
    ok &= check("desert humid_map endlich", bool(np.all(np.isfinite(results["desert"].humid_map))))

    swamp_humid = float(results["swamp"].humid_map.mean())
    desert_humid = float(results["desert"].humid_map.mean())
    ok &= check(f"Sumpf-Bodenfeuchte (95%) fuehrt zu mehr Luftfeuchte "
                f"({swamp_humid:.2f}) als Wuesten-Bodenfeuchte (10%, {desert_humid:.2f})",
                swamp_humid > desert_humid)
    return ok


def run_altitude_lapse_persists_through_advection():
    """Nutzer-Bug-Report 2026-07-23: die Hoehen-Abkuehlung (altitude_cooling-
    Lapse-Rate) wurde nur EINMALIG beim Seeding in theta eingebacken, aber
    theta wird danach als "elevations-unabhaengige" potentielle Temperatur
    behandelt und per Wind advehiert - Luft, die vom Berg herunterweht,
    "vergisst" ihre Herkunftshoehe nicht (bleibt kalt in der warmen Ebene),
    waehrend Luft, die in die Berge weht, sich nie abkuehlt (Berg wird zu
    warm). Root Cause: t_real wurde nur mit dem FESTEN Schicht-AGL-Versatz
    berechnet, nie mit der tatsaechlichen lokalen Terrainhoehe. Fix: t_real
    wird jetzt JEDEN Schritt frisch aus theta UND der lokalen heightmap
    abgeleitet (siehe core/weather_generator.py, Suchbegriff "t_real[i] =").
    Test: ein Berg in der Kartenmitte, staerkerer Wind quer drueber (erzwingt
    tatsaechliche Advektion ueber das Terrain hinweg), nach der vollen
    Simulation muss der Berg-Gipfel spuerbar kaelter sein als das flache
    Tiefland (nicht nur unwesentlich, und nicht umgekehrt)."""
    size = 48
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    center = size / 2.0
    dist = np.hypot(x - center, y - center)
    mountain_height_m = 2500.0
    heightmap = (300.0 + mountain_height_m * np.exp(-(dist ** 2) / (2 * (size * 0.12) ** 2))).astype(np.float32)
    shadowmap = np.full((size, size, 7), 0.6, dtype=np.float32)

    dlm = DataLODManager()
    dlm.set_map_distance_km(20.0)
    geo = WeatherSystemGenerator(map_seed=15, data_lod_manager=dlm)
    params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': 2.5,
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': 0.0,
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 45.0,
        'map_longitude': 0.0,
    }
    try:
        result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised: {e}")
        traceback.print_exc()
        return False

    center_idx = result.temp_map.shape[0] // 2
    mountain_temp = float(result.temp_map[center_idx, center_idx])
    corner_temp = float(result.temp_map[2, 2])  # weit vom Berg entfernt, Tiefland
    expected_min_diff = (mountain_height_m - 300.0) * (WEATHER.ALTITUDE_COOLING["default"] / 1000.0) * 0.3

    ok = check("weather_data.is_valid()", result.is_valid())
    ok &= check(f"Berg-Gipfel ({mountain_temp:.1f}C) deutlich kaelter als Tiefland "
                f"({corner_temp:.1f}C), erwartete Mindest-Differenz {expected_min_diff:.1f}C "
                f"(Wind weht ueber den Berg, Advektion darf die Hoehen-Abkuehlung "
                f"nicht mehr 'wegtragen')",
                (corner_temp - mountain_temp) > expected_min_diff)
    return ok


def run_lod_inheritance_same_resolution_temperature_stability():
    """Nutzer-Bug-Report 2026-07-24: "alles wird viel zu kalt" ueber mehrere
    LOD-Runden. Root Cause gefunden: initial_state['temp_layers'] (bereits
    zurueckgeschnitten auf die UNGEPOLSTERTE target_size der Vorstufe) wurde
    bicubisch direkt auf die GEPOLSTERTE Arbeitsgittergroesse (height/width)
    dieser Runde hochskaliert statt zuerst auf die eigene ungepolsterte
    target_size UND DANACH mit demselben Rand-Puffer wie heightmap/
    shadowmap gepolstert zu werden - jeder geerbte Temperatur-Pixel landete
    dadurch an einer verschobenen Position relativ zur aktuellen (korrekt
    positionierten) heightmap, wodurch sich der Hoehen-Anteil beim
    theta<->t_real-Roundtrip nicht mehr sauber aufhob (akkumulierender
    Fehler pro LOD-Runde). Test: zwei GLEICH GROSSE LOD-Stufen in Folge
    (LOD3->LOD4, beide 128x128 bei diesem map_size) - eine Berg-Spitze darf
    sich zwischen zwei Runden OHNE Aufloesungsaenderung nicht mehr nennenswert
    veraendern (vorher: ~8C Drift selbst bei identischer Gittergroesse)."""
    size = 128
    rng = np.random.RandomState(11)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    nx, ny = x / (size - 1), y / (size - 1)
    heightmap = (300.0 + 2800.0 * np.exp(-(((nx - 0.4) ** 2 + (ny - 0.4) ** 2) / (2 * 0.09 ** 2)))
                 + 30.0 * rng.randn(size, size)).astype(np.float32)
    shadowmap = np.full((size, size, 7), 0.6, dtype=np.float32)

    dlm = DataLODManager()
    dlm.set_map_distance_km(20.0)
    geo = WeatherSystemGenerator(map_seed=13, data_lod_manager=dlm)
    params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 45.0,
        'map_longitude': 15.0,
    }

    peak_y, peak_x = int(0.4 * (size - 1)), int(0.4 * (size - 1))
    try:
        result_lod3 = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=3)
        result_lod4 = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=4)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised: {e}")
        traceback.print_exc()
        return False

    peak_temp_lod3 = float(result_lod3.temp_map[peak_y, peak_x])
    peak_temp_lod4 = float(result_lod4.temp_map[peak_y, peak_x])
    drift = abs(peak_temp_lod4 - peak_temp_lod3)

    ok = check("beide Ergebnisse gueltig", result_lod3.is_valid() and result_lod4.is_valid())
    ok &= check(f"Bergspitzen-Temperatur bleibt bei gleicher Aufloesung stabil "
                f"(LOD3={peak_temp_lod3:.1f}C, LOD4={peak_temp_lod4:.1f}C, Drift={drift:.1f}C, erwartet < 5C)",
                drift < 5.0)
    return ok


def run_water_mass_balance():
    """Weather-Rework Punkt H: Wasserbilanz. Kondensation muss die kondensierte
    Menge exakt AUS q entfernen, wenn sie in precip_accum einfliesst (siehe
    core/weather_generator.py Zeilen um 1174/1181/1218 - HIGH-Layer-Leck war
    frueher in dieser Session bereits gefunden und gefixt), UND der Rand-
    Puffer/Sponge-Layer (Punkt C) darf beim Zurueckschneiden auf die Original-
    groesse kein Wasser erzeugen/vernichten. Ohne Verdunstung
    (soil_moisture_field=0) darf die Gesamt-Feuchte (Summe humid_layers +
    precip_map, jeweils auf der zurueckgeschnittenen Kartenflaeche) NUR durch
    Kondensation von q nach precip_accum umverteilt werden, nicht aus dem
    Nichts entstehen oder verschwinden - Summe VORHER (q) muss ungefaehr
    gleich Summe NACHHER (q + precip) sein (semi-Lagrange-Advektion/DCT-
    Projektion sind nicht exakt massenerhaltend in einem berandeten - nicht
    periodischen - Gitter, daher Toleranz statt exakter Gleichheit)."""
    size = 32
    rng = np.random.RandomState(4)
    heightmap = (300.0 + 40.0 * rng.randn(size, size)).astype(np.float32)
    shadowmap = np.full((size, size, 7), 0.5, dtype=np.float32)

    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    geo = WeatherSystemGenerator(map_seed=13, data_lod_manager=dlm)

    base_params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 20.0,
        'map_longitude': 0.0,
    }
    month_params = geo._generate_seasonal_parameters(base_params, 0)

    initial_humid_layers = np.stack([
        np.full((size, size), 50.0, dtype=np.float32),
        np.full((size, size), 20.0, dtype=np.float32),
        np.full((size, size), 5.0, dtype=np.float32),
    ], axis=0)
    initial_state = {
        'temp_layers': np.stack([np.full((size, size), 15.0, dtype=np.float32) for _ in range(3)], axis=0),
        'wind_layers': np.stack([np.full((size, size, 2), 1.5, dtype=np.float32) for _ in range(3)], axis=0),
        'humid_layers': initial_humid_layers,
    }
    zero_soil_moisture = np.zeros((size, size), dtype=np.float32)

    try:
        result = geo._run_coupled_atmosphere_simulation(
            heightmap, shadowmap, month_params, size, n_steps=15,
            initial_state=initial_state, soil_moisture_field=zero_soil_moisture)
    except Exception as e:
        print(f"[FAIL] _run_coupled_atmosphere_simulation raised: {e}")
        traceback.print_exc()
        return False

    ok = check("humid_layers endlich", bool(np.all(np.isfinite(result['humid_layers']))))
    ok &= check("precip_map endlich und >= 0", bool(np.all(np.isfinite(result['precip_map']))
                                                    and np.all(result['precip_map'] >= 0)))

    initial_total = float(initial_humid_layers.sum())
    final_total = float(result['humid_layers'].sum() + result['precip_map'].sum())
    rel_diff = abs(final_total - initial_total) / max(initial_total, 1e-9)
    ok &= check(f"Gesamt-Feuchte vorher ({initial_total:.1f}) vs. nachher inkl. Niederschlag "
                f"({final_total:.1f}) - relative Abweichung {rel_diff * 100:.2f}%, erwartet < 15% "
                f"(keine grobe Masseerzeugung/-vernichtung ohne Verdunstung/Rand-Puffer)",
                rel_diff < 0.15)
    return ok


def run_end_to_end():
    """calculate_weather_system() muss nach den Loop-Aenderungen (HIGH-Layer-
    Kondensation, Klimatologie) weiterhin fehlerfrei durchlaufen und plausible
    Werte liefern."""
    size = 64
    rng = np.random.RandomState(3)
    x = np.linspace(0, 4, size)
    y = np.linspace(0, 4, size)
    X, Y = np.meshgrid(x, y)
    heightmap = (800 + 600 * np.sin(X) * np.cos(Y) + 100 * rng.randn(size, size)).astype(np.float32)
    shadowmap = np.clip(0.5 + 0.3 * rng.randn(size, size, 7), 0.0, 1.0).astype(np.float32)

    dlm = DataLODManager()
    dlm.set_map_distance_km(10.0)
    geo = WeatherSystemGenerator(map_seed=5, data_lod_manager=dlm)

    params = {
        'air_temp_entry': WEATHER.AIR_TEMP_ENTRY["default"],
        'ground_temp_offset': WEATHER.GROUND_TEMP_OFFSET["default"],
        'altitude_cooling': WEATHER.ALTITUDE_COOLING["default"],
        'thermic_effect': WEATHER.THERMIC_EFFECT["default"],
        'wind_speed_factor': WEATHER.WIND_SPEED_FACTOR["default"],
        'terrain_factor': WEATHER.TERRAIN_FACTOR["default"],
        'prevailing_wind_direction': WEATHER.PREVAILING_WIND_DIRECTION["default"],
        'air_humidity_entry': WEATHER.AIR_HUMIDITY_ENTRY["default"],
        'map_latitude': 48.0,
        'map_longitude': 15.0,
    }

    try:
        result = geo.calculate_weather_system(heightmap, shadowmap, params, lod_level=1)
    except Exception as e:
        print(f"[FAIL] calculate_weather_system raised: {e}")
        traceback.print_exc()
        return False

    ok = check("weather_data.is_valid()", result.is_valid())
    ok &= check("temp_map finite", bool(np.all(np.isfinite(result.temp_map))))
    ok &= check("precip_map finite and >= 0", bool(np.all(np.isfinite(result.precip_map)) and
                                                    np.all(result.precip_map >= 0)))
    ok &= check("humid_map finite and >= 0", bool(np.all(np.isfinite(result.humid_map)) and
                                                   np.all(result.humid_map >= 0)))
    ok &= check("wind_map finite", bool(np.all(np.isfinite(result.wind_map))))
    ok &= check("temp_map plausible range [-60,60]",
                bool(np.all(result.temp_map >= -60) and np.all(result.temp_map <= 60)))

    # Layer-Diagnose (weather_tab.py Ground/Mid/High-Umschalter, siehe
    # set_weather_data_complete_lod()) - muss nach der Speicherung über
    # get_weather_data("*_layers") abrufbar sein, sonst fällt die GUI
    # stillschweigend auf die Ground-Ebene zurück.
    dlm.set_weather_data_complete_lod(result, 1, params)
    temp_layers = dlm.get_weather_data("temp_map_layers")
    wind_layers = dlm.get_weather_data("wind_map_layers")
    humid_layers = dlm.get_weather_data("humid_map_layers")
    # LOD 1 rechnet auf seiner eigenen (kleineren) Standardgröße statt der
    # rohen Eingabe-Heightmap-Größe - Formkontrolle daher gegen temp_map
    # (bereits oben als gültig geprüft), nicht gegen die feste Eingabegröße.
    grid_h, grid_w = result.temp_map.shape
    ok &= check("temp_map_layers retrievable with shape (3,H,W)",
                temp_layers is not None and temp_layers.shape == (3, grid_h, grid_w))
    ok &= check("wind_map_layers retrievable with shape (3,H,W,2)",
                wind_layers is not None and wind_layers.shape == (3, grid_h, grid_w, 2))
    ok &= check("humid_map_layers retrievable with shape (3,H,W)",
                humid_layers is not None and humid_layers.shape == (3, grid_h, grid_w))

    # Monats-Animation für Mid/High (weather_tab.py _current_monthly_key()) -
    # *_layers_monthly muss als Liste von 6 (3,H,W)-Arrays abrufbar sein,
    # sonst zeigt die GUI für Mid/High nur einen statischen Wert statt der
    # 6-Monats-Animation wie bei Ground.
    temp_layers_monthly = dlm.get_weather_data("temp_map_layers_monthly")
    wind_layers_monthly = dlm.get_weather_data("wind_map_layers_monthly")
    humid_layers_monthly = dlm.get_weather_data("humid_map_layers_monthly")
    ok &= check("temp_map_layers_monthly: 6 Monate, je (3,H,W)",
                bool(temp_layers_monthly) and len(temp_layers_monthly) == 6
                and temp_layers_monthly[0].shape == (3, grid_h, grid_w))
    ok &= check("wind_map_layers_monthly: 6 Monate, je (3,H,W,2)",
                bool(wind_layers_monthly) and len(wind_layers_monthly) == 6
                and wind_layers_monthly[0].shape == (3, grid_h, grid_w, 2))
    ok &= check("humid_map_layers_monthly: 6 Monate, je (3,H,W)",
                bool(humid_layers_monthly) and len(humid_layers_monthly) == 6
                and humid_layers_monthly[0].shape == (3, grid_h, grid_w))
    return ok


if __name__ == "__main__":
    results = {
        "edge_padding_sanity": run_edge_padding_sanity(),
        "climatology_sanity": run_climatology_sanity(),
        "precip_annual_rescale": run_precip_annual_rescale(),
        "climatology_reference_table_match": run_climatology_reference_table_match(),
        "baroclinic_wind_scaling": run_baroclinic_wind_scaling(),
        "slope_aspect_solar_heating": run_slope_aspect_solar_heating(),
        "thermal_pressure_toggle": run_thermal_pressure_toggle(),
        "lod_inheritance_path_dependence": run_lod_inheritance_path_dependence(),
        "soil_moisture_coupling": run_soil_moisture_coupling(),
        "water_mass_balance": run_water_mass_balance(),
        "altitude_lapse_persists_through_advection": run_altitude_lapse_persists_through_advection(),
        "lod_inheritance_same_resolution_temperature_stability":
            run_lod_inheritance_same_resolution_temperature_stability(),
        "ground_temp_climatology_tracking": run_ground_temp_climatology_tracking(),
        "wind_dependent_ground_equilibration": run_wind_dependent_ground_equilibration(),
        "airmass_dampening": run_airmass_dampening(),
        "biome_solar_absorption": run_biome_solar_absorption(),
        "end_to_end": run_end_to_end(),
    }
    print("\n=== SUMMARY ===")
    overall = True
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        overall &= ok
    sys.exit(0 if overall else 1)
