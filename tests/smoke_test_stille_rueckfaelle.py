"""
Path: tests/smoke_test_stille_rueckfaelle.py

Ticket #54 ("Stille Ruckfaelle laut machen", Nachtlauf 2026-09-21/22).

DAS MUSTER, GEGEN DAS DIESER TEST WACHT: ein except-Block (oder eine
if/else-Weiche, ein .get(key, default)) faengt einen Fehler ab und liefert
einen Ersatzwert, der wie ein echtes Ergebnis aussieht - ohne Logzeile,
ohne Warnung. Das ist in diesem Projekt bereits SIEBENMAL passiert (siehe
CLAUDE.md: Shader-Pfade nach Dateiumzug, adaptives 3D-Netz mit falscher
Vorbedingung, GPU-Dispatch in der Geologie, Luecken in impact_matrix,
Karten ohne Farbtabelle im 3D, hasattr-Weichen die nie trafen, ein falscher
Erosions-Befund im Testbericht) - siebenmal unbemerkt ueber Tage bis Wochen.

WAS DIESER TEST PRUEFT: er durchsucht eine feste Liste von Hotspot-Dateien
(HOTSPOT_DATEIEN unten - die Dateien, in denen die sieben Faelle oben und
die beim Nachtlauf #54 gefundenen Faelle tatsaechlich lagen) per AST nach
except-Bloecken ohne laute Meldung (kein logging.warning/error/critical/
exception, kein self.rendering_error.emit, kein print, kein raise).

Jeder so gefundene stille Block muss NAMENTLICH in ALLOWLIST stehen, mit
Begruendung, warum er bewusst ohne eigene Logzeile bleibt (z.B. echtes
Aufraeum-/Cleanup-Cleanup, erwarteter Zustand, oder die Meldung passiert
bereits an anderer Stelle). Taucht ein NEUER stiller except-Block in einer
dieser Dateien auf, schlaegt test_keine_neuen_stillen_excepts fehl - analog
zu FEHLT_IM_3D in smoke_test_display_methoden_existieren.py, das auch nur
mit Begruendung wachsen darf.

AUSDRUECKLICH NICHT DAS ZIEL: Vollstaendigkeit ueber das gesamte Projekt.
HOTSPOT_DATEIEN ist eine kuratierte Liste (siehe Begruendung je Eintrag im
Nachtlauf-Commit #54), keine vollstaendige Untersuchung von core/, gui/ und
managers/. Insbesondere NICHT durchsucht: .get(key, default)-Faelle in
core/*.py ausserhalb der hier gelisteten Dateien, und hasattr-Weichen (die
hat bereits smoke_test_display_methoden_existieren.py als eigenes,
spezialisiertes Werkzeug). Eine ehrlich begrenzte, aber verlaesslich
wiederholbare erste Runde ist hier bewusst einer behaupteten, aber nicht
zutreffenden Vollstaendigkeit vorgezogen (CLAUDE.md).
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, ".")

PROJEKT_ROOT = Path(__file__).resolve().parent.parent

HOTSPOT_DATEIEN = [
    "core/erosion_generator.py",
    "managers/shader_manager.py",
    "gui/widgets/map_display_3d.py",
    "core/wegsuche_schnell.py",
    "core/biome_generator.py",
    "core/weather_generator.py",
    "core/geology_generator.py",
    "core/settlement_generator.py",
    "gui/tabs/kontinent_tab.py",
    "gui/tabs/river_tab.py",
    "gui/tabs/base_tab.py",
    "gui/tabs/settlement_tab.py",
    "gui/tabs/settlement_regional_tab.py",
    "gui/widgets/widgets.py",
]

_LAUTE_ATTRIBUTE = {"warning", "error", "critical", "exception", "emit", "debug", "info"}

# (Datei relativ zum Projekt-Root, Zeilennummer des `except`) -> Begruendung.
# Zeilennummern gehoeren zum Stand des Nachtlaufs #54 (2026-09-21/22) -
# verschiebt eine spaetere Aenderung die Zeile, meldet
# test_allowlist_eintraege_zeigen_noch_auf_stille_stellen das, und die
# Zeile muss hier nachgezogen werden (oder der Eintrag entfaellt, wenn der
# Block inzwischen laut geworden ist).
ALLOWLIST = {
    ("managers/shader_manager.py", 122):
        "os.path-Scan beim Modulimport, sammelt Int-Uniform-Namen aus den "
        "Shader-Dateien; OSError bei einer einzelnen nicht lesbaren Datei "
        "wird uebersprungen. Betrifft nur Autocomplete-Metadaten fuer "
        "glUniform-Aufrufe, keinen Rechenpfad.",
    ("managers/shader_manager.py", 1817):
        "queue.Empty in der Polling-Schleife des GPUWorker ist der "
        "Normalfall (noch keine Antwort da), kein Fehler.",
    ("managers/shader_manager.py", 1836):
        "Falsch-Positiv der AST-Heuristik: die Exception wird in "
        "result_box['error'] abgelegt und von submit() unmittelbar per "
        "'raise result_box[\"error\"]' an den Aufrufer weitergereicht - "
        "die Heuristik prueft nur den except-Block selbst, nicht dessen "
        "Verwendung eine Ebene hoeher. Tatsaechlich nicht still.",
    ("gui/widgets/map_display_3d.py", 1351):
        "ImportError -> Feature-Flag-Default KUESTEN_SCHNITT_AKTIV = False; "
        "steuert nur eine optionale Zusatzdarstellung, kein Rechenergebnis "
        "und keine der neun RUECKFAELLE-Marken aus "
        "nachtbetrieb/morgenbericht.py.",
    ("gui/widgets/map_display_3d.py", 1567):
        "_cleanup_wegband_gl_buffer: GL-Puffer-Teardown beim Schliessen "
        "des Widgets - ein Fehler hier ist im schlimmsten Fall ein "
        "VRAM-Leck beim Beenden, kein falsches Rechen- oder Anzeigeergebnis "
        "waehrend des Betriebs.",
    ("core/wegsuche_schnell.py", 64):
        "numba-Import schlaegt fehl -> NUMBA_DA = False (No-Op-Dekorator "
        "fuer @njit). Die tatsaechliche Konsequenz (A*-Wegsuche ohne JIT) "
        "wird an anderer Stelle in derselben Datei (~Zeile 229) bereits "
        "geloggt und ist eine der 9 RUECKFAELLE-Marken in "
        "nachtbetrieb/morgenbericht.py ('Wegsuche ohne numba').",
    ("core/geology_generator.py", 998):
        "ImportError von gui.config.value_default -> default_km = 10.0 "
        "als reiner Konstanten-Fallback fuer Standalone-/Testnutzung "
        "ausserhalb der GUI (z.B. Kopfzeilen-Skripte). Kein Betriebsfehler "
        "der laufenden App.",
    ("gui/tabs/base_tab.py", 68):
        "ImportError von gui.utils.error_handler -> No-Op-Dekorator statt "
        "des echten Fehler-Wrappers. Betrifft nur das (optionale) "
        "Fehler-UI-Framework selbst, nicht den damit geschuetzten Code, "
        "der bei einem echten Fehler weiterhin ueber seine eigenen "
        "try/except-Bloecke laut wird.",
    ("gui/tabs/base_tab.py", 1199):
        "TypeError/RuntimeError beim disconnect() eines Qt-Signals, das "
        "bereits getrennt ist (Kommentar im Code: 'Signal bereits "
        "disconnected') - das ist der erwartete, dokumentierte Zustand, "
        "kein Fehler.",
    ("gui/tabs/settlement_tab.py", 31):
        "Gleiches Muster wie gui/tabs/base_tab.py:68 (No-Op-Dekorator bei "
        "fehlendem error_handler-Import), gleiche Begruendung.",
    ("gui/widgets/widgets.py", 321):
        "ValueError bei ungueltiger Zahleneingabe im Eingabefeld -> "
        "setValue() faellt auf den letzten gueltigen Wert zurueck. "
        "Standard-Validierungsmuster: der Nutzer sieht die Ablehnung "
        "direkt und sofort am Feld selbst, keine versteckte Ersatzrechnung.",
}


def _ist_laut(handler: ast.ExceptHandler) -> bool:
    """True, wenn irgendwo im except-Block eine laute Meldung passiert:
    ein Logging-Aufruf, ein print(), oder ein re-raise."""
    for node in ast.walk(handler):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _LAUTE_ATTRIBUTE:
                return True
            if isinstance(func, ast.Name) and func.id == "print":
                return True
        if isinstance(node, ast.Raise):
            return True
    return False


def _stille_stellen(relativ: str):
    pfad = PROJEKT_ROOT / relativ
    baum = ast.parse(pfad.read_text(encoding="utf-8"), filename=relativ)
    return [
        (relativ, node.lineno)
        for node in ast.walk(baum)
        if isinstance(node, ast.ExceptHandler) and not _ist_laut(node)
    ]


def run():
    """Kernpruefung: jeder in den Hotspot-Dateien gefundene stille
    except-Block muss in ALLOWLIST stehen. Ein neuer, nicht gelisteter
    Fund lässt den Test fehlschlagen."""
    gefunden = []
    durchsuchte_zeilen = 0
    ok = True

    for relativ in HOTSPOT_DATEIEN:
        pfad = PROJEKT_ROOT / relativ
        if not pfad.exists():
            print(f"[FAIL] Hotspot-Datei fehlt: {relativ}")
            ok = False
            continue
        durchsuchte_zeilen += len(pfad.read_text(encoding="utf-8").splitlines())
        gefunden.extend(_stille_stellen(relativ))

    unbekannt = [ort for ort in gefunden if ort not in ALLOWLIST]
    if unbekannt:
        ok = False
        for datei, zeile in unbekannt:
            print(f"[FAIL] neuer stiller except-Block ohne Logzeile: "
                  f"{datei}:{zeile} - entweder eine laute Meldung ergaenzen "
                  f"(logging.warning/error, self.rendering_error.emit, "
                  f"print oder raise) oder, falls tatsaechlich harmlos, mit "
                  f"Begruendung in ALLOWLIST eintragen.")

    if ok:
        print(f"[OK] {len(HOTSPOT_DATEIEN)} Hotspot-Dateien, "
              f"{durchsuchte_zeilen} Zeilen durchsucht, "
              f"{len(gefunden)} still gebliebene except-Bloecke, alle in "
              f"ALLOWLIST mit Begruendung erfasst.")
    return ok


def run_allowlist_eintraege_zeigen_noch_auf_stille_stellen():
    """Schuetzt gegen eine ALLOWLIST, die durch Zeilenverschiebung heimlich
    veraltet: jeder Eintrag muss auf einen tatsaechlich noch vorhandenen,
    weiterhin stillen except-Block zeigen - sonst pruefen wir etwas, das es
    an der Stelle gar nicht mehr gibt."""
    alle_stellen = set()
    for relativ in HOTSPOT_DATEIEN:
        pfad = PROJEKT_ROOT / relativ
        if pfad.exists():
            alle_stellen.update(_stille_stellen(relativ))

    veraltet = [ort for ort in ALLOWLIST if ort not in alle_stellen]
    if veraltet:
        for datei, zeile in veraltet:
            print(f"[FAIL] ALLOWLIST-Eintrag zeigt auf keine stille Stelle "
                  f"mehr: {datei}:{zeile} (Zeile verschoben, Datei "
                  f"geaendert, oder Block ist inzwischen laut) - "
                  f"Zeilennummer nachziehen oder Eintrag entfernen.")
        return False

    print(f"[OK] alle {len(ALLOWLIST)} ALLOWLIST-Eintraege zeigen noch auf "
          f"tatsaechlich stille except-Bloecke.")
    return True


if __name__ == "__main__":
    ergebnisse = {
        "keine_neuen_stillen_excepts": run(),
        "allowlist_eintraege_aktuell": run_allowlist_eintraege_zeigen_noch_auf_stille_stellen(),
    }
    print("\n=== SUMMARY ===")
    for name, bestanden in ergebnisse.items():
        print(f"{name}: {'PASS' if bestanden else 'FAIL'}")
    sys.exit(0 if all(ergebnisse.values()) else 1)
