# Prüfliste am laufenden Programm — Stand 2026-08-24

Alles hier ist **headless nicht prüfbar** und braucht deinen Blick. Was
automatisch geprüft werden konnte, ist geprüft (siehe `docs/SITZUNGSLOG.md`).

Start:

    .venv/Scripts/python.exe main.py

---

## 1. Wege in 3D — der Hauptpunkt dieser Sitzung

Karte generieren, **Settlement-Reiter**, Häkchen bei *Roads*, dann auf
**3D-View** umschalten.

| # | Was ansehen | Was richtig ist | Was schiefgehen kann |
|---|---|---|---|
| 1.1 | Kamera flach über den Boden legen und an einem Weg entlangschauen | Der Weg liegt **auf** dem Gelände | Bisher schwebte er sichtbar darüber — man konnte darunterschauen. Wenn er jetzt stattdessen flackert oder streckenweise verschwindet, ist der Tiefenversatz zu klein |
| 1.2 | Nah an einen Wegrand heranzoomen | Der Rand **läuft weich aus** statt an einer harten Kante zu enden | Sieht der Rand ausgefranst oder gepunktet aus, ist die Alpha-Schwelle falsch |
| 1.3 | Einen Weg von schräg oben ansehen | Die Fahrbahn wirkt **leicht gewölbt**, mit hellerer Mitte und dunkleren Schultern | Sieht sie aus wie eine Wurst, ist `NORMALEN_WOELBUNG` zu hoch; sieht sie flach aus, zu niedrig |
| 1.4 | Einen Weg an einem **Querhang** verfolgen | Er bleibt quer eben und versinkt nicht im Hang | |
| 1.5 | Einen Weg über eine **Kuppe** verfolgen | Er schneidet leicht ein, wie eine echte Trasse | Gemessen bis 0,7 m. Wenn das Gelände sichtbar durch den Weg stößt, ist es zu viel |
| 1.6 | Kamera drehen und schwenken | Das Band bleibt ruhig | Springen/Flackern beim Bewegen wäre derselbe Fehler wie beim gescheiterten Klippenband |
| 1.7 | Roads **mehrfach an- und ausschalten**, Konsole beobachten | Kein Ruckeln, keine wachsende Verzögerung | Die GL-Puffer werden jetzt wiederverwendet; wenn es langsamer statt schneller wird, greift die Freigabe nicht |

**Wenn in der Konsole `FEHLER: wegband-Shaderprogramm fehlt` steht:** die
Shader ließen sich nicht übersetzen. Bitte die Meldung schicken — das ist neu
und war vorher unsichtbar.

## 2. Umschalten der Overlays — soll spürbar schneller sein

Settlement-Reiter, **2D**. *Settlements*, *Landmarks*, *Roadsites*, *Roads*
mehrfach an- und ausschalten.

* **Erwartet:** deutlich flotter als vorher (gemessen 150 → 42 ms je
  Neuzeichnung, und es wird nur noch einmal statt zweimal gezeichnet).
* **Achten auf:** dass die Karte nach dem Umschalten **wirklich neu
  gezeichnet** ist. `draw_idle()` zeichnet verzögert — falls ein Overlay erst
  nach einer weiteren Mausbewegung erscheint, ist das der Punkt, an dem es
  hakt.

## 3. Wetter-Monatsanimation

Weather-Reiter, Temperature/Wind wählen, Monatsanimation läuft. **Auf einen
anderen Reiter wechseln, eine Minute arbeiten, zurückwechseln.**

* **Erwartet:** im Hintergrund wird nicht mehr gerechnet, und beim
  Zurückkommen läuft die Animation bei **demselben Monat** weiter, an dem du
  sie verlassen hast.
* **Achten auf:** dass sie überhaupt weiterläuft. Bleibt sie stehen, greift
  der Sichtbarkeits-Test zu hart.

## 4. Erosionsreiter

Erosion-Reiter öffnen.

* **Erwartet:** oben ein **gelber Hinweisstreifen**, dass die Erosionskette
  abgeschaltet ist und die Regler deshalb gesperrt sind.
* Das ist der Punkt, an dem bisher unklar war, ob die wirkungslosen Regler
  Absicht oder ein Fehler sind.

## 5. Export

Karte vollständig generieren, dann exportieren.

* **Erwartet:** alle PNG sind **2048 × 2048**, unabhängig von der eingestellten
  Kartengröße.
* Im Ordner liegen zusätzlich **`noise_damping_mask.png`** (schwarz an Wegen,
  Bauflächen und Ufern, weiß in der Wildnis) und **`vektor.json`**.
* `manifest.json` nennt `meter_pro_pixel`. **Bei 21,3 km sollten dort 10,4
  stehen.**
* In `vektor.json` unter `"fehlt"` steht ausdrücklich, dass **Flüsse nicht
  dabei sind** — das ist bekannt und begründet, kein Fehler.

## 6. Mesh-Werkstatt (aus der vorigen Sitzung, weiterhin unbestätigt)

    .venv/Scripts/python.exe tools/mesh_werkstatt.py

Vier Netzarten durchschalten, Regler bewegen, „Fehler messen" anhaken.
Delatin braucht ~20 s und zeigt währenddessen den Fortschritt.

## 7. Immer noch der größte offene Blocker: 3b.1

**Die gesamte Vektor-Küste ist visuell unbestätigt** — weder die neue
Küstenform noch der Meeresboden noch die Regionsübergänge. Das hängt an
nichts anderem als daran, dass du es dir einmal ansiehst.

---

## Was diese Sitzung NICHT geprüft hat

* Alles OpenGL — headless gibt es keinen GL-Kontext.
* Ob die Wegoptik dir **gefällt**. Die Messungen sagen, dass es technisch
  richtig liegt; ob es schön ist, sagen sie nicht.
