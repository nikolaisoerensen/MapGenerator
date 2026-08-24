#version 330 core

// Gegenstueck zu wegband.vert. Absichtlich EIGENES Paar statt den
// bestehenden wind_vector-Shader (shaders/3d_display/wind_vector.*) zu
// erweitern: der bleibt unveraendert fuer die Windpfeile im Wetter-Reiter
// zustaendig (dort ist unlit richtig - ein Pfeil-Glyph, kein Gelaendeobjekt).
//
// Beleuchtung bewusst dieselbe simple Formel wie terrain.frag (Ambient +
// Diffuse, keine Schattenkarte) - der Punkt ist, dass die Baender auf
// dieselbe Lichtrichtung reagieren wie das Gelaende darunter, nicht ein
// physikalisch vollstaendiges Modell. useAlpha macht aus dem Band einen
// Decal-artigen, leicht durchscheinenden Belag statt einer blickdichten
// Flaeche (Nutzerwunsch 2026-08-16: "so dass die unregelmaessige form sich
// schoen auf die textur schmiegt").

in vec3 FragPos;
in vec3 Normal;
in float Deckung;

uniform vec3 lightPos;
uniform vec3 wegFarbe;
uniform float wegAlpha;
// Roetliche Einfaerbung bei Auswahl (Nutzervorgabe: "werden die wege
// einfach roetlich oder so eingefaerbt statt realistisch auszusehen") -
// ein Uniform-Schalter statt einer zweiten Farbe je Vertex, weil die
// Auswahl sich pro Frame aendern kann, die Geometrie selbst aber nicht neu
// gebaut werden soll.
uniform bool ausgewaehlt;

out vec4 FragColor;

void main() {
    vec3 farbe = ausgewaehlt ? vec3(0.85, 0.15, 0.15) : wegFarbe;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);

    vec3 ambient = 0.45 * farbe;
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * farbe;

    // WEICHER RAND (2026-08-24). Vorher endete das Band an einer harten
    // Polygonkante - im Bild eine gerade Linie quer durchs Gelaende, die
    // nichts mit dem Untergrund zu tun hat. Jetzt laeuft die Deckung zu
    // den Kanten hin aus, das Band verlaeuft im Boden statt abzubrechen.
    float rand = clamp(Deckung, 0.0, 1.0);
    // Leicht angehoben (Wurzel), damit der sichtbare Belag nicht zu
    // schmal wirkt: linear faellt die Haelfte der Breite unter 50 %.
    rand = sqrt(rand);
    if (rand <= 0.002) discard;

    FragColor = vec4(ambient + diffuse, wegAlpha * rand);
}
