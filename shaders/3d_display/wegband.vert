#version 330 core

// Wegband-Shader (docs/OFFENE_PUNKTE.md 6.28, Nutzerfeedback 2026-08-16:
// die Baender sahen mit dem bisherigen wind_shader_program - komplett unlit,
// eine Uniform-Farbe - "fake" aus und passten nicht zur Terrain-Beleuchtung.
// Eigenes kleines Paar statt den geteilten wind_vector-Shader zu erweitern:
// der bleibt fuer die Windpfeile unveraendert (siehe wegband.frag-Kopf).

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
// Deckung quer zur Fahrtrichtung: 1 auf der Fahrbahn, 0 an den Kanten.
// Gebaut in wege_geometrie.band_aus_pfad(); die lineare Interpolation
// zwischen Schulter und Kante ERZEUGT den weichen Rand von allein.
layout (location = 2) in float deckung;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;
out float Deckung;

void main() {
    // Modelmatrix ist in dieser Anzeige immer Identitaet (siehe
    // _update_model_matrix()) - Position/Normale koennen deshalb ohne
    // Normalenmatrix (inverse Transponierte) direkt mitgegeben werden, ohne
    // bei einer nicht-uniformen Skalierung zu verzerren.
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(model) * normal;
    Deckung = deckung;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
