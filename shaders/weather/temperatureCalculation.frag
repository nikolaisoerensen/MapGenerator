#version 300 es
precision highp float;

uniform sampler2D heightmapTex;
uniform sampler2D shademapTex;
uniform vec2 resolution;
uniform float time;

// Parameter
uniform float air_temp_entry;      // Eingangstemperatur
uniform float solar_power;         // Max. solare Gewinne (20°C)
uniform float altitude_cooling;    // Abkühlung pro 100m (6°C)

in vec2 texCoord;
in vec2 fragCoord;

out float temperature;

// OpenSimplex Noise Funktion
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float opensimplex(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void main()
{
    // Höhenabhängige Temperatur
    float height = texture(heightmapTex, texCoord).r * 1000.0; // Höhe in Metern
    float temp_altitude = air_temp_entry - (height / 100.0) * altitude_cooling;

    // Solare Erwärmung
    float shade_factor = texture(shademapTex, texCoord).r;
    float temp_solar = shade_factor * solar_power;

    // Breitengrad-Effekt (Y-Position)
    float latitude_factor = texCoord.y; // 0 = Süd, 1 = Nord
    float temp_latitude = latitude_factor * 5.0; // 5°C Unterschied Nord-Süd

    // Noise für natürliche Variationen an Kartenrändern
    vec2 noise_coord = fragCoord * 0.01 + time * 0.1;
    float temp_noise = opensimplex(noise_coord) * 3.0; // ±3°C Variation

    // Stärkere Variation an Kartenrändern
    float edge_distance = min(min(texCoord.x, 1.0 - texCoord.x), min(texCoord.y, 1.0 - texCoord.y));
    float edge_factor = 1.0 - smoothstep(0.0, 0.1, edge_distance);
    temp_noise *= (1.0 + edge_factor * 2.0);

    // Finale Temperatur
    temperature = temp_altitude + temp_solar + temp_latitude + temp_noise;
}