#version 300 es
precision highp float;

uniform sampler2D heightmapTex;
uniform sampler2D slopemapTex;
uniform vec2 resolution;
uniform float time;

// Parameter
uniform float wind_speed_factor;
uniform float terrain_factor;

in vec2 texCoord;
in vec2 texCoordXm, texCoordXp, texCoordYm, texCoordYp;
in vec2 fragCoord;

out vec2 windVelocity;

float opensimplex(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = fract(sin(dot(i, vec2(127.1, 311.7))) * 43758.5453);
    float b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7))) * 43758.5453);
    float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);
    float d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7))) * 43758.5453);

    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void main()
{
    // Basis-Druckgradient West -> Ost
    float pressure_gradient = (1.0 - texCoord.x) * 1000.0; // Pa/m

    // Noise-Modulation für natürliche Turbulenzen
    vec2 noise_coord = fragCoord * 0.005 + time * 0.05;
    float pressure_noise = opensimplex(noise_coord) * 200.0;
    pressure_gradient += pressure_noise;

    // Grundwind aus Druckgradient
    vec2 base_wind = vec2(pressure_gradient * wind_speed_factor, 0.0);

    // Geländeablenkung berechnen
    float height_here = texture(heightmapTex, texCoord).r;
    float height_xm = texture(heightmapTex, texCoordXm).r;
    float height_xp = texture(heightmapTex, texCoordXp).r;
    float height_ym = texture(heightmapTex, texCoordYm).r;
    float height_yp = texture(heightmapTex, texCoordYp).r;

    // Höhengradienten
    vec2 height_gradient = vec2(
        (height_xp - height_xm) * resolution.x * 0.5,
        (height_yp - height_ym) * resolution.y * 0.5
    );

    // Geländesteigung aus Slopemap
    vec2 slope = texture(slopemapTex, texCoord).xy - 0.5; // -0.5 bis 0.5

    // Wind-Ablenkung um Berge
    vec2 terrain_deflection = vec2(
        -height_gradient.y, // Wind weicht seitlich aus
        height_gradient.x
    ) * terrain_factor;

    // Abbremsung durch Gelände
    float terrain_resistance = length(slope) * terrain_factor * 0.5;
    float speed_reduction = 1.0 / (1.0 + terrain_resistance);

    // Düseneffekt in Tälern
    float valley_factor = 1.0;
    if (height_here < (height_xm + height_xp + height_ym + height_yp) * 0.25) {
        valley_factor = 1.0 + (1.0 - height_here) * 0.3; // Beschleunigung in Tälern
    }

    // Finale Windgeschwindigkeit
    windVelocity = (base_wind + terrain_deflection) * speed_reduction * valley_factor;
}