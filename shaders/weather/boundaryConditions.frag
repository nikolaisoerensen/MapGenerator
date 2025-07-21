#version 300 es
precision highp float;

uniform sampler2D weatherTex;
uniform vec2 resolution;
uniform float time;

// Eingabeparameter
uniform float air_temp_entry;
uniform float wind_speed_entry;
uniform float humidity_entry;

in vec2 texCoord;
in vec2 fragCoord;

out vec4 boundaryWeather;

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
    vec4 current_weather = texture(weatherTex, texCoord);
    vec4 result = current_weather;

    // Randdistanz berechnen
    float edge_distance = min(min(texCoord.x, 1.0 - texCoord.x), min(texCoord.y, 1.0 - texCoord.y));
    float boundary_zone = 0.05; // 5% der Karte als Randzone

    if (edge_distance < boundary_zone) {
        float boundary_strength = 1.0 - (edge_distance / boundary_zone);

        // Wetterfront-Variationen mit Noise
        vec2 noise_coord = fragCoord * 0.02 + time * 0.03;
        float temp_variation = opensimplex(noise_coord) * 5.0; // ±5°C
        float wind_variation = opensimplex(noise_coord + vec2(100.0)) * 3.0; // ±3 m/s
        float humidity_variation = opensimplex(noise_coord + vec2(200.0)) * 2.0; // ±2 g/m³

        // Eingangswerte mit Variationen
        float boundary_temp = air_temp_entry + temp_variation;
        float boundary_wind_x = wind_speed_entry + wind_variation;
        float boundary_humidity = humidity_entry + humidity_variation;

        // Westrand: Wind-Eintritt
        if (texCoord.x < boundary_zone) {
            result.b = mix(current_weather.b, boundary_wind_x, boundary_strength);
            result.r = mix(current_weather.r, boundary_temp, boundary_strength * 0.7);
            result.g = mix(current_weather.g, boundary_humidity, boundary_strength * 0.8);
        }

        // Andere Ränder: Weiche Übergänge
        else {
            result.r = mix(current_weather.r, boundary_temp, boundary_strength * 0.3);
            result.g = mix(current_weather.g, boundary_humidity, boundary_strength * 0.2);
        }

        // Y-Wind an Nord/Süd-Rändern
        if (texCoord.y < boundary_zone || texCoord.y > 1.0 - boundary_zone) {
            float boundary_wind_y = wind_variation * 0.5;
            result.a = mix(current_weather.a, boundary_wind_y, boundary_strength * 0.4);
        }
    }

    boundaryWeather = result;
}