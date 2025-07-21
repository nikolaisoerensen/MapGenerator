#version 300 es
precision highp float;

uniform sampler2D temperatureTex;
uniform sampler2D shademapTex;
uniform sampler2D windTex;
uniform vec2 texelSize;

// Parameter
uniform float thermic_effect;

in vec2 texCoord;
in vec2 texCoordXm, texCoordXp, texCoordYm, texCoordYp;

out vec2 thermalWind;

void main()
{
    // Aktuelle Temperatur und Wind
    float temp_here = texture(temperatureTex, texCoord).r;
    vec2 base_wind = texture(windTex, texCoord).xy;

    // Temperaturgradienten berechnen
    float temp_xm = texture(temperatureTex, texCoordXm).r;
    float temp_xp = texture(temperatureTex, texCoordXp).r;
    float temp_ym = texture(temperatureTex, texCoordYm).r;
    float temp_yp = texture(temperatureTex, texCoordYp).r;

    vec2 temp_gradient = vec2(
        (temp_xp - temp_xm) / (2.0 * texelSize.x),
        (temp_yp - temp_ym) / (2.0 * texelSize.y)
    );

    // Sonneneinstrahlung für lokale Aufheizung
    float shade_factor = texture(shademapTex, texCoord).r;

    // Thermische Aufwinde über warmen Bereichen
    float thermal_strength = (shade_factor - 0.5) * thermic_effect;

    // Horizontale thermische Winde (von kalt zu warm)
    vec2 thermal_wind = temp_gradient * thermic_effect * 0.1;

    // Hangwinde: Wind folgt Temperaturgradienten
    // Warme Hänge erzeugen Aufwinde, kalte Abwinde
    float slope_thermal = (temp_here - 15.0) * 0.01; // Referenz 15°C
    thermal_wind += vec2(0.0, slope_thermal) * thermic_effect;

    // Konvektive Zirkulation
    float convection_x = sin(temp_here * 0.1) * thermal_strength * 0.05;
    float convection_y = cos(temp_here * 0.1) * thermal_strength * 0.05;

    // Kombiniere mit Basiswind
    thermalWind = base_wind + thermal_wind + vec2(convection_x, convection_y);
}