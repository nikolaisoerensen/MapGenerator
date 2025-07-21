#version 300 es
precision highp float;

uniform sampler2D moistureTex;
uniform sampler2D temperatureTex;
uniform sampler2D windTex;
uniform sampler2D heightmapTex;
uniform vec2 texelSize;

in vec2 texCoord;
in vec2 texCoordXm, texCoordXp, texCoordYm, texCoordYp;

out vec4 precipData; // .r = Niederschlag, .g = Kondensationswärme, .b = neue Feuchtigkeit, .a = neue Temperatur

void main()
{
    float moisture = texture(moistureTex, texCoord).r;
    float temperature = texture(temperatureTex, texCoord).r;
    vec2 wind = texture(windTex, texCoord).xy;

    // Maximale Wasserdampfdichte nach Formel
    float rho_max = 5.0 * exp(0.06 * temperature);

    // Relative Luftfeuchtigkeit
    float relative_humidity = moisture / rho_max;

    // Orographische Hebung berechnen
    float height_here = texture(heightmapTex, texCoord).r;
    float height_xm = texture(heightmapTex, texCoordXm).r;
    float height_xp = texture(heightmapTex, texCoordXp).r;
    float height_ym = texture(heightmapTex, texCoordYm).r;
    float height_yp = texture(heightmapTex, texCoordYp).r;

    vec2 height_gradient = vec2(
        (height_xp - height_xm) / (2.0 * texelSize.x),
        (height_yp - height_ym) / (2.0 * texelSize.y)
    );

    // Vertikale Windkomponente durch orographische Hebung
    float orographic_lift = dot(wind, height_gradient) * 1000.0; // m/s

    // Adiabatische Abkühlung bei Hebung (6.5°C/km)
    float cooling_rate = orographic_lift * 0.0065; // °C/s
    float cooled_temp = temperature - cooling_rate;

    // Neue maximale Wasserdampfdichte bei gekühlter Temperatur
    float rho_max_cooled = 5.0 * exp(0.06 * cooled_temp);

    // Niederschlag wenn Sättigung überschritten
    float precipitation = 0.0;
    float latent_heat = 0.0;
    float new_moisture = moisture;
    float new_temperature = temperature;

    if (moisture > rho_max_cooled && relative_humidity > 1.0) {
        // Kondensation
        precipitation = moisture - rho_max_cooled;
        new_moisture = rho_max_cooled;

        // Latente Wärmefreisetzung (2260 kJ/kg)
        latent_heat = precipitation * 2.26; // Vereinfacht
        new_temperature = temperature + latent_heat * 0.001; // °C
    }

    precipData = vec4(precipitation, latent_heat, new_moisture, new_temperature);
}