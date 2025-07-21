#version 300 es
precision highp float;

uniform sampler2D windTex;
uniform sampler2D temperatureTex;
uniform sampler2D moistureTex;
uniform sampler2D heightmapTex;
uniform vec2 texelSize;

in vec2 texCoord;
in vec2 texCoordXm, texCoordXp, texCoordYm, texCoordYp;

out vec4 orographicData; // .r = neue Temperatur, .g = neue Feuchtigkeit, .ba = Windmodifikation

void main()
{
    vec2 wind = texture(windTex, texCoord).xy;
    float temperature = texture(temperatureTex, texCoord).r;
    float moisture = texture(moistureTex, texCoord).r;

    // Geländeanalyse
    float height_here = texture(heightmapTex, texCoord).r;
    float height_xm = texture(heightmapTex, texCoordXm).r;
    float height_xp = texture(heightmapTex, texCoordXp).r;
    float height_ym = texture(heightmapTex, texCoordYm).r;
    float height_yp = texture(heightmapTex, texCoordYp).r;

    vec2 height_gradient = vec2(
        (height_xp - height_xm) / (2.0 * texelSize.x),
        (height_yp - height_ym) / (2.0 * texelSize.y)
    );

    // Bestimme Luv-/Lee-Seite
    float wind_terrain_dot = dot(normalize(wind), normalize(height_gradient));
    bool is_windward = wind_terrain_dot > 0.0; // Luv-Seite

    float new_temperature = temperature;
    float new_moisture = moisture;
    vec2 wind_modification = vec2(0.0);

    if (is_windward) {
        // LUV-SEITE: Staueffekte
        // Verstärkte Abkühlung durch erzwungenen Aufstieg
        new_temperature -= abs(wind_terrain_dot) * 2.0;

        // Verstärkte Kondensation
        float condensation_boost = abs(wind_terrain_dot) * 0.5;
        new_moisture *= (1.0 - condensation_boost);

        // Aufwinde verstärken
        wind_modification.y = abs(wind_terrain_dot) * 2.0;

    } else {
        // LEE-SEITE: Föhneffekte
        // Trockenadiabatische Erwärmung
        float lee_warming = abs(wind_terrain_dot) * 3.0;
        new_temperature += lee_warming;

        // Weitere Austrocknung
        float drying_factor = abs(wind_terrain_dot) * 0.3;
        new_moisture *= (1.0 - drying_factor);

        // Abwinde
        wind_modification.y = -abs(wind_terrain_dot) * 1.5;
    }

    // Regenschatten-Verstärkung
    float avg_upwind_height = 0.0;
    vec2 upwind_dir = normalize(-wind);

    // Sample mehrere Punkte in Windrichtung
    for(int i = 1; i <= 5; i++) {
        vec2 sample_pos = texCoord + upwind_dir * texelSize * float(i) * 2.0;
        if(sample_pos.x >= 0.0 && sample_pos.x <= 1.0 && sample_pos.y >= 0.0 && sample_pos.y <= 1.0) {
            avg_upwind_height += texture(heightmapTex, sample_pos).r;
        }
    }
    avg_upwind_height /= 5.0;

    // Regenschatten wenn höhere Berge in Windrichtung
    if(avg_upwind_height > height_here + 0.1) {
        float shadow_factor = (avg_upwind_height - height_here) * 2.0;
        new_moisture *= (1.0 - shadow_factor * 0.4);
        new_temperature += shadow_factor * 1.5; // Föhn-Erwärmung
    }

    orographicData = vec4(new_temperature, new_moisture, wind_modification);
}