#version 300 es
precision highp float;

uniform sampler2D windTex;
uniform sampler2D temperatureTex;
uniform sampler2D moistureTex;
uniform sampler2D precipitationTex;
uniform sampler2D orographicTex;
uniform sampler2D thermalWindTex;

uniform float dt;
uniform float feedback_strength;

in vec2 texCoord;

out vec4 integratedWeather; // .r = finale Temperatur, .g = finale Feuchtigkeit, .ba = finaler Wind

void main()
{
    // Alle Komponenten einlesen
    vec2 base_wind = texture(windTex, texCoord).xy;
    vec2 thermal_wind = texture(thermalWindTex, texCoord).xy;
    float base_temp = texture(temperatureTex, texCoord).r;
    float base_moisture = texture(moistureTex, texCoord).r;

    vec4 precip_data = texture(precipitationTex, texCoord);
    float precipitation = precip_data.r;
    float latent_heat = precip_data.g;
    float precip_moisture = precip_data.b;
    float precip_temp = precip_data.a;

    vec4 orographic_data = texture(orographicTex, texCoord);
    float oro_temp = orographic_data.r;
    float oro_moisture = orographic_data.g;
    vec2 oro_wind = orographic_data.ba;

    // Rückkopplungen berechnen

    // Temperatur: Basis + Latente Wärme + Orographische Effekte
    float final_temp = mix(mix(base_temp, precip_temp, 0.5), oro_temp, 0.3);
    final_temp += latent_heat * 0.1 * feedback_strength;

    // Feuchtigkeit: Transport + Kondensation + Orographische Effekte
    float final_moisture = mix(mix(base_moisture, precip_moisture, 0.7), oro_moisture, 0.5);
    final_moisture = max(0.0, final_moisture - precipitation);

    // Wind: Basis + Thermik + Orographische Ablenkung
    vec2 final_wind = base_wind + thermal_wind * 0.3 + oro_wind * 0.4;

    // Konvergenz/Divergenz-Effekte
    float temp_diff = final_temp - 15.0; // Referenztemperatur
    float convection_factor = temp_diff * 0.01;
    final_wind.y += convection_factor; // Vertikale Komponente durch Konvektion

    // Stabilisierung
    final_wind = clamp(final_wind, vec2(-50.0), vec2(50.0)); // Max 50 m/s
    final_temp = clamp(final_temp, -40.0, 60.0); // Realistische Grenzen
    final_moisture = clamp(final_moisture, 0.0, 100.0); // Max 100 g/m³

    integratedWeather = vec4(final_temp, final_moisture, final_wind);
}