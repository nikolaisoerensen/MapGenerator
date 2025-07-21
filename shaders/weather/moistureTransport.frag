#version 300 es
precision highp float;

uniform sampler2D moistureTex;
uniform sampler2D soilMoistureTex;
uniform sampler2D temperatureTex;
uniform sampler2D windTex;
uniform vec2 texelSize;
uniform float dt; // Zeitschritt

in vec2 texCoord;
in vec2 texCoordXm, texCoordXp, texCoordYm, texCoordYp;

out float humidity;

void main()
{
    // Aktuelle Werte
    float moisture_here = texture(moistureTex, texCoord).r;
    float soil_moisture = texture(soilMoistureTex, texCoord).r;
    float temperature = texture(temperatureTex, texCoord).r;
    vec2 wind = texture(windTex, texCoord).xy;

    // Evaporation von Bodenfeuchte
    float evap_rate = soil_moisture * 0.001 * max(0.0, temperature - 5.0); // Mehr Verdunstung bei Wärme
    float evaporation = evap_rate * dt;

    // Advektions-Transport durch Wind
    float moisture_xm = texture(moistureTex, texCoordXm).r;
    float moisture_xp = texture(moistureTex, texCoordXp).r;
    float moisture_ym = texture(moistureTex, texCoordYm).r;
    float moisture_yp = texture(moistureTex, texCoordYp).r;

    // Upwind-Schema für stabile Advektion
    vec2 moisture_gradient;
    if (wind.x > 0.0) {
        moisture_gradient.x = (moisture_here - moisture_xm) / texelSize.x;
    } else {
        moisture_gradient.x = (moisture_xp - moisture_here) / texelSize.x;
    }

    if (wind.y > 0.0) {
        moisture_gradient.y = (moisture_here - moisture_ym) / texelSize.y;
    } else {
        moisture_gradient.y = (moisture_yp - moisture_here) / texelSize.y;
    }

    float advection = -dot(wind, moisture_gradient) * dt;

    // Diffusion für Glättung
    float laplacian = (moisture_xp + moisture_xm + moisture_yp + moisture_ym - 4.0 * moisture_here) / (texelSize.x * texelSize.x);
    float diffusion = laplacian * 0.0001 * dt; // Kleine Diffusionskonstante

    // Neue Luftfeuchtigkeit
    humidity = moisture_here + evaporation + advection + diffusion;
    humidity = max(0.0, humidity); // Keine negative Feuchtigkeit
}