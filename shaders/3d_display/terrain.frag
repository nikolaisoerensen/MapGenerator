#version 330 core

// Output
out vec4 FragColor;

// Input from Vertex Shader
in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 LightPos;

// Terrain Parameters
uniform float heightScale;
uniform float maxHeight;
uniform int renderMode;  // 0=terrain, 1=geology, 2=weather, 3=water, 4=biome, 5=settlement

// Shadow Mapping
uniform sampler2D shadowMap;
uniform bool useShadows;

// Overlay Textures (for different tabs)
uniform sampler2D overlayTexture;
uniform bool useOverlay;
uniform float overlayStrength;

// Contour Lines (globaler Shell-Toggle, siehe MapDisplay3D.set_contour_overlay()).
// FragPos.y ist bereits Weltraum-Hoehe = rohe Heightmap-Meter * heightScale (siehe
// _generate_terrain_mesh() pos_y = heightmap * terrain_height_scale) - daher genuegt
// FragPos.y/heightScale, um dieselbe Roh-Hoehe wie _calculate_contour_levels() in
// map_display_2d.py zu bekommen, ohne eine eigene Heightmap-Textur hochzuladen. Der
// Mesh basiert immer auf der kombinierten Heightmap (heightmap_combined), unabhaengig
// vom aktuell gezeigten renderMode/Overlay - Konturen bleiben also korrekt, egal
// welcher Layer gerade angezeigt wird.
uniform bool useContours;
uniform float contourInterval;

vec3 getTerrainColor() {
    // Height-based terrain coloring. Kein Blau mehr (sah wie Wasser aus) -
    // Grün ist jetzt die niedrigste Farbe (height=0), Weiß die höchste
    // (height=1). Dieselben 5 Stützstellen wie die 2D-Colormap
    // (gui/widgets/map_display_2d.py, plt.cm.terrain ab 25% zugeschnitten),
    // damit 2D- und 3D-Ansicht farblich konsistent aussehen.
    float height = FragPos.y / maxHeight;

    if (height < 0.25) {
        // Green to pale yellow-green
        return mix(vec3(0.0, 0.8, 0.4), vec3(0.76, 0.95, 0.55), height / 0.25);
    } else if (height < 0.5) {
        // Pale yellow-green to tan
        return mix(vec3(0.76, 0.95, 0.55), vec3(0.75, 0.67, 0.46), (height - 0.25) / 0.25);
    } else if (height < 0.75) {
        // Tan to grayish brown
        return mix(vec3(0.75, 0.67, 0.46), vec3(0.63, 0.53, 0.51), (height - 0.5) / 0.25);
    } else {
        // Grayish brown to white
        return mix(vec3(0.63, 0.53, 0.51), vec3(1.0, 1.0, 1.0), (height - 0.75) / 0.25);
    }
}

vec3 getTerrainColorBlended() {
    // Wie die anderen getXColor()-Funktionen (Geology/Weather/Water/Biome/
    // Settlement), aber für renderMode 0 (Terrain-Tab) - z.B. für den
    // Slope-Overlay. getTerrainColor() selbst bleibt unverändert (wird auch
    // von den anderen getXColor()-Funktionen als Basis-Farbe wiederverwendet).
    vec3 baseColor = getTerrainColor();
    if (useOverlay) {
        vec3 overlayColor = texture(overlayTexture, TexCoord).rgb;
        return mix(baseColor, overlayColor, overlayStrength);
    }
    return baseColor;
}

vec3 getGeologyColor() {
    // Geology-specific coloring would use overlay texture
    if (useOverlay) {
        vec3 overlayColor = texture(overlayTexture, TexCoord).rgb;
        vec3 baseColor = getTerrainColor();
        return mix(baseColor, overlayColor, overlayStrength);
    }
    return vec3(0.6, 0.6, 0.6);  // Default gray
}

vec3 getWeatherColor() {
    // Weather-specific coloring
    if (useOverlay) {
        vec3 overlayColor = texture(overlayTexture, TexCoord).rgb;
        vec3 baseColor = getTerrainColor();
        return mix(baseColor, overlayColor, overlayStrength);
    }
    return getTerrainColor();
}

vec3 getWaterColor() {
    // Water-specific coloring with blue tints
    vec3 baseColor = getTerrainColor();
    if (useOverlay) {
        vec3 overlayColor = texture(overlayTexture, TexCoord).rgb;
        return mix(baseColor, overlayColor, overlayStrength);
    }
    return baseColor;
}

vec3 getBiomeColor() {
    // Biome-specific coloring
    if (useOverlay) {
        vec3 overlayColor = texture(overlayTexture, TexCoord).rgb;
        return overlayColor;  // Biomes use full overlay color
    }
    return getTerrainColor();
}

vec3 getSettlementColor() {
    // Settlement base with optional civ map overlay
    vec3 baseColor = getTerrainColor();
    if (useOverlay) {
        vec3 overlayColor = texture(overlayTexture, TexCoord).rgb;
        return mix(baseColor, overlayColor, overlayStrength * 0.5);  // Subtle overlay
    }
    return baseColor;
}

void main() {
    // Select color based on render mode
    vec3 color;
    switch(renderMode) {
        case 0: color = getTerrainColorBlended(); break;
        case 1: color = getGeologyColor(); break;
        case 2: color = getWeatherColor(); break;
        case 3: color = getWaterColor(); break;
        case 4: color = getBiomeColor(); break;
        case 5: color = getSettlementColor(); break;
        default: color = getTerrainColor(); break;
    }

    // Contour Lines - dunkler, anti-aliaster Streifen bei jedem Vielfachen von
    // contourInterval (gleiche Grauton-Familie wie CanvasSettings.CANVAS_2D
    // "contour_colors"[0] = #7f8c8d, fuer 2D/3D-Konsistenz).
    if (useContours && contourInterval > 0.0) {
        float rawHeight = FragPos.y / heightScale;
        float m = mod(rawHeight, contourInterval) / contourInterval;
        float d = max(fwidth(rawHeight) / contourInterval, 0.0001);
        float line = 1.0 - smoothstep(0.0, d * 1.5, min(m, 1.0 - m));
        color = mix(color, vec3(0.498, 0.549, 0.553), line * 0.85);
    }

    // Shadow mapping
    float shadow = 1.0;
    if (useShadows) {
        shadow = texture(shadowMap, TexCoord).r;
        shadow = 0.3 + 0.7 * shadow;  // Prevent completely black shadows
    }

    // Phong Lighting Model
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(LightPos - FragPos);

    // Ambient lighting
    vec3 ambient = 0.3 * color;

    // Diffuse lighting
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color;

    // Specular lighting - Terrain (Fels/Erde/Gras) ist eine raue, matte Oberfläche
    // ohne nennenswerte Spiegelung. Vorher: Shininess 32 bei 30% Stärke - das ist ein
    // Material-Modell für glatte/glänzende Oberflächen (Metall, Kunststoff, nasses
    // Material) und erzeugte den gemeldeten "glossy"-Eindruck. Jetzt: breiter, viel
    // schwächerer Streulicht-Anteil statt eines scharfen Glanzpunkts.
    vec3 viewDir = normalize(-FragPos);  // Camera at origin
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 4);
    vec3 specular = 0.03 * spec * vec3(1.0, 1.0, 1.0);

    // Combine lighting with shadow
    vec3 result = (ambient + diffuse + specular) * shadow;

    FragColor = vec4(result, 1.0);
}