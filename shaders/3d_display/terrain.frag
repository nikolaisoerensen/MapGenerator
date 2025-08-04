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

vec3 getTerrainColor() {
    // Height-based terrain coloring
    float height = FragPos.y / maxHeight;

    if (height < 0.3) {
        // Water to Beach (blue to sandy brown)
        return mix(vec3(0.2, 0.4, 0.8), vec3(0.8, 0.7, 0.4), height / 0.3);
    } else if (height < 0.7) {
        // Beach to Grass (sandy brown to green)
        return mix(vec3(0.8, 0.7, 0.4), vec3(0.2, 0.6, 0.2), (height - 0.3) / 0.4);
    } else {
        // Grass to Snow (green to white)
        return mix(vec3(0.2, 0.6, 0.2), vec3(0.9, 0.9, 0.9), (height - 0.7) / 0.3);
    }
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
        case 0: color = getTerrainColor(); break;
        case 1: color = getGeologyColor(); break;
        case 2: color = getWeatherColor(); break;
        case 3: color = getWaterColor(); break;
        case 4: color = getBiomeColor(); break;
        case 5: color = getSettlementColor(); break;
        default: color = getTerrainColor(); break;
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

    // Specular lighting
    vec3 viewDir = normalize(-FragPos);  // Camera at origin
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = 0.3 * spec * vec3(1.0, 1.0, 1.0);

    // Combine lighting with shadow
    vec3 result = (ambient + diffuse + specular) * shadow;

    FragColor = vec4(result, 1.0);
}