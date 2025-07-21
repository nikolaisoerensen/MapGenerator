#version 300 es
precision highp float;

in vec2 vertPosition;
in vec2 vertTexCoord;

uniform vec2 texelSize;

out vec2 texCoord;
out vec2 fragCoord;

// Nachbar-Koordinaten für finite Differenzen
out vec2 texCoordXm, texCoordXp;  // Links/Rechts
out vec2 texCoordYm, texCoordYp;  // Unten/Oben

void main()
{
    fragCoord = vertTexCoord;
    texCoord = vertTexCoord * texelSize;

    // Nachbar-Koordinaten für Gradientenberechnung
    texCoordXm = texCoord + vec2(-texelSize.x, 0.0);
    texCoordXp = texCoord + vec2(texelSize.x, 0.0);
    texCoordYm = texCoord + vec2(0.0, -texelSize.y);
    texCoordYp = texCoord + vec2(0.0, texelSize.y);

    gl_Position = vec4(vertPosition, 0.0, 1.0);
}