#version 330 core

// Vertex Attributes
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texCoord;

// Transformation Matrices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Lighting
uniform vec3 lightPos;

// Output to Fragment Shader
out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 LightPos;

void main() {
    // Transform vertex position to world space
    FragPos = vec3(model * vec4(position, 1.0));

    // Transform normal to world space (accounting for non-uniform scaling)
    Normal = mat3(transpose(inverse(model))) * normal;

    // Pass through texture coordinates
    TexCoord = texCoord;

    // Pass light position to fragment shader
    LightPos = lightPos;

    // Final vertex position in clip space
    gl_Position = projection * view * vec4(FragPos, 1.0);
}