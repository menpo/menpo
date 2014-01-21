#version 330
#extension GL_ARB_explicit_attrib_location : require

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;
uniform vec4 lightDirection;

layout(location = 0) in vec4 point;
layout(location = 1) in vec2 tcoordIn;
layout(location = 2) in vec3 linearMappingCoordIn;

smooth out vec2 tcoord;
smooth out vec3 linearMappingCoord;

void main() {
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * point;
    tcoord = tcoordIn;
    linearMappingCoord = point.xyz;
}
