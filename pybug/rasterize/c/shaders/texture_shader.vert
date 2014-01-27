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
    // position is what we would normally pass straight through
    vec4 position = projectionMatrix * viewMatrix * modelMatrix * point;
    // flip the y axis to deal with textures being passed in flipped!
    // note that this in effect 'flips' the triangles from being CCW to CW.
    // this will only work when used with the global flag glFrontFace(GL_CW)
    // set.
    position.y = -1.0 * position.y;
    gl_Position = position;
    // same idea, but for the texture space. This deals with the texuture
    // being upside down
    tcoord = vec2(tcoordIn.s, 1.0 - tcoordIn.t);
    linearMappingCoord = linearMappingCoordIn;
}

