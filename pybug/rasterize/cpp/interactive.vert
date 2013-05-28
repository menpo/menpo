#version 330

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 textureCoord;
layout(location = 2) in vec3 vertexNormal;

smooth out vec2 tcoord;
smooth out float interpColor;

uniform mat4 perspectiveMatrix;
uniform mat4 rotationMatrix;
uniform vec4 translationVector;
uniform vec3 lightDirection;

void main()
{
    vec4 cameraPos = rotationMatrix*position + translationVector;
    gl_Position = perspectiveMatrix*cameraPos;
    tcoord = textureCoord;
    vec4 cameraNormal = rotationMatrix*vec4(vertexNormal,1.0);
    vec3 camNormalised = normalize(cameraNormal.xyz);
    float cosAngIncidence = dot(camNormalised,lightDirection);
    interpColor = camNormalised.z;
}

