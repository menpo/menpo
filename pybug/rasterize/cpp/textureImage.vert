#version 330

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 textureCoord;
layout(location = 2) in vec3 non_tps_coord;

smooth out vec2 texCoord;
smooth out vec3 coord;

void main()
{
    gl_Position = position;
    texCoord = textureCoord;
    coord = non_tps_coord;
}

