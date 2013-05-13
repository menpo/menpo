#version 330

layout(location = 0) in vec4 point;
layout(location = 1) in vec2 tcoord;
layout(location = 2) in vec3 non_tps_coord;

smooth out vec2 tcoord;
smooth out vec3 coord;

void main()
{
    gl_Position = point;
    texCoord = tcoord;
    coord = non_tps_coord;
}