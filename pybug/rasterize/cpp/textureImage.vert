#version 330

layout(location = 0) in vec4 point_in;
layout(location = 1) in vec2 tcoord_in;
layout(location = 2) in vec3 color_in;

smooth out vec2 tcoord;
smooth out vec3 color;

void main()
{
    gl_Position = point_in;
    tcoord = tcoord_in;
    color = color_in;
}
