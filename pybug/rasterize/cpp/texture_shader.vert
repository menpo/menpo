#version 330
#extension GL_ARB_explicit_attrib_location : require

layout(location = 0) in vec4 point;
layout(location = 1) in vec2 tcoord_in;
layout(location = 2) in vec3 linear_mapping_coord_in;

smooth out vec2 tcoord;
smooth out vec3 linear_mapping_coord;

void main() {
    gl_Position = point;
    tcoord = tcoord_in;
    linear_mapping_coord = linear_mapping_coord_in;
}

