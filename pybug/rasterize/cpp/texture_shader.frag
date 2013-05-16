#version 330

uniform sampler2D texture_image;
smooth in vec2 tcoord;
smooth in vec3 linear_mapping_coord;

layout(location = 0) out vec3 output_color;
layout(location = 1) out vec3 output_linear_mapping;

void main() {
   output_color = texture(texture_image, tcoord).rgb;
   output_linear_mapping = linear_mapping_coord;
}

