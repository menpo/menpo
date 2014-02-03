#version 330

uniform sampler2D texture_image;
smooth in vec2 tcoord;
smooth in float interpColor;

layout(location = 0) out vec3 outputColor;

void main()
{
   outputColor = interpColor*interpColor*texture(texture_image, tcoord).rgb;
}

