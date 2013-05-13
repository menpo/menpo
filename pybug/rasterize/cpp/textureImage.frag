#version 330

uniform sampler2D textureImage;
smooth in vec2 tcoord;
smooth in vec3 coord;

layout(location = 0) out vec3 outputColor;
layout(location = 1) out vec3 outputCoord;

void main()
{
   outputColor = texture(textureImage, tcoord).rgb;
   outputCoord = coord;
}
