#version 130
#extension GL_ARB_explicit_attrib_location : require

uniform sampler2D textureImage;
smooth in vec2 texCoord;
smooth in float interpColor;

layout(location = 0) out vec3 outputColor;

void main()
{
   outputColor = interpColor*interpColor*texture(textureImage, texCoord).rgb;
}

