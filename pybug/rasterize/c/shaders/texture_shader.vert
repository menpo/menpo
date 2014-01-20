#version 330
#extension GL_ARB_explicit_attrib_location : require

#define PI 3.1415926535897932384626433832795

mat4 CreatePerspectiveMatrix(in float fov, in float aspect,
    in float near, in float far)
{
    mat4 m = mat4(0.0);

    float angle = (fov / 180.0f) * PI;
    float f = 1.0f / tan( angle * 0.5f );

    /* Note, matrices are accessed like 2D arrays in C.
       They are column major, i.e m[y][x] */

    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][2] = (far + near) / (near - far);
    m[2][3] = -1.0f;
    m[3][2] = (2.0f * far*near) / (near - far);

    return m;
}

layout(location = 0) in vec4 point;
layout(location = 1) in vec2 tcoord_in;
layout(location = 2) in vec3 linear_mapping_coord_in;

smooth out vec2 tcoord;
smooth out vec3 linear_mapping_coord;

void main() {
    mat4 clipMatrix = CreatePerspectiveMatrix(90.0, 4.0/3.0, 0.001, 500.0);
    mat4 worldMatrix = CreateSomeWorldMatrix();
    vec4 translate =

    gl_Position = clipMatrix * worldMatrix * point;
    tcoord = tcoord_in;
    linear_mapping_coord = linear_mapping_coord_in;
}
