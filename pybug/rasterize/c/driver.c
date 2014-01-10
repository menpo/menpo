#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "glrasterizer.h"
#include "glr.h"

void init_points(double* points) 
{
    //v1
    points[0]  =  0.f;
    points[1]  =  0.f;
    points[2]  =  0.f;
    points[3]  =  1.f;
    //v2
    points[4]  =  0.f;
    points[5]  =  0.75f;
    points[6]  =  0.f;
    points[7]  =  1.f;
    //v3
    points[8]  =  0.75f;
    points[9]  =  0.f;
    points[10] =  0.f;
    points[11] =  1.f;
}

void set_pixel_values(uint8_t *p, uint8_t R, uint8_t G, uint8_t B, uint8_t A)
{
    *p++ = R;
    *p++ = G;
    *p++ = B;
    *p++ = A;
}

void init_texture(uint8_t *texture, unsigned n_pixels) 
{
    unsigned i = 0;
    for(; i < n_pixels; i += 4)
        set_pixel_values(&texture[i], 255, 0, 0, 255);
    for(; i < n_pixels * 2; i += 4)
        set_pixel_values(&texture[i], 0, 255, 0, 255);
    for(; i < n_pixels * 3; i += 4)
        set_pixel_values(&texture[i], 0, 0, 255, 255);
    for(; i < n_pixels * 4; i += 4)
        set_pixel_values(&texture[i], 255, 255, 0, 255);
}

void init_tcoords(float* tc)
{
    tc[0]  = 0.;
    tc[1]  = 0.;

    tc[2]  = 0.;
    tc[3]  = 1.;

    tc[4]  = 1.;
    tc[5]  = 0.;
}

void init_color(float* c)
{
    c[0]  = 0.;
    c[1]  = 0.;
    c[2]  = 1.;

    c[3]  = 0.;
    c[4]  = 0.;
    c[5]  = 1.;

    c[6]  = 0.;
    c[7]  = 0.;
    c[8]  = 1.;
}

int main(int argc, char** argv)
{
    unsigned int trilist[] =
    {
        0, 2, 1,
    };
    size_t n_points = 3;
    size_t n_tris = 1;
    size_t t_w = 64;
    size_t t_h = 64;
    double points [n_points * 4];
    uint8_t texture [t_w * t_h * 4];
    float tcoords [n_points * 2];

    init_points(points);
    init_tcoords(tcoords);
    init_texture(texture, t_w * t_h);

    int output_w = 128;
    int output_h = 128;

    clock_t start = clock(), diff;
    glr_glfw_context context = init_offscreen_context(128, 128);
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken for context: %d ms\n", msec);
    start = clock();
    glr_scene scene = init_scene(points, n_points, trilist, n_tris, tcoords, 
            texture, t_w, t_h);
    // attach the context to the scene
    scene.context = &context;

    uint8_t pixels [output_w * output_h * 4];
    return_FB_pixels(&scene, pixels);
    diff = clock() - start;
    msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken: %d ms\n", msec);
    return(0);
}

