#pragma once
#include <stdint.h>

void display(void);

void init_scene(double* points, size_t n_points, unsigned int* trilist,
		size_t n_tris, float* tcoords, uint8_t* texture, size_t texture_width,
		size_t texture_height);

void return_FB_pixels(uint8_t *pixels, int width, int height);

void init(void);

void _init_program_and_shaders(void);

void _init_frame_buffer(void);

void display(void);

void grab_framebuffer_and_cleanup(void);
