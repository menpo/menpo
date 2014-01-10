#pragma once
#include <stdint.h>
#include "glr.h"

glr_scene init_scene(double* points, size_t n_points, unsigned int* trilist,
		size_t n_tris, float* tcoords, uint8_t* texture, size_t texture_width,
		size_t texture_height);

void return_FB_pixels(glr_scene* scene, uint8_t *pixels, int width, int height);

void init(glr_scene* scene);

void _init_program_and_shaders(glr_scene* scene);

void _init_frame_buffer(glr_scene* scene);

void grab_framebuffer_and_cleanup(glr_scene* scene);
