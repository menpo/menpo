#pragma once
#include <stdint.h>
#include "glr.h"

void return_FB_pixels(glr_scene* scene, uint8_t *pixels);

void init(glr_scene* scene);

void _init_program_and_shaders(glr_scene* scene);

void _init_frame_buffer(glr_scene* scene);

void grab_framebuffer_and_cleanup(glr_scene* scene);

