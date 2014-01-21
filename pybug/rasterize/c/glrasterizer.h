#pragma once
#include <stdint.h>
#include "glr.h"

void init_program_to_texture_shader(glr_scene* scene);

void init_frame_buffer(glr_scene* scene, uint8_t* pixels);

void render_texture_shader_to_fb(glr_scene* scene);

void init(glr_scene* scene);

void _init_frame_buffer(glr_scene* scene);

void grab_framebuffer_and_cleanup(glr_scene* scene);

