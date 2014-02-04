#pragma once
#include <stdint.h>
#include "glr.h"

void init_program_to_texture_shader(glr_scene* scene);

// NOTE - fb_rgb_target and fb_f3v_target have to be set 
// prior to invoking this method!
void init_frame_buffer(glr_scene* scene);

void render_texture_shader_to_fb(glr_scene* scene);

void init(glr_scene* scene);

void _init_frame_buffer(glr_scene* scene);

void grab_framebuffer_and_cleanup(glr_scene* scene);

