#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glrasterizer.h"
#include "glrglfw.h"
#include "shaders.h"

void init_program_to_texture_shader(glr_scene* scene)
{
	printf("init_program_and_shaders()\n");
	GLuint shaders [2];
	shaders[0] = glr_create_shader_from_string(
			GL_VERTEX_SHADER, texture_shader_vert_str);
	shaders[1] = glr_create_shader_from_string(
			GL_FRAGMENT_SHADER, texture_shader_frag_str);
	scene->program = glr_create_program(shaders, 2);
	glDeleteShader(shaders[0]);
	glDeleteShader(shaders[1]);
}

void init_frame_buffer(glr_scene* scene)
{
	printf("init_frame_buffer()\n");
	// for a framebuffer we don't actually care about the texture unit.
	// however, glr_init_texture will bind the unit before performing the
	// initialization for consistency. We can safely set a (usually illegal)
	// value of zero here so that the unit binding is basically a no op.
	scene->fb_rgb_target.unit = 0;
	scene->fb_f3v_target.unit = 0;
	glr_init_texture(&scene->fb_rgb_target);
	glr_init_texture(&scene->fb_f3v_target);
    // ask OpenGL to make us a framebuffer object for the fbo
	glGenFramebuffers(1, &scene->fbo);
    // init the two framebuffer textures
	glr_init_framebuffer(&scene->fbo, &scene->fb_rgb_target, GL_COLOR_ATTACHMENT0);
	glr_init_framebuffer(&scene->fbo, &scene->fb_f3v_target, GL_COLOR_ATTACHMENT1);
	// We set the RGB framebuffer to GL_COLOR_ATTACHMENT0 - anything rendered to
	// layout(location = 0) in the fragment shader will end up here.
	// The float framebuffer is attached to GL_COLOR_ATTACHMENT1 - anything rendered to
	// layout(location = 1) in the fragment shader will end up here.
	GLenum buffers[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glr_register_draw_framebuffers(scene->fbo, 2, buffers);
	// now, the depth buffer
	GLuint depth_buffer;
	glGenRenderbuffers(1, &depth_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
			scene->fb_rgb_target.width, scene->fb_rgb_target.height);
	glBindFramebuffer(GL_FRAMEBUFFER, scene->fbo);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
			GL_RENDERBUFFER, depth_buffer);
	// THIS BEING GL_DEPTH_COMPONENT means that the depth information at each
	// fragment will end up here. Note that we must manually set up the depth
	// buffer when using framebuffers.
	GLenum status;
	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Framebuffer error: 0x%04X\n", status);
		exit(EXIT_FAILURE);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glr_check_error();
}


void render_texture_shader_to_fb(glr_scene* scene)
{
	printf("render_texture_shader_to_fb(...)\n");
	// call the init
	init(scene);

    // render to the framebuffer, and pull off the state
    glr_render_to_framebuffer(scene);

    // clear up our OpenGL state
	glr_destroy_vbos_on_trianglar_mesh(&(scene->mesh));
}


void init(glr_scene* scene)
{
	printf("init()\n");
	glUseProgram(scene->program);
	glr_check_error();
	// now we have an instantiated glr_textured_mesh, we have to choose
	// some the OpenGL properties and set them. We decide that the vertices
	// should be bound to input 0 into the shader, while tcoords should be
	// input 1, and the float 3 vec is 2.
	scene->mesh.vertices.attribute_pointer = 0;
	scene->mesh.tcoords.attribute_pointer = 1;
	scene->mesh.f3v_data.attribute_pointer = 2;
	// assign the meshes texture to be on unit 1 and initialize the buffer for
	// it
	scene->mesh.texture.unit = 1;
	glr_init_vao(&scene->mesh);
	glr_check_error();
	glr_init_texture(&scene->mesh.texture);
	glr_check_error();
}

