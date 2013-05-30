#include <stdio.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "glrasterizer.h"
#include "glr.h"
#include "glrglut.h"
#include "shaders.h"

glr_scene scene;


void init_scene(double* points, size_t n_points, unsigned int* trilist,
		size_t n_tris, float* tcoords, uint8_t* texture, size_t texture_width,
		size_t texture_height)
{
	printf("init_scene(...)\n");
	scene.mesh = glr_build_textured_mesh(points, n_points, trilist, n_tris,
											 tcoords, texture, texture_width,
											 texture_height);
	glr_math_float_matrix_eye(scene.camera.perspective);
	glr_math_float_matrix_eye(scene.camera.rotation);
	memset(scene.camera.translation, 0, sizeof(float) * 4);
	scene.light.position[2] = 1.0;
}


void return_FB_pixels(uint8_t *pixels, int width, int height)
{
	printf("return_FB_pixels(...)\n");
	scene.fb_texture = glr_build_rgba_texture(pixels, width, height);
	memset(scene.camera.translation, 0, sizeof(float) * 4);
    glr_math_float_matrix_eye(scene.camera.perspective);
    glr_math_float_matrix_eye(scene.camera.rotation);
    // set the glut config
	scene.config = glr_build_glut_config(width, height);
	// start glut
	glr_glut_init(scene.config);
	// call the init
	init();
	// start the glut loop
	glutMainLoop();
}


void init(void)
{
	printf("init()\n");
	glr_global_state_settings();
	_init_program_and_shaders();
	glUseProgram(scene.program);
	glr_check_error();
	// now we have an instantiated glr_textured_mesh, we have to choose
	// some the OpenGL properties and set them. We decide that the h_points
	// should be bound to input 0 into the shader, while tcoords should be
	// input 1...
	scene.mesh.h_points.attribute_pointer = 0;
	scene.mesh.tcoords.attribute_pointer = 1;
	// assign the meshes texture to be on unit 1 in initialize the buffer for
	// it
	scene.mesh.texture.unit = 1;
	glr_init_buffers_from_textured_mesh(&scene.mesh);
	glr_check_error();
	glr_init_texture(&scene.mesh.texture);
	glr_check_error();
	glr_bind_texture_to_program(&scene.mesh.texture, scene.program);
	glr_check_error();
	_init_frame_buffer();
	glr_check_error();
}


void _init_program_and_shaders(void)
{
	printf("init_program_and_shaders()\n");
	GLuint shaders [2];
	shaders[0] = glr_create_shader_from_string(
			GL_VERTEX_SHADER, texture_shader_vert_str);
	shaders[1] = glr_create_shader_from_string(
			GL_FRAGMENT_SHADER, texture_shader_frag_str);
	scene.program = glr_create_program(shaders, 2);
	glDeleteShader(shaders[0]);
	glDeleteShader(shaders[1]);
}
//
//	glGenFramebuffers(1, &_fbo);
//	glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
//	// first, build a texture:
//	_texture_fb = glr_build_rgba_texture(_fb_texture_pixels, WINDOW_WIDTH,
//			WINDOW_HEIGHT);
//	// assign it to a new unit
//	_texture_fb.unit = 2;
//	// and initialise it
//	glr_init_texture(_texture_fb);
//	// now we can bind to the active framebuffer.
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
//		GL_TEXTURE_2D, _texture_fb.texture_ID, 0);
//
////	// repeat for the position rendering
////	_texture_fb_color = glr_build_rgb_float_texture(_fbo_color_pixels,
////			WINDOW_WIDTH, WINDOW_HEIGHT);
////    _texture_fb_color.unit = 3;
////    glr_init_texture(_texture_fb_color);
////	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
////		GL_TEXTURE_2D, _texture_fb_color.texture_ID, 0);
//
////	// make a new texture (as normal)
////	glBindTexture(GL_TEXTURE_2D, 0);
////	_fb_texture_unit = 2;
////	glActiveTexture(GL_TEXTURE0 + _fb_texture_unit);
////	glGenTextures(1, &_fb_texture);
////	glBindTexture(GL_TEXTURE_2D, _fb_texture);
////	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
////	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
////	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
////	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
////	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0,
////		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
////	glBindTexture(GL_TEXTURE_2D, 0);
////
////	// attach the texture to the framebuffer
////	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
////		GL_TEXTURE_2D, _fb_texture, 0);
////    // THIS BEING GL_COLOR_ATTACHMENT0 means that anything rendered to
////    // layout(location = 0) in the fragment shader will end up here.
////	glr_check_error();
////	glBindTexture(GL_TEXTURE_2D, 0);
//
//
//
//	glGenTextures(1, &_fb_color_id);
//



//	// repeat for the position rendering
	//	_texture_fb_color = glr_build_rgb_float_texture(_fbo_color_pixels,
	//			WINDOW_WIDTH, WINDOW_HEIGHT);
	//    _texture_fb_color.unit = 3;
	//    glr_init_texture(_texture_fb_color);
	//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
	//		GL_TEXTURE_2D, _texture_fb_color.texture_ID, 0);
	//	// make a new texture (as normal)
	//	glBindTexture(GL_TEXTURE_2D, 0);
	//	scene.fb_texture_unit = 2;
	//	glActiveTexture(GL_TEXTURE0 + scene.fb_texture_unit);
	//	glGenTextures(1, &_fb_texture);
	//	glBindTexture(GL_TEXTURE_2D, _fb_texture);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0,
	//		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//	glBindTexture(GL_TEXTURE_2D, 0);
	//



void _init_frame_buffer(void)
{
	printf("_init_frame_buffer()\n");
	// for a framebuffer we don't actually care about the texture unit.
	// however, glr_init_texture will bind the unit before performing the
	// initialization for consistency. We can safely set a (usually illegal)
	// value of zero here so that the unit binding is basically a no op.
	scene.fb_texture.unit = 0;
	glr_init_texture(&scene.fb_texture);
	glr_init_framebuffer(&scene.fbo, &scene.fb_texture,
			GL_COLOR_ATTACHMENT0);
	// We set the framebuffer to GL_COLOR_ATTACHMENT0 - anything rendered to
	// layout(location = 0) in the fragment shader will end up here.
	const GLenum buffers[] = {GL_COLOR_ATTACHMENT0};
	glr_register_draw_framebuffers(scene.fbo, 1, buffers);
	// now, the depth buffer
	GLuint depth_buffer;
	glGenRenderbuffers(1, &depth_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT,
			scene.fb_texture.width, scene.fb_texture.height);
	glBindFramebuffer(GL_FRAMEBUFFER, scene.fbo);
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
}

void display(void)
{
	printf("display()\n");
	glBindFramebuffer(GL_FRAMEBUFFER, scene.fbo);
	glr_render_scene(&scene);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glutLeaveMainLoop();
}


void grab_framebuffer_and_cleanup(void)
{
	printf("grab_framebuffer_and_cleanup()\n");
	glr_get_framebuffer(scene.fb_texture.unit, scene.fb_texture.texture_ID,
			scene.fb_texture.format, scene.fb_texture.type,
			scene.fb_texture.data);
	glr_destroy_program();
	glr_destroy_vbos_on_trianglar_mesh(scene.mesh);
}
