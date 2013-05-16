/*
 * gllr.cpp
 *
 *  Created on: 16 May 2013
 *      Author: jab08
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <GL/glew.h>
#include <vector>
#include <GL/freeglut.h>
#include "glr.h"

void glr_check_error() {
	GLenum err;
	err = glGetError();
	if (err != GL_NO_ERROR) {
		printf("Error. glError: 0x%04X", err);
		std::cout << " - " << gluErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
}

GLuint glr_create_shader_from_string(GLenum shader_type,
								     const GLchar* string) {
	GLuint shader = glCreateShader(shader_type);
	glShaderSource(shader, 1, &string, NULL);
	glCompileShader(shader);
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		GLint info_log_length;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
		GLchar *str_info_log = new GLchar[info_log_length + 1];
		glGetShaderInfoLog(shader, info_log_length, NULL, str_info_log);
		const char *strShaderType = NULL;
		switch (shader_type) {
			case GL_VERTEX_SHADER:   strShaderType = "vertex";   break;
			case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
			case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
		}
		fprintf(stderr, "Compile failure in %s shader: \n%s\n",
				strShaderType, str_info_log);
		delete[] str_info_log;
		exit(EXIT_FAILURE);
	}
	return shader;
}

GLuint glr_create_program(const std::vector<GLuint> &shaders) {
	GLuint program = glCreateProgram();
	for(size_t i = 0; i < shaders.size(); i++)
		glAttachShader(program, shaders[i]);
	glLinkProgram(program);
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		GLint info_log_length;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
		GLchar *str_info_log = new GLchar[info_log_length + 1];
		glGetProgramInfoLog(program, info_log_length, NULL, str_info_log);
		fprintf(stderr, "Linker failure: %s\n", str_info_log);
		delete[] str_info_log;
	}
	for(size_t i = 0; i < shaders.size(); i++)
		glDetachShader(program, shaders[i]);
	return program;
}

glr_texture glr_build_rgba_texture(uint8_t* texture, size_t width,
								   size_t height) {
	glr_texture texture_tmp;
	texture_tmp.unit = 999; // the texture unit this texture binds to. Set to
	// 999 as a safety - must be changed!
	texture_tmp.internal_format = GL_RGBA8;
	texture_tmp.width = width;
	texture_tmp.height = height;
	texture_tmp.format = GL_RGBA;
	texture_tmp.type = GL_UNSIGNED_BYTE;
	texture_tmp.data = texture;
	return texture_tmp;
}

glr_texture glr_build_rgb_float_texture(float* texture, size_t width,
								   	    size_t height) {
	glr_texture texture_tmp;
	texture_tmp.unit = 999; // the texture unit this texture binds to. Set to
	// 999 as a safety - must be changed!
	texture_tmp.internal_format = GL_RGB32F_ARB;
	texture_tmp.width = width;
	texture_tmp.height = height;
	texture_tmp.format = GL_RGB;
	texture_tmp.type = GL_FLOAT;
	texture_tmp.data = texture;
	return texture_tmp;
}

glr_vectorset glr_build_h_points(double* points, size_t n_points) {
	glr_vectorset points_tmp;
	points_tmp.datatype = GL_DOUBLE;
	points_tmp.n_dims = 4;
	points_tmp.n_vectors = n_points;
	points_tmp.size = sizeof(GLdouble);
	points_tmp.vectors = points;
	return points_tmp;
}

glr_vectorset glr_build_tcoords(float* tcoords, size_t n_points) {
	glr_vectorset tcoords_tmp;
	tcoords_tmp.datatype = GL_FLOAT;
	tcoords_tmp.n_dims = 2;
	tcoords_tmp.n_vectors = n_points;
	tcoords_tmp.size = sizeof(float);
	tcoords_tmp.vectors = tcoords;
	return tcoords_tmp;
}

glr_vectorset glr_build_trilist(unsigned int* trilist, size_t n_tris) {
	glr_vectorset trilist_tmp;
	trilist_tmp.datatype = GL_UNSIGNED_INT;
	trilist_tmp.n_dims = 3;
	trilist_tmp.n_vectors = n_tris;
	trilist_tmp.size = sizeof(GLuint);
	trilist_tmp.vectors = trilist;
	return trilist_tmp;
}

glr_textured_mesh glr_build_textured_mesh(double* points, size_t n_points,
		unsigned int* trilist, size_t n_tris, float* tcoords,
		uint8_t* texture, size_t texture_width, size_t texture_height) {
	glr_textured_mesh textured_mesh;
	textured_mesh.h_points = glr_build_h_points(points, n_points);
	textured_mesh.tcoords = glr_build_tcoords(tcoords, n_points);
	textured_mesh.trilist = glr_build_trilist(trilist, n_tris);
	textured_mesh.texture = glr_build_rgba_texture(texture,
			texture_width, texture_height);
	return textured_mesh;
}

void glr_init_array_buffer_from_vectorset(glr_vectorset& vector) {
	glGenBuffers(1, &(vector.vbo));
	glBindBuffer(GL_ARRAY_BUFFER, vector.vbo);
	glBufferData(GL_ARRAY_BUFFER,
				 vector.size * vector.n_vectors * vector.n_dims,
				 vector.vectors, GL_STATIC_DRAW);
	glEnableVertexAttribArray(vector.attribute_pointer);
	glVertexAttribPointer(vector.attribute_pointer, vector.n_dims,
						  vector.datatype, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void glr_init_element_buffer_from_vectorset(glr_vectorset& vector) {
	glGenBuffers(1, &(vector.vbo));
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vector.vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
				 vector.size * vector.n_vectors * vector.n_dims,
				 vector.vectors, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void glr_setup_buffers_on_textured_mesh(glr_textured_mesh& mesh) {
	glGenVertexArrays(1, &(mesh.vao));
	glBindVertexArray(mesh.vao);
	glr_init_array_buffer_from_vectorset(mesh.h_points);
	glr_init_array_buffer_from_vectorset(mesh.tcoords);
	glr_init_element_buffer_from_vectorset(mesh.trilist);
	glBindVertexArray(0);
}

// TODO make the texture sampler a seperate customizable thing.
void glr_init_texture(glr_texture& texture) {
	// activate this textures unit
	glActiveTexture(GL_TEXTURE0 + texture.unit);
	glGenTextures(1, &(texture.texture_ID));
	glBindTexture(GL_TEXTURE_2D, texture.texture_ID);
	glTexImage2D(GL_TEXTURE_2D, 0, texture.internal_format,
		texture.width, texture.height, 0, texture.format,
		texture.type, texture.data);
	// Create the description of the texture (sampler) and bind it to the
	// correct texture unit
	glGenSamplers(1, &(texture.sampler));
	glSamplerParameteri(texture.sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glSamplerParameteri(texture.sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glSamplerParameteri(texture.sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glBindSampler(texture.unit, texture.sampler);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void glr_bind_texture_to_program(glr_texture& texture, GLuint program) {
	glActiveTexture(GL_TEXTURE0 + texture.unit);
	glBindTexture(GL_TEXTURE_2D, texture.texture_ID);
	// bind the texture to a uniform called "texture" which can be
	// accessed from shaders
	texture.uniform = glGetUniformLocation(program, "texture_image");
	glUniform1i(texture.uniform, texture.unit);
	// set the active Texture to 0 - as long as this is not changed back
	// to textureImageUnit, we know our shaders will find textureImage bound to
	// GL_TEXTURE_2D when they look in textureImageUnit
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void glr_global_state_settings() {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);
	glDepthFunc(GL_LEQUAL);
}

void glr_get_framebuffer(unsigned int texture_unit_offset,
		             GLuint texture_framebuffer, GLenum texture_specification,
		             GLenum texture_datatype, void* texture) {
	glActiveTexture(GL_TEXTURE0 + texture_unit_offset);
	glBindTexture(GL_TEXTURE_2D, texture_framebuffer);
	glGetTexImage(GL_TEXTURE_2D, 0, texture_specification,
			      texture_datatype, texture);
	glActiveTexture(GL_TEXTURE0);
}

void glr_destroy_program() {
	glUseProgram(0);
}

void glr_destroy_vbos_on_trianglar_mesh(glr_textured_mesh mesh) {
	glDisableVertexAttribArray(mesh.h_points.attribute_pointer);
	glDisableVertexAttribArray(mesh.tcoords.attribute_pointer);
	// TODO this needs to be the color array
	glDisableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &(mesh.h_points.vbo)); // _points_buffer
	//glDeleteBuffers(1, &_color_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &(mesh.trilist.vbo));
	// now are buffers are all cleared, we can unbind and delete the vao
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &(mesh.vao));
}
