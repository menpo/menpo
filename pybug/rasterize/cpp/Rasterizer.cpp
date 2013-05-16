#include "Rasterizer.h"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <cmath>

/**
 * Checks the global OpenGL state to see if an error has occurred.
 *
 * If an error is encountered, the program is exited, and a log to stderr
 * of the problem made.
 */
void glr_check_error();

/**
 * Builds an OpenGL shader of the type shader_type, where shader_type can be:
 * 	- GL_VERTEX_SHADER
 * 	- GL_GEOMETRY_SHADER
 * 	- GL_FRAGMENT_SHADER
 *
 * 	Returns the GLuint that points to the shader. If there is an error, the
 * 	program is exited, and the error message produced.
 */
GLuint glr_create_shader_from_string(GLenum shader_type, const GLchar* string);

/*
 * Taking a vector of shaders, links the shaders into a single program,
 * returning the programs binding. If there is an error in the link,
 * an error message is produced and the program is terminated. If this program
 * successfully returns, it is safe to delete the individual shaders themselves
 * immediately - the program is built as is kept in the OpenGL state for us.
 */
GLuint glr_create_program(const std::vector<GLuint> &shaders);

/*
 * Builds a glr_vectorset storing the pertinent information about a set of
 * homogeneous (X, Y, Z, W) points. Returns the vectorset ready for use.
 */
glr_vectorset glr_build_h_points(double* points, size_t n_points);

/*
 * Builds a glr_vectorset storing the pertinent information about a set of
 * (S,T) texture coordinates. Returns the vectorset ready for use.
 */
glr_vectorset glr_build_tcoords(float* tcoords, size_t n_points);

/*
 * Builds a glr_vectorset storing the pertinent information about a set of
 * (v1, v2, v3) triangluation indices (a triangle list). Returns the vectorset
 * ready for use.
 */
glr_vectorset glr_build_trilist(unsigned int* trilist, size_t n_tris);





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

void read_file(const char* filepath, GLchar* file_string) {
	std::ifstream file(filepath, std::ifstream::in);
	unsigned int i  = 0;
	while(file.good()) {
		file_string[i] = file.get();
		i++;
	}
	file_string[i-1] = '\0';
}

void matrix_x_vector(float* matrix, float* vector, float*result) {
	result[0] = 0;
	result[1] = 0;
	result[2] = 0;
	result[3] = 0;
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			result[i] += matrix[4*i+ j]*vector[j];
		}
	}
}

GLuint create_shader_from_filepath(GLenum shader_type,
						           const char* filepath){
	GLchar file_string[100000];
	read_file(filepath, file_string);
	// now we've done reading, const the data
	const GLchar* const_file_string = (const char*)file_string;
	return glr_create_shader_from_string(shader_type, const_file_string);
}


Rasterizer::Rasterizer(double* points, float* color, size_t n_points,
					   unsigned int* trilist, size_t n_tris, float* tcoords,
					   uint8_t* texture, size_t texture_width,
					   size_t texture_height, bool INTERACTIVE_MODE) {
	_light_vector = new float[3];
	memset(_light_vector,0,3);
	_light_vector[2] = 1.0;

	title = "MM3D Viewer";
	std::cout << "Rasterizer::Rasterizer(...)" << std::endl;
	_textured_mesh = glr_build_textured_mesh(points, n_points, trilist, n_tris,
											 tcoords, texture, texture_width,
											 texture_height);
	// now we have an instantiated glr_textured_mesh, we have to choose
	// some the OpenGL properties and set them. We decide that the h_points
	// should be bound to input 0 into the shader, while tcoords should be
	// input 1...
	_textured_mesh.h_points.attribute_pointer = 0;
	_textured_mesh.tcoords.attribute_pointer = 1;
	// and we assign the texture we have to unit 1.
	_textured_mesh.texture.unit = 1;
	_color = color;
	_n_points = n_points;
	// start viewing straight on
	_last_angle_X = 0.0;
	_last_angle_Y = 0.0;
	if(INTERACTIVE_MODE)
		RETURN_FRAMEBUFFER = false;
	else
		RETURN_FRAMEBUFFER = true;
}

Rasterizer::~Rasterizer() {
	std::cout << "Rasterizer::~Rasterizer()" << std::endl;
	delete [] _light_vector;
}

void Rasterizer::init() {
	std::cout << "Rasterizer::init()" << std::endl;
	glr_global_state_settings();
	init_program();
	glUseProgram(_the_program);
	glr_check_error();
	init_buffers();
	glr_check_error();
	std::cout << "Rasterizer::init_texture()" << std::endl;
	// choose which unit to use and activate it
	glr_init_texture(_textured_mesh.texture);
	glr_bind_texture_to_program(_textured_mesh.texture, _the_program);
	glr_check_error();
	if(RETURN_FRAMEBUFFER) {
		init_frame_buffer();
	}
	glr_check_error();
}

void Rasterizer::init_buffers() {
	std::cout << "Rasterizer::init_vertex_buffer()" << std::endl;

    glr_setup_buffers_on_textured_mesh(_textured_mesh);

	// --- SETUP COLORBUFFER (2)
	glGenBuffers(1, &_color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, _color_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_n_points*3, 
		_color, GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}


void Rasterizer::init_frame_buffer() {
	std::cout << "Rasterizer::init_frame_buffer()" << std::endl;
	glr_check_error();

	glGenFramebuffers(1, &_fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, _fbo);

	_fb_texture_unit = 2;
	glActiveTexture(GL_TEXTURE0 + _fb_texture_unit);
	glGenTextures(1, &_fb_texture);
	glBindTexture(GL_TEXTURE_2D, _fb_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
		GL_TEXTURE_2D, _fb_texture, 0);
	// THIS BEING GL_COLOR_ATTACHMENT0 means that anything rendered to
	// layout(location = 0) in the fragment shader will end up here.
	glr_check_error();
	glBindTexture(GL_TEXTURE_2D, 0);
	glGenTextures(1, &_fb_color);
	glBindTexture(GL_TEXTURE_2D, _fb_color);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glr_check_error();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 
		GL_RGB, GL_FLOAT, NULL);
	glr_check_error();
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 
		GL_TEXTURE_2D, _fb_color, 0);
	// THIS BEING GL_COLOR_ATTACHMENT1 means that anything rendered to
	// layout(location = 1) in the fragment shader will end up here.
	glBindTexture(GL_TEXTURE_2D, 0);
	glr_check_error();
	const GLenum buffs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	GLsizei buffsSize = 2;
	glDrawBuffers(buffsSize, buffs);
		// now, the depth buffer
	GLuint depthBuffer;
	glGenRenderbuffers(1,  &depthBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER,depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,depthBuffer);
	// THIS BEING GL_DEPTH_COMPONENT means that the depth information at each
	// fragment will end up here. Note that we must manually set up the depth
	// buffer when using framebuffers.
	GLenum status;
	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(status != GL_FRAMEBUFFER_COMPLETE)
	{
		printf("Framebuffer error: 0x%04X\n", status);
		//std::exit(EXIT_FAILURE);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Rasterizer::display() 
{
	std::cout << "Rasterizer::display()" << std::endl;
	if(RETURN_FRAMEBUFFER)
		glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
	else
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(_the_program);
	glBindVertexArray(_textured_mesh.vao);
	if(!RETURN_FRAMEBUFFER) {
		perspectiveMatrixUnif = glGetUniformLocation(_the_program, "perspectiveMatrix");
		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, perspectiveMatrix);
		rotationMatrixUinf = glGetUniformLocation(_the_program, "rotationMatrix");
		glUniformMatrix4fv(rotationMatrixUinf, 1, GL_FALSE, rotationMatrix);
		translationVectorUnif = glGetUniformLocation(_the_program, "translationVector");
		glUniform4fv(translationVectorUnif, 1, translationVector);
		GLuint lightDirectionUnif = glGetUniformLocation(_the_program, "lightDirection");
		glUniform3fv(lightDirectionUnif, 1, _light_vector);
	}
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _textured_mesh.trilist.vbo);
	glActiveTexture(GL_TEXTURE0 + _textured_mesh.texture.unit);
	glBindTexture(GL_TEXTURE_2D, _textured_mesh.texture.texture_ID);
	glDrawElements(GL_TRIANGLES, _textured_mesh.trilist.n_vectors * 3,
				   GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
	glutSwapBuffers();
	if(RETURN_FRAMEBUFFER)
		glutLeaveMainLoop();
}

void Rasterizer::init_program() {
	std::cout << "Rasterizer::init_program()" << std::endl;
	std::vector<GLuint> shaders;
	std::string vertex_shader_str;
	std::string fragment_shader_str;
	if(!RETURN_FRAMEBUFFER) {
		vertex_shader_str = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/interactive.vert";
		fragment_shader_str = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/interactive.frag";
	} else {
		vertex_shader_str = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/texture_shader.vert";
		fragment_shader_str = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/texture_shader.frag";
	}
	shaders.push_back(create_shader_from_filepath(GL_VERTEX_SHADER,   vertex_shader_str.c_str()));
	shaders.push_back(create_shader_from_filepath(GL_FRAGMENT_SHADER, fragment_shader_str.c_str()));
	_the_program = glr_create_program(shaders);
	std::for_each(shaders.begin(), shaders.end(), glDeleteShader);
}

void Rasterizer::cleanup() {
	std::cout << "Rasterizer::cleanup()" << std::endl;
	if(RETURN_FRAMEBUFFER)
		grab_framebuffer_data();
	glr_destroy_program();
	glr_destroy_vbos_on_trianglar_mesh(_textured_mesh);
}

void Rasterizer::grab_framebuffer_data() {
	std::cout << "Rasterizer::grab_framebuffer_data()" << std::endl;
	if(RETURN_FRAMEBUFFER) {
		glr_get_framebuffer(_fb_texture_unit, _fb_texture, GL_RGBA,
				         GL_UNSIGNED_BYTE, _fbo_pixels);
		glr_get_framebuffer(_fb_color_unit, _fb_color, GL_RGB,
				         GL_FLOAT, _fbo_color_pixels);
	} else {
		std::cout << "Trying to return FBO on an interactive session!"
		          << std::endl;
	}
}

void Rasterizer::render(int argc, char *argv[]) {
	std::cout << "Rasterizer::render()" << std::endl;
	if(!RETURN_FRAMEBUFFER) {
		float fFrustumScale = 1.0f; float fzNear = 0.5f; float fzFar = 10.0f;
		memset(perspectiveMatrix,0, sizeof(float) * 16);
		perspectiveMatrix[0] = fFrustumScale;
		perspectiveMatrix[5] = fFrustumScale;
		perspectiveMatrix[10] = (fzFar + fzNear) / (fzNear - fzFar);
		perspectiveMatrix[14] = (2 * fzFar * fzNear) / (fzNear - fzFar);
		perspectiveMatrix[11] = -1.0;
		memset(translationVector,0, sizeof(float) * 4);
		translationVector[2] = -2.0;
		start_framework(argc, argv);
	} else {
		std::cout << "Trying to render a RETURN_FRAMEBUFFER object!"
				  << std::endl;
	}
}

void Rasterizer::return_FB_pixels(int argc, char *argv[], uint8_t *pixels,
						          float *color_pixels, int width, int height) {
	std::cout << "Rasterizer::return_FB_pixels()" << std::endl;
	_fbo_pixels = pixels;
	_fbo_color_pixels = color_pixels;
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
	RETURN_FRAMEBUFFER = true;
	// set the rotation, perspective, and translation objects to
	// unitary (we just want orthogonal projection)
	memset(translationVector,0, sizeof(float) * 4);
	memset(perspectiveMatrix,0, sizeof(float) * 16);
	perspectiveMatrix[0]  = 1.0;
	perspectiveMatrix[5]  = 1.0;
	perspectiveMatrix[10] = 1.0;
	perspectiveMatrix[15] = 1.0;
	memset(rotationMatrix,0, sizeof(float) * 16);
	rotationMatrix[0]  = 1.0;
	rotationMatrix[5]  = 1.0;
	rotationMatrix[10] = 1.0;
	rotationMatrix[15] = 1.0;
	start_framework(argc, argv);
}

void Rasterizer::reshape(int width, int height) {
	std::cout << "Rasterizer::reshape(...)" << std::endl;
	// if in interactive mode -> adjust perspective matrix
	if(!RETURN_FRAMEBUFFER) {
		float fFrustumScale = 1.4;
		perspectiveMatrix[0] = fFrustumScale / (width / (float)height);
		perspectiveMatrix[5] = fFrustumScale;
		glUseProgram(_the_program);
		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, perspectiveMatrix);
		glUseProgram(0);
	}
    glViewport(0, 0, (GLsizei) width, (GLsizei) height);
}

void Rasterizer::mouseMove(int x, int y) {
	std::cout << "Rasterizer::mouseMove(...)" << std::endl;
	// if in interactive mode
	if(!RETURN_FRAMEBUFFER) {
		int width = glutGet(GLUT_WINDOW_WIDTH);
		int height = glutGet(GLUT_WINDOW_HEIGHT);
		float pi = atan2f(0.0,-1.0);
		//std::cout << "width: " << width << "\theight : " << height << std::endl;
		int deltaX = lastX - x;
		int deltaY = lastY - y;
		//std::cout << "dX: " << deltaX << "\tdY: " << deltaY << std::endl;

		angleX = _last_angle_X + (1.0*deltaY)*pi/height;
		angleY = _last_angle_Y + (1.0*deltaX)*pi/width;
	
		if(angleX < -pi/2)
			angleX = -pi/2;
		if(angleX > pi/2)
			angleX = pi/2;
		if(angleY < -pi/2)
			angleY = -pi/2;
		if(angleX > pi/2)
			angleX = pi/2;
		setRotationMatrixForAngleXAngleY(angleX,angleY);
		glutPostRedisplay();
	}
}

void Rasterizer::setRotationMatrixForAngleXAngleY(float angleX,float angleY) {
	rotationMatrix[5]  =  cos(angleX);
	rotationMatrix[6]  = -sin(angleX);
	rotationMatrix[9]  =  sin(angleX);
	rotationMatrix[10] =  cos(angleX);
	rotationMatrix[0]  =  cos(angleY);
	rotationMatrix[2]  =  sin(angleY);
	rotationMatrix[8] = -sin(angleY);
	rotationMatrix[10] =  cos(angleY);
}

void Rasterizer::mouseButtonPress(int button, int state, int x, int y) {
	std::cout << "Rasterizer::mouseButtonPress(...)" << std::endl;
	if(state) {
		std::cout << "Released"  << std::endl;
		// button let go - remember current angle
		_last_angle_X = angleX;
		_last_angle_Y = angleY;
	} else {
		std::cout << "Pressed" << std::endl;
		// button pressed - remember starting position
		lastX = x;
		lastY = y;
	}
}

void Rasterizer::keyboardDown(unsigned char key, int x, int y ) {
	std::cout << "Rasterizer::keyboardDown(...)" << std::endl;
	float pi = atan2f(0.0,-1.0);
	if(key == 32) { //space bar
		// reset the rotation to centre
		memset(rotationMatrix, 0, sizeof(float) * 16);
		rotationMatrix[0] = 1.0;
		rotationMatrix[5] = 1.0;
		rotationMatrix[10] = 1.0;
		rotationMatrix[15] = 1.0;
		glutPostRedisplay();
	} else if (key==27)// ESC key
        glutLeaveMainLoop ();
	else if (key == 'p') {
		setRotationMatrixForAngleXAngleY(-0.10,pi/9.0);
		glutPostRedisplay();
	} else if (key == 's') {
		setRotationMatrixForAngleXAngleY(0,pi/2.);
		glutPostRedisplay();
	} else
		std::cout << "Keydown: " << key << std::endl;
}
