#include "Rasterizer.h"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <cmath>

void read_file(const char* filepath, GLchar* file_string) {
	std::ifstream file(filepath, std::ifstream::in);
	unsigned int i  = 0;
	while(file.good()) {
		file_string[i] = file.get();
		i++;
	}
	file_string[i-1] = '\0';
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
	std::cout << "Rasterizer::init_buffers()" << std::endl;

    glr_init_buffers_from_textured_mesh(_textured_mesh);

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

	// first, build a texture:
	_texture_fb = glr_build_rgba_texture(_fb_texture_pixels, WINDOW_WIDTH,
			WINDOW_HEIGHT);
	// assign it to a new unit
	_texture_fb.unit = 2;
	// and initialise it
	glr_init_texture(_texture_fb);
	// now we can bind to the active framebuffer.
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
		GL_TEXTURE_2D, _texture_fb.texture_ID, 0);

//	// repeat for the position rendering
//	_texture_fb_color = glr_build_rgb_float_texture(_fbo_color_pixels,
//			WINDOW_WIDTH, WINDOW_HEIGHT);
//    _texture_fb_color.unit = 3;
//    glr_init_texture(_texture_fb_color);
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
//		GL_TEXTURE_2D, _texture_fb_color.texture_ID, 0);

//	// make a new texture (as normal)
//	glBindTexture(GL_TEXTURE_2D, 0);
//	_fb_texture_unit = 2;
//	glActiveTexture(GL_TEXTURE0 + _fb_texture_unit);
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
//	// attach the texture to the framebuffer
//	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
//		GL_TEXTURE_2D, _fb_texture, 0);
//    // THIS BEING GL_COLOR_ATTACHMENT0 means that anything rendered to
//    // layout(location = 0) in the fragment shader will end up here.
//	glr_check_error();
//	glBindTexture(GL_TEXTURE_2D, 0);



	glGenTextures(1, &_fb_color_id);
	glBindTexture(GL_TEXTURE_2D, _fb_color_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glr_check_error();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, WINDOW_WIDTH, WINDOW_HEIGHT, 0,
		GL_RGB, GL_FLOAT, NULL);
	glr_check_error();


	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
		GL_TEXTURE_2D, _fb_color_id, 0);
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
	if(status != GL_FRAMEBUFFER_COMPLETE) {
		printf("Framebuffer error: 0x%04X\n", status);
		std::exit(EXIT_FAILURE);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Rasterizer::display() {
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
		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, _m_perspective);
		rotationMatrixUinf = glGetUniformLocation(_the_program, "rotationMatrix");
		glUniformMatrix4fv(rotationMatrixUinf, 1, GL_FALSE, _m_rotation);
		translationVectorUnif = glGetUniformLocation(_the_program, "translationVector");
		glUniform4fv(translationVectorUnif, 1, _v_translation);
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
//		glr_get_framebuffer(_fb_texture_unit, _fb_texture, GL_RGBA,
//				         GL_UNSIGNED_BYTE, _fbo_pixels);
		glr_get_framebuffer(_fb_color_unit, _fb_color_id, GL_RGB,
				         GL_FLOAT, _fbo_color_pixels);
		glr_get_framebuffer(_texture_fb.unit, _texture_fb.texture_ID,
				_texture_fb.format, _texture_fb.type, _texture_fb.data);
//		glr_get_framebuffer(_texture_fb_color.unit,
//				_texture_fb_color.texture_ID, _texture_fb_color.format,
//				_texture_fb_color.type, _texture_fb.data);
	} else {
		std::cout << "Trying to return FBO on an interactive session!"
		          << std::endl;
	}
}

void Rasterizer::render(int argc, char *argv[]) {
	std::cout << "Rasterizer::render()" << std::endl;
	if(!RETURN_FRAMEBUFFER) {
		float fFrustumScale = 1.0f; float fzNear = 0.5f; float fzFar = 10.0f;
		memset(_m_perspective,0, sizeof(float) * 16);
		_m_perspective[0] = fFrustumScale;
		_m_perspective[5] = fFrustumScale;
		_m_perspective[10] = (fzFar + fzNear) / (fzNear - fzFar);
		_m_perspective[14] = (2 * fzFar * fzNear) / (fzNear - fzFar);
		_m_perspective[11] = -1.0;
		memset(_v_translation,0, sizeof(float) * 4);
		_v_translation[2] = -2.0;
		start_framework(argc, argv);
	} else {
		std::cout << "Trying to render a RETURN_FRAMEBUFFER object!"
				  << std::endl;
	}
}

void Rasterizer::return_FB_pixels(int argc, char *argv[], uint8_t *pixels,
						          float *color_pixels, int width, int height) {
	std::cout << "Rasterizer::return_FB_pixels()" << std::endl;
	_fb_texture_pixels = pixels;
	_fbo_color_pixels = color_pixels;
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
	RETURN_FRAMEBUFFER = true;
	// set the rotation, perspective, and translation objects to
	// unitary (we just want orthogonal projection)
	memset(_v_translation,0, sizeof(float) * 4);
    glr_math_float_matrix_eye(_m_perspective);
    glr_math_float_matrix_eye(_m_rotation);
	start_framework(argc, argv);
}

void Rasterizer::reshape(int width, int height) {
	std::cout << "Rasterizer::reshape(...)" << std::endl;
	// if in interactive mode -> adjust perspective matrix
	if(!RETURN_FRAMEBUFFER) {
		float fFrustumScale = 1.4;
		_m_perspective[0] = fFrustumScale / (width / (float)height);
		_m_perspective[5] = fFrustumScale;
		glUseProgram(_the_program);
		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, _m_perspective);
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
		glr_math_float_matrix_rotation_for_angles(_m_rotation, angleX, angleY);
		glutPostRedisplay();
	}
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
        glr_math_float_matrix_eye(_m_rotation);
		glutPostRedisplay();
	} else if (key==27)// ESC key
        glutLeaveMainLoop ();
	else if (key == 'p') {
		glr_math_float_matrix_rotation_for_angles(_m_rotation, -0.10, pi/9.0);
		glutPostRedisplay();
	} else if (key == 's') {
		glr_math_float_matrix_rotation_for_angles(_m_rotation, 0, pi/2.);
		glutPostRedisplay();
	} else
		std::cout << "Keydown: " << key << std::endl;
}
