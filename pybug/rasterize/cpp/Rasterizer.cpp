#include "Rasterizer.h"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <cmath>


GLuint create_program(const std::vector<GLuint> &shaderList) {
	GLuint program = glCreateProgram();
	for(size_t i = 0; i < shaderList.size(); i++)
		glAttachShader(program, shaderList[i]);
	glLinkProgram(program);
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		GLint infoLogLength;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);
		GLchar *strInfoLog = new GLchar[infoLogLength + 1];
		glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
		fprintf(stderr, "Linker failure: %s\n", strInfoLog);
		delete[] strInfoLog;
	}
	for(size_t i = 0; i < shaderList.size(); i++)
		glDetachShader(program, shaderList[i]);
	return program;
}


GLuint create_shader(GLenum eShaderType,  std::string &strShaderFilename){
	GLuint shader = glCreateShader(eShaderType);
	std::ifstream shaderFile(strShaderFilename.c_str(), std::ifstream::in);
	GLchar strFileData[10000];
	unsigned int i  = 0;
	while(shaderFile.good()) {
		strFileData[i] = shaderFile.get();
		std::cout << strFileData[i];
		i++;
	}
	strFileData[i-1] = '\0';
	const GLchar* constStrFileData = (const char*)strFileData;
	std::cout << constStrFileData << std::endl;
	glShaderSource(shader, 1, &constStrFileData, NULL);
	glCompileShader(shader);
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		GLint infoLogLength;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);
		GLchar *strInfoLog = new GLchar[infoLogLength + 1];
		glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);
		const char *strShaderType = NULL;
		switch (eShaderType) {
			case GL_VERTEX_SHADER:   strShaderType = "vertex";   break;
			case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
			case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
		}
		fprintf(stderr, "Compile failure in %s shader: \n%s\n",
			strShaderType, strInfoLog);
		delete[] strInfoLog;
		exit(EXIT_FAILURE);
	}
	return shader;
}

void check_error() {
	GLenum err;
	err = glGetError();
	if (err != GL_NO_ERROR) {
		printf("Error. glError: 0x%04X", err);
		std::cout << " - " << gluErrorString(err) << std::endl;
		exit(EXIT_FAILURE);
	}
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


Rasterizer::Rasterizer(double* points, float* color, size_t n_points,
					   unsigned int* trilist, size_t n_tris, float* tcoords,
					   uint8_t* texture, size_t texture_width,
					   size_t texture_height, bool INTERACTIVE_MODE) {
	_light_vector = new float[3];
	memset(_light_vector,0,3);
	_light_vector[2] = 1.0;

	title = "MM3D Viewer";
	std::cout << "Rasterizer::Rasterizer" << std::endl;
	_h_points = points;
	_color = color;
	_trilist = trilist;
	_n_points = n_points;
	_n_tris = n_tris;
	_tcoords = tcoords;
	_texture = texture;
	_texture_width = texture_width;
	_texture_height = texture_height;
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
	check_error();
	glEnable (GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
	glGenVertexArrays(1, &_vao);
	check_error();
	glBindVertexArray(_vao);
	check_error();
	init_program();
	glUseProgram(_the_program);
	check_error();
	init_vertex_buffer();
	check_error();
	init_texture();
	check_error();
	if(RETURN_FRAMEBUFFER)
	{
		glDepthFunc(GL_LEQUAL);
		init_frame_buffer();
	}
	check_error();
}

void Rasterizer::init_vertex_buffer() {
	// --- SETUP TPSCOORDBUFFER (0)
	glGenBuffers(1, &_points_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, _points_buffer);
	// allocate enough memory to store tpsCoord to the GL_ARRAY_BUFFER
	// target (which due to the above line is tpsCoordBuffer) and store it
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLdouble)*_n_points*4, 
		_h_points, GL_STATIC_DRAW);
	// enable the coord array (will be location = 0 in shader)
	glEnableVertexAttribArray(0);
	//prescribe how the data is stored
	glVertexAttribPointer(0, 4, GL_DOUBLE, GL_FALSE, 0, 0);
	// detatch from GL_ARRAY_BUFFER (good practice)
	glBindBuffer(GL_ARRAY_BUFFER, 0);

//	if(!TEXTURE_IMAGE)
//	{
//		// --- SETUP TEXTUREVECTORBUFFER (1)
//		glGenBuffers(1, &_textureVectorBuffer);
//		glBindBuffer(GL_ARRAY_BUFFER, _textureVectorBuffer);
//		glBufferData(GL_ARRAY_BUFFER, sizeof(GLdouble)*_n_points*4,
//			textureVector, GL_STATIC_DRAW);
//		glEnableVertexAttribArray(1);
//		glVertexAttribPointer(1, 4, GL_DOUBLE, GL_FALSE, 0, 0);
//		glBindBuffer(GL_ARRAY_BUFFER, 0);
//	}
	// --- SETUP TCOORDBUFFER (1)
	glGenBuffers(1, &_tcoord_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, _tcoord_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_n_points*2,
		_tcoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// --- SETUP COLORBUFFER (2)
	glGenBuffers(1, &_color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, _color_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*_n_points*3, 
		_color, GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &_trilist_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _trilist_buffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*_n_tris*3, 
		_trilist, GL_STATIC_DRAW);
}

void Rasterizer::init_texture()
{
	// choose which unit to use and activate it
	_texture_unit = 1;
	glActiveTexture(GL_TEXTURE0 + _texture_unit);
	// specify the data storage and actually get OpenGL to 
	// store our textureImage
	glGenTextures(1, &_texture_ID);
	glBindTexture(GL_TEXTURE_2D, _texture_ID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
		_texture_width, _texture_height, 0, GL_RGBA, 
		GL_UNSIGNED_BYTE, _texture);

	// Create the description of the texture (sampler) and bind it to the 
	// correct texture unit
	glGenSamplers(1, &_texture_sampler);
	glSamplerParameteri(_texture_sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glSamplerParameteri(_texture_sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glSamplerParameteri(_texture_sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glBindSampler(_texture_unit, _texture_sampler);
    // bind the texture to a uniform called "textureImage" which can be
	// accessed from shaders
	_texture_uniform = glGetUniformLocation(_the_program, "textureImage");
	glUniform1i(_texture_uniform, _texture_unit);

	// set the active Texture to 0 - as long as this is not changed back
	// to textureImageUnit, we know our shaders will find textureImage bound to
	// GL_TEXTURE_2D when they look in textureImageUnit
	glActiveTexture(GL_TEXTURE0);
	// note now we are free to unbind GL_TEXTURE_2D
	// on unit 0 - the state of our textureUnit is safe.
	glBindTexture(GL_TEXTURE_2D, 0);
}

void Rasterizer::init_frame_buffer()
{
	check_error();

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
	check_error();
	glBindTexture(GL_TEXTURE_2D, 0);
	glGenTextures(1, &_fb_color);
	glBindTexture(GL_TEXTURE_2D, _fb_color);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	check_error();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 
		GL_RGB, GL_FLOAT, NULL);
	check_error();
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 
		GL_TEXTURE_2D, _fb_color, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	check_error();
	const GLenum buffs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	GLsizei buffsSize = 2;
	glDrawBuffers(buffsSize, buffs);
		// now, the depth buffer
	GLuint depthBuffer;
	glGenRenderbuffers(1,  &depthBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER,depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,depthBuffer);
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
	if(!RETURN_FRAMEBUFFER) {
		perspectiveMatrixUnif = glGetUniformLocation(_the_program, "perspectiveMatrix");
		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, perspectiveMatrix);
		rotationMatrixUinf = glGetUniformLocation(_the_program, "rotationMatrix");
		glUniformMatrix4fv(rotationMatrixUinf, 1, GL_FALSE, rotationMatrix);
		translationVectorUnif = glGetUniformLocation(_the_program, "translationVector");
		glUniform4fv(translationVectorUnif, 1, translationVector);
		GLuint lightDirectionUnif = glGetUniformLocation(_the_program, "lightDirection");
		glUniform3fv(lightDirectionUnif, 1, _light_vector);
		printUnitTests();
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _trilist_buffer);
	glActiveTexture(GL_TEXTURE0 + _texture_unit);
	glBindTexture(GL_TEXTURE_2D, _texture_ID);
	glDrawElements(GL_TRIANGLES, _n_tris*3, GL_UNSIGNED_INT, 0);
	glutSwapBuffers();
	if(RETURN_FRAMEBUFFER)
		glutLeaveMainLoop();
}

void Rasterizer::printUnitTests() {
	float* input = new float[4];
	memset(input,0.,4);
	input[0] = 0;
	input[1] = 0;
	input[2] = 1;
	input[3] = 1;

	float * result = new float[4];
	float * tempResult = new float[4];	
	matrix_x_vector(rotationMatrix,input,tempResult);
	for(int i = 0; i < 4; i++)
		tempResult[i] += translationVector[i];
	matrix_x_vector(perspectiveMatrix,tempResult,result);
	for(int i = 0; i < 4; i ++)
		printf("%2.2f\t%2.2f\t%2.2f\n",input[i],tempResult[i]-translationVector[i],result[i]);
	std::cout << std::endl;
	delete [] input;
	delete [] tempResult;
	delete [] result;
}

void Rasterizer::init_program() {
	std::cout << "Rasterizer::init_program()" << std::endl;
	std::vector<GLuint> shaderList;
	std::string strVertexShader;
	std::string strFragmentShader;
	if(!RETURN_FRAMEBUFFER) {
		strVertexShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/interactive.vert";
		strFragmentShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/interactive.frag";
	} else {
		strVertexShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/textureImage.vert";
		strFragmentShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/textureImage.frag";
	}
	shaderList.push_back(create_shader(GL_VERTEX_SHADER,   strVertexShader  ));
	shaderList.push_back(create_shader(GL_FRAGMENT_SHADER, strFragmentShader));
	_the_program = create_program(shaderList);
	std::for_each(shaderList.begin(), shaderList.end(), glDeleteShader);
	std::cout << "done." << std::endl;
}

void Rasterizer::cleanup() {
	std::cout << "Rasterizer::cleanup()" << std::endl;
	if(RETURN_FRAMEBUFFER)
		grab_framebuffer_data();
	destroy_shaders();
	destroy_VBO();
}

void Rasterizer::grab_framebuffer_data() {
	if(RETURN_FRAMEBUFFER) {
		glActiveTexture(GL_TEXTURE0 + _fb_texture_unit);
		glBindTexture(GL_TEXTURE_2D, _fb_texture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, _fbo_pixels);
		glActiveTexture(GL_TEXTURE0 + _fb_color_unit);
		glBindTexture(GL_TEXTURE_2D, _fb_color);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, _fbo_color_pixels);
	} else
		std::cout << "Trying to return FBO on an interactive session!" << std::endl;
}

void Rasterizer::destroy_shaders() {
	glUseProgram(0);
}

void Rasterizer::destroy_VBO() {
	GLenum errorCheckValue = glGetError();
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &_textureVectorBuffer);
	glDeleteBuffers(1, &_color_buffer);
	glDeleteBuffers(1, &_points_buffer);
	glDeleteBuffers(1, &_trilist_buffer);
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &_vao);
	errorCheckValue = glGetError();
    if (errorCheckValue != GL_NO_ERROR) {
        fprintf(stderr, "ERROR: Could not destroy the VBO: %s \n",
                gluErrorString(errorCheckValue));
        exit(-1);
    }
}

void Rasterizer::render(int argc, char *argv[]) {
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
		startFramework(argc, argv);
	} else
		std::cout << "Trying to render a RETURN_FRAMEBUFFER object!"
				  << std::endl;
}

void Rasterizer::return_FB_pixels(int argc, char *argv[], uint8_t *pixels,
						          float *color_pixels, int width, int height) {
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
	startFramework(argc, argv);
}

void Rasterizer::reshape(int width, int height) {
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

void Rasterizer::keyboardDown( unsigned char key, int x, int y ) {
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

