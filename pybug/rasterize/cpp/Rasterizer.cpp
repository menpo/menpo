#include "Rasterizer.h"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <cmath>

Rasterizer *Rasterizer::instance = NULL;

// ----- OPENGL HELPER FUNCTIONS ------ //

GLuint createProgram(const std::vector<GLuint> &shaderList) {
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

GLuint createShader(GLenum eShaderType,
		                          std::string &strShaderFilename) {
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

void checkError() {
    GLenum err;
    err = glGetError();
    if (err != GL_NO_ERROR) {
        printf("Error. glError: 0x%04X", err);
        std::cout << " - " << gluErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

Rasterizer::Rasterizer(double* points, float* color,  size_t num_points,
					   unsigned int* trilist, size_t num_tris,
					   float* tcoords, uint8_t* texture,
					   size_t texture_width, size_t texture_height){
    WINDOW_WIDTH = 768;
    WINDOW_HEIGHT = 768;
    WINDOW_X_POSITION = 100;
    WINDOW_Y_POSITION = 100;
    _title = "OpenGL Rasterizer";
    _points = points;
    _color_f = color;
    _trilist = trilist;
    _num_points = num_points;
    _num_tris = num_tris;
    _tcoords = tcoords;
    _texture = texture;
    _texture_width = texture_width;
    _texture_height = texture_height;
}

void Rasterizer::init(){
    checkError();
    glEnable (GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
    glGenVertexArrays(1, &_vao);
    checkError();
    glBindVertexArray(_vao);
    checkError();
    initialize_program();
    glUseProgram(_the_program);
    checkError();
    initialize_vertex_buffer();
    checkError();
    initialize_texture();
    checkError();
	glDepthFunc(GL_LEQUAL);
	initialize_frame_buffer();
    checkError();
}

void Rasterizer::setInstance() {
    instance = this;
}

void Rasterizer::startFramework(int argc, char *argv[]) {
    setInstance();
    // Sets the instance to self, used in the callback wrapper functions
    std::cout << "Calling startFramework in GLRFramework" << std::endl;
    // Fire up GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    //glutInitContextVersion(4, 0);
    //glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitWindowPosition(WINDOW_X_POSITION, WINDOW_Y_POSITION);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow(_title.c_str());

    // Fire up GLEW
    GLenum status = glewInit();
    if (status != GLEW_OK) {
        fprintf(stderr, "GLEW Failed to start! Error: %s\n", glewGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    fprintf(stdout, "Using GLEW %s\n", glewGetString(GLEW_VERSION));

    // Set up function callbacks with wrapper functions
    glutCloseFunc(cleanupWrapper);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    fprintf(stdout,"OpenGL Version: %s\n",glGetString(GL_VERSION));

    // Call subclasses init
    init();

    // Start the main GLUT thread
    glutMainLoop();
}

// ******************************************************************
// ** Static functions which are passed to Glut function callbacks **
// ******************************************************************

void Rasterizer::displayWrapper() {
    instance->display();
}

void Rasterizer::cleanupWrapper() {
    instance->cleanup();
}

void Rasterizer::initialize_vertex_buffer(){
    // --- SETUP POINTS BUFFER (0)
    glGenBuffers(1, &_points_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, _points_buffer);
    // allocate enough memory to store points to the GL_ARRAY_BUFFER
    // target (which due to the above line is points_buffer) and store it
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLdouble)*_num_points*4,
            _points, GL_STATIC_DRAW);
    // enable the point array (will be location = 0 in shader)
    glEnableVertexAttribArray(0);
    //prescribe how the data is stored
    glVertexAttribPointer(0, 4, GL_DOUBLE, GL_FALSE, 0, 0);
    // Detach from GL_ARRAY_BUFFER (good practice)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
	// TCOORD BUFFER (1)
	glGenBuffers(1, &_tcoord_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, _tcoord_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * _num_points * 2,
			_tcoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
    // SETUP COORDBUFFER (2)
    glGenBuffers(1, &_color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, _color_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * _num_points * 3,
            _color_f, GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // TRILIST BUFFER
    glGenBuffers(1, &_trilist_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _trilist_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * _num_tris * 3,
            _trilist, GL_STATIC_DRAW);
}

void Rasterizer::initialize_texture(){
    // choose which unit to use and activate it
    _texture_image_unit = 1;
    glActiveTexture(GL_TEXTURE0 + _texture_image_unit);
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
    glBindSampler(_texture_image_unit, _texture_sampler);
    // bind the texture to a uniform called "textureImage" which can be
    // accessed from shaders
    _textureUniform = glGetUniformLocation(_the_program, "texture_image");
    glUniform1i(_textureUniform, _texture_image_unit);

    // set the active Texture to 0 - as long as this is not changed back
    // to textureImageUnit, we know our shaders will find textureImage bound to
    // GL_TEXTURE_2D when they look in textureImageUnit
    glActiveTexture(GL_TEXTURE0);
    // note now we are free to unbind GL_TEXTURE_2D
    // on unit 0 - the state of our textureUnit is safe.
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Rasterizer::initialize_frame_buffer(){
    checkError();
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
    checkError();
    glBindTexture(GL_TEXTURE_2D, 0);
    glGenTextures(1, &_fb_coord);
    glBindTexture(GL_TEXTURE_2D, _fb_coord);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    checkError();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, WINDOW_WIDTH, WINDOW_HEIGHT, 0,
            GL_RGB, GL_FLOAT, NULL);
    checkError();
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
            GL_TEXTURE_2D, _fb_coord, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    checkError();
    const GLenum buffs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    GLsizei buffsSize = 2;
    glDrawBuffers(buffsSize, buffs);
    // now, the depth buffer
    GLuint depth_buffer;
    glGenRenderbuffers(1,  &depth_buffer);
    glBindRenderbuffer(GL_RENDERBUFFER,depth_buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH,
    					  WINDOW_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
    					      GL_RENDERBUFFER, depth_buffer);
    GLenum status;
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if(status != GL_FRAMEBUFFER_COMPLETE)
    {
        printf("Framebuffer error: 0x%04X\n", status);
        //std::exit(EXIT_FAILURE);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Rasterizer::display(){
    //std::cout << "Calling the Rasterizer display method" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(_the_program);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _trilist_buffer);
    glActiveTexture(GL_TEXTURE0 + _texture_image_unit);
    glBindTexture(GL_TEXTURE_2D, _texture_ID);
    glDrawElements(GL_TRIANGLES, _num_tris*3, GL_UNSIGNED_INT, 0);
    glutSwapBuffers();
    glutLeaveMainLoop();
}

void Rasterizer::initialize_program(){
    std::cout << "initialize_program()...";
    std::vector<GLuint> shaderList;
    std::string strVertexShader;
    std::string strFragmentShader;
    strVertexShader = "/home/jab08/gits/msc_project/matlab/GLRenderer/textureImage.vert";
    strFragmentShader = "/home/jab08/gits/msc_project/matlab/GLRenderer/textureImage.frag";
    shaderList.push_back(createShader(GL_VERTEX_SHADER,   strVertexShader  ));
    shaderList.push_back(createShader(GL_FRAGMENT_SHADER, strFragmentShader));

    _the_program = createProgram(shaderList);

    std::for_each(shaderList.begin(), shaderList.end(), glDeleteShader);
    std::cout << "done." << std::endl;
}

void Rasterizer::cleanup(){
    std::cout << "Rasterizer::cleanup()" << std::endl;
    grabFrameBufferData();
    destroy_shaders();
    destroy_VBO();
}

void Rasterizer::grabFrameBufferData(){
    glActiveTexture(GL_TEXTURE0 + _fb_texture_unit);
    glBindTexture(GL_TEXTURE_2D, _fb_texture);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, _fbo_pixels);
    glActiveTexture(GL_TEXTURE0 + _fb_coord_unit);
    glBindTexture(GL_TEXTURE_2D, _fb_coord);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, _fbo_coords);
}

void Rasterizer::destroy_shaders(){
    glUseProgram(0);
}

void Rasterizer::destroy_VBO(){
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
    if (errorCheckValue != GL_NO_ERROR){
        fprintf(stderr,
                "ERROR: Could not destroy the VBO: %s \n",
                gluErrorString(errorCheckValue)
               );
        exit(-1);
    }
}

void Rasterizer::return_FB_pixels(uint8_t *fboPixels, float *fboCoords,
								  int width, int height){
    _fbo_pixels = fboPixels;
    _fbo_coords = fboCoords;
    WINDOW_WIDTH = width;
    WINDOW_HEIGHT = height;
    char *blank  = "blank";
    startFramework(0, &blank);
}
