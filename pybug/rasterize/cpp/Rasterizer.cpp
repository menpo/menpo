#include "Rasterizer.h"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <cmath>

Rasterizer::Rasterizer(double* points, float* color,  size_t num_points,
					   unsigned int* trilist, size_t num_tris,
					   float* tcoords, uint8_t* texture,
					   size_t texture_width, size_t texture_height){
    _light_vector = new float[3];
    memset(_light_vector, 0, 3);
    _light_vector[2] = 1.0;
    _title = "OpenGL Rasterizer";
    _TEXTURE_IMAGE = true;
    std::cout << "Rasterizer::Rasterizer(TextureImage)" << std::endl;
    _points = points;
    _color_f = color;
    _trilist = trilist;
    _num_points = num_points;
    _num_tris = num_tris;
    _tcoords = tcoords;
    _texture = texture;
    _texture_width = texture_width;
    _texture_height = texture_height;
    // start viewing straight on
    _last_angle_X = 0.0;
    _last_angle_Y = 0.0;
}

Rasterizer::~Rasterizer(){
    std::cout << "Rasterizer::~Rasterizer()" << std::endl;
    delete [] _light_vector;
}

void Rasterizer::init(){
    std::cout << "Rasterizer::init()" << std::endl;
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
    if(_TEXTURE_IMAGE)
        initialize_texture();
    checkError();
	glDepthFunc(GL_LEQUAL);
	initialize_frame_buffer();
    checkError();
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

    if(!_TEXTURE_IMAGE){
        //  TEXTUREVECTORBUFFER (1)
        glGenBuffers(1, &_textureVectorBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, _textureVectorBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLdouble)*_num_points*4,
                _textureVector, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_DOUBLE, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    else{
        // TCOORD BUFFER (1)
        glGenBuffers(1, &_tcoord_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, _tcoord_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * _num_points * 2,
                _tcoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // SETUP COORDBUFFER (2)
    glGenBuffers(1, &_coordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, _coordBuffer);
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
    _textureUniform = glGetUniformLocation(_the_program, "textureImage");
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

void Rasterizer::printUnitTests(){
    float* input = new float[4];
    memset(input,0.,4);
    input[0] = 0;
    input[1] = 0;
    input[2] = 1;
    input[3] = 1;
    float * result = new float[4];
    float * tempResult = new float[4];
    matrix_times_vector(_rotation_matrix,input,tempResult);
    for(int i = 0; i < 4; i++)
        tempResult[i] += _translation_vector[i];
    matrix_times_vector(_perspective_matrix,tempResult,result);
    for(int i = 0; i < 4; i ++)
        printf("%2.2f\t%2.2f\t%2.2f\n",input[i],tempResult[i]-_translation_vector[i],result[i]);
    std::cout << std::endl;
    delete [] input;
    delete [] tempResult;
    delete [] result;
}

void Rasterizer::matrix_times_vector(float* m, float* v, float*m_x_v){
    m_x_v[0] = 0;
    m_x_v[1] = 0;
    m_x_v[2] = 0;
    m_x_v[3] = 0;
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
            m_x_v[i] += m[4*i+ j]*v[j];
        }
    }
}

void Rasterizer::initialize_program(){
    std::cout << "initialize_program()...";
    std::vector<GLuint> shaderList;
    std::string strVertexShader;
    std::string strFragmentShader;
    if(_TEXTURE_IMAGE){
        strVertexShader = "/home/jab08/gits/msc_project/matlab/GLRenderer/textureImage.vert";
        strFragmentShader = "/home/jab08/gits/msc_project/matlab/GLRenderer/textureImage.frag";
    }
    else{
        strVertexShader = "/home/jab08/gits/msc_project/matlab/GLRenderer/shader.vert";
        strFragmentShader = "/home/jab08/gits/msc_project/matlab/GLRenderer/shader.frag";
    }
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
    glDeleteBuffers(1, &_coordBuffer);
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
    // set the rotation, perspective, and translation objects to unitary
    //(we just want orthogonal projection)
    memset(_translation_vector,0, sizeof(float) * 4);
    memset(_perspective_matrix,0, sizeof(float) * 16);
    _perspective_matrix[0]  = 1.0;
    _perspective_matrix[5]  = 1.0;
    _perspective_matrix[10] = 1.0;
    _perspective_matrix[15] = 1.0;
    memset(_rotation_matrix,0, sizeof(float) * 16);
    _rotation_matrix[0]  = 1.0;
    _rotation_matrix[5]  = 1.0;
    _rotation_matrix[10] = 1.0;
    _rotation_matrix[15] = 1.0;
    char *blank  = "blank";
    startFramework(0, &blank);
}
