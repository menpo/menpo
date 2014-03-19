#pragma once

#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <GL/glew.h>
#include <vector>
#include <GL/freeglut.h>

GLuint createShader(GLenum eShaderType, std::string &strShaderFile);
GLuint createProgram(const std::vector<GLuint> &shaderList);
void checkError();


class Rasterizer {
    private:
    	std::string _title;
        double*  _points;  // (X,Y,Z,W) position
        float* _color_f;  // (X,Y,Z) float value per point
        double*  _textureVector;  // (R,G,B,A) color per point (redundant if texture is used)
        float* _tcoords;  // (s,t) texture coordinates
        uint8_t* _texture;
        GLuint* _trilist;

        size_t _num_points;
        size_t _num_tris;
        size_t _texture_width;
        size_t _texture_height;
        int _texture_image_unit;  // what Texture Unit the texture is bound to

        // --- Handles to GL objects ---
        GLuint _vao;
        GLuint _the_program;
        GLuint _perspectiveMatrixUnif;
        GLuint _translationVectorUnif;
        GLuint _rotationMatrixUinf;

        GLuint _points_buffer;
        GLuint _tcoord_buffer;
        GLuint _trilist_buffer;
        GLuint _color_buffer;
        GLuint _textureVectorBuffer;

        GLuint _texture_ID;
        // object storing texture traits.
        GLuint _texture_sampler;

        //fbo parameters
        GLuint _fbo;
        GLuint _fb_texture;
        GLuint _fb_coord;
        int _fb_texture_unit;
        int _fb_coord_unit;
        GLubyte* _fbo_pixels;
        GLfloat* _fbo_coords;

        GLuint _textureUniform;  // Handles to uniforms


    protected:
        static Rasterizer *instance;

    public:
        int WINDOW_WIDTH;
        int WINDOW_HEIGHT;
        int WINDOW_X_POSITION;
        int WINDOW_Y_POSITION;

        Rasterizer(double* points, float* coord, size_t n_coords,
                unsigned int* trilist, size_t num_tris,
                float* texCoord_in, uint8_t* textureImage_in,
                size_t textureWidth_in, size_t textureHeight_in);
        void return_FB_pixels(uint8_t* pixels, float* coords, int width, int height);

    private:
        void init();
        void initialize_vertex_buffer();
        void display();
        void setInstance();
        void initialize_program();
        void initialize_texture();
        void initialize_frame_buffer();
        void cleanup();
        void destroy_shaders();
        void destroy_VBO();
        void grabFrameBufferData();
        void startFramework(int argc, char *argv[]);
        static void displayWrapper();
        static void reshapeWrapper(int width, int height);
        static void cleanupWrapper();
};

