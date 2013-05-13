#pragma once
#include "GLRFramework.h"
#include <vector>
#include <stdint.h>

class Rasterizer : public GLRFramework {
    private:
        double*  _points;  // (X,Y,Z,W) position
        float* _color_f;  // (X,Y,Z) float value per point
        double*  _textureVector;  // (R,G,B,A) color per point (redundant if texture is used)
        float* _tcoords;  // (s,t) texture coordinates
        uint8_t* _texture;
        GLuint* _trilist;
        float* _light_vector;  // vector to the direction of light

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
        GLuint _coordBuffer;
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

        bool _TEXTURE_IMAGE;

        // variables tracking last place pressed
        int _last_X, _last_Y;
        float _last_angle_X, _last_angle_Y;
        float _angle_X, _angle_Y;

    public:
        // basic constructor only taking in coords and a textureVector
        //MM3DRenderer(double* tpsCoord_in, float* coord_in, size_t numCoords_in, double* textureVector_in,
        //	unsigned int* coordIndex_in, size_t numTriangles_in);
        // constructor taking in textureImage
        Rasterizer(double* points, float* coord, size_t n_coords,
                unsigned int* trilist, size_t num_tris,
                float* texCoord_in, uint8_t* textureImage_in,
                size_t textureWidth_in, size_t textureHeight_in);
        void return_FB_pixels(uint8_t* pixels, float* coords, int width, int height);
        ~Rasterizer();

    private:
        void init();
        void initialize_vertex_buffer();
        void display();
        void initialize_program();
        void initialize_texture();
        void initialize_frame_buffer();
        void cleanup();
        void destroy_shaders();
        void destroy_VBO();
        void grabFrameBufferData();
        void printUnitTests();
        void matrix_times_vector(float* matrix, float* vector, float* result);
};

