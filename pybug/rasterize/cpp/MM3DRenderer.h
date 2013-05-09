#pragma once
#include "GLRFramework.h"
#include <vector>
#include <stdint.h>

class MM3DRenderer : public GLRFramework
{

    private:
        // stores (X,Y,Z,W) position tpsCoordinates
        double*  tpsCoord;
        // (X,Y,Z) procrustes alligned coords (non tps'd)
        float* coord;
        // provides an (R,G,B,A) color for each coord
        // (redundent if textureImage is used)
        double*  textureVector;
        // (s,t) texture coords
        float* texCoord;
        uint8_t* textureImage;
        // index into coord/textureVector
        GLuint* coordIndex;

        // vector to the direction of light
        float* lightVector;

        size_t  numCoord;
        size_t  numTriangles;
        size_t textureWidth;
        size_t textureHeight;
        // what Texture Unit we wish to bind the texture to
        int textureImageUnit;

        // --- Handles to GL objects ---
        GLuint vao;
        GLuint theProgram;
        GLuint perspectiveMatrixUnif;
        GLuint translationVectorUnif;
        GLuint rotationMatrixUinf;

        GLuint tpsCoordBuffer;
        GLuint coordBuffer;
        GLuint textureVectorBuffer;
        GLuint texCoordBuffer;

        GLuint indexBuffer;

        GLuint textureImageID;
        // object storing texture traits.
        GLuint textureSampler;

        //fbo parameters
        GLuint fbo;
        GLuint fbTexture;
        GLuint fbCoord;
        int fbTextureUnit;
        int fbCoordUnit;
        GLubyte* fboPixels;
        GLfloat* fboCoords;

        // Handles to uniforms
        GLuint textureUniform;

        bool TEXTURE_IMAGE;
        // if true we are rendering to just return the framebuffer.
        bool RETURN_FRAMEBUFFER;

        // variables tracking last place pressed
        int lastX, lastY;
        float lastAngleX, lastAngleY;
        float angleX, angleY;
    public:
        // basic constructor only taking in coords and a textureVector
        //MM3DRenderer(double* tpsCoord_in, float* coord_in, size_t numCoords_in, double* textureVector_in,
        //	unsigned int* coordIndex_in, size_t numTriangles_in);
        // constructor taking in textureImage
        MM3DRenderer(double* tpsCoord_in, float* coord_in, size_t numCoords_in,
                unsigned int* coordIndex_in, size_t numTriangles_in,
                float* texCoord_in, uint8_t* textureImage_in,
                size_t textureWidth_in, size_t textureHeight_in);
        void returnFBPixels(uint8_t* pixels, float* coords, int width, int height);
        ~MM3DRenderer();

    private:
        void init();
        void initializeVertexBuffer();
        void display();
        void initializeProgram();
        void initializeTexture();
        void initializeFrameBuffer();
        void cleanup();
        void destroyShaders();
        void destroyVBO();
        void grabFrameBufferData();
        void printUnitTests();
        void matrixTimesVector(float* matrix, float* vector, float*result);
};

