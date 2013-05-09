#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <GL/glew.h>
#include <vector>
#include <GL/freeglut.h> 

class GLRFramework {

    protected:
        std::string title;
        static GLRFramework *instance;

    public:
        int WINDOW_WIDTH;
        int WINDOW_HEIGHT;
        int WINDOW_X_POSITION;
        int WINDOW_Y_POSITION;
        float *perspectiveMatrix;
        float *rotationMatrix;
        float *translationVector;

    public:
        GLRFramework();
        virtual ~GLRFramework();

    protected:
        void startFramework(int argc, char *argv[]);
        virtual void init();
        virtual void display();
        virtual void cleanup();

        GLuint createShader(GLenum eShaderType, std::string &strShaderFile);
        GLuint createProgram(const std::vector<GLuint> &shaderList);
        void checkError();

        void setInstance(); 
        static void displayWrapper();
        static void reshapeWrapper(int width, int height);
        static void cleanupWrapper();
};

