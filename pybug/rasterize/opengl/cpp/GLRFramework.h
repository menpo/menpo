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
        virtual void reshape(int width, int height);
        virtual void cleanup();

        GLuint createShader(GLenum eShaderType, std::string &strShaderFile);
        GLuint createProgram(const std::vector<GLuint> &shaderList);
        void checkError();

        virtual void mouseButtonPress(int button, int state, int x, int y);
        virtual void mouseMove(int x, int y);
        virtual void keyboardDown( unsigned char key, int x, int y );
        virtual void keyboardUp( unsigned char key, int x, int y );
        virtual void specialKeyboardDown( int key, int x, int y );
        virtual void specialKeyboardUp( int key, int x, int y ); 

        void setInstance(); 
        static void displayWrapper();
        static void reshapeWrapper(int width, int height);
        static void cleanupWrapper();
        static void mouseButtonPressWrapper(int button, int state, int x, int y);
        static void mouseMoveWrapper(int x, int y);
        static void keyboardDownWrapper(unsigned char key, int x, int y);
        static void keyboardUpWrapper(unsigned char key, int x, int y);
        static void specialKeyboardDownWrapper(int key, int x, int y);
        static void specialKeyboardUpWrapper(int key, int x, int y);
};

