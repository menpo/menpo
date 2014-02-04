#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glrglfw.h"
#include "glrasterizer.h"

glr_glfw_context glr_build_glfw_context_offscreen(int width, int height){
	glr_glfw_context context;
	context.title = "Offscreen Viewer";
    context.window_width= width;
    context.window_height = height;
    context.offscreen = true;
    return context;
}

glr_glfw_context glr_build_glfw_context_onscreen(int width, int height){
	glr_glfw_context context;
	context.title = "Onscreen Viewer";
    context.window_width= width;
    context.window_height = height;
    context.offscreen = false;
    return context;
}

void _glr_glew_init(void) {
	// Fire up GLEW
    // Flag is required for use with Core Profiles (which we need for OS X)
    // http://www.opengl.org/wiki/OpenGL_Loading_Library#GLEW
    glewExperimental = true;
	GLenum status = glewInit();
	if (status != GLEW_OK) {
	   fprintf(stderr, "GLEW Failed to start! Error: %s\n",
			   glewGetErrorString(status));
	   exit(EXIT_FAILURE);
	}
	fprintf(stdout, "  - Using GLEW %s\n", glewGetString(GLEW_VERSION));
	if(GLEW_ARB_texture_buffer_object_rgb32)
	   fprintf(stdout, "  - Float (X,Y,Z) rendering is supported\n");
	else
	   fprintf(stdout, "  - Float (X,Y,Z) rendering not supported\n");

	fprintf(stdout,"  - OpenGL Version: %s\n",glGetString(GL_VERSION));
    // GLEW initialization sometimes sets the GL_INVALID_ENUM state even
    // though all is fine - swallow it here (and warn the user)
    // http://www.opengl.org/wiki/OpenGL_Loading_Library#GLEW
    GLenum err = glGetError();
    if (err == GL_INVALID_ENUM)
        fprintf(stdout,"swallowing GL_INVALID_ENUM error\n");
}

void glr_glfw_init(glr_glfw_context* context)
{
	printf("glr_glfw_init(...)\n");
	// Fire up glfw
    if (!glfwInit())
        exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_VISIBLE, !context->offscreen);
    // ask for at least OpenGL 3.3 (might be able to
    // relax this in future to 3.2/3.1)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // OS X will only give us such a profile if we ask for a forward
    // compatable core proflile. Not that the forward copatibility is
    // a noop as we ask for 3.3, but unfortunately OS X needs it.
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    context->window = glfwCreateWindow(
            context->window_width, context->window_height,
            context->title, NULL, NULL);
    if (!context->window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(context->window);
    printf("Have context.\n");
    _glr_glew_init();
    // trigger a viewport resize (seems to be required in 10.9)
	glViewport(0, 0, (GLsizei) context->window_width, 
                     (GLsizei) context->window_height);
    // set the global state to the sensible defaults
    glr_set_global_settings();
}


void glr_glfw_terminate(glr_glfw_context* context)
{
    // clear up our GLFW state
    glfwDestroyWindow(context->window);
    glfwTerminate();
}

