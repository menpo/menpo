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

void _glr_glew_init() {
	// Fire up GLEW
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
}

void glr_glfw_init(glr_glfw_context* context)
{
	printf("glr_glfw_init(...)\n");
	// Fire up glfw
    if (!glfwInit())
        exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_VISIBLE, !context->offscreen);
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
}


void glr_glfw_terminate(glr_glfw_context* context)
{
    // clear up our GLFW state
    glfwDestroyWindow(context->window);
    glfwTerminate();
}

