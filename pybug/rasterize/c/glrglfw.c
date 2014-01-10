#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glrglfw.h"
#include "glrasterizer.h"

glr_glfw_config glr_build_glfw_config_offscreen(int width, int height){
	glr_glfw_config config;
	config.title = "Offscreen Viewer";
    config.window_width= width;
    config.window_height = height;
    config.offscreen = true;
    return config;
}

glr_glfw_config glr_build_glfw_config_onscreen(int width, int height){
	glr_glfw_config config;
	config.title = "Onscreen Viewer";
    config.window_width= width;
    config.window_height = height;
    config.offscreen = false;
    return config;
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

void glr_glfw_init(glr_glfw_config* config)
{
	printf("glr_glfw_init(...)\n");
	// Fire up glfw
    if (!glfwInit())
        exit(EXIT_FAILURE);
    glfwWindowHint(GLFW_VISIBLE, !config->offscreen);
    config->window = glfwCreateWindow(
            config->window_width, config->window_height,
            config->title, NULL, NULL);
    if (!config->window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(config->window);
    printf("Have context.\n");
    _glr_glew_init();
}

