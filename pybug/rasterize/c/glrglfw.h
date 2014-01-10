#pragma once

#include <stdbool.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

typedef struct {
    int window_width;
	int window_height;
	const char *title;
	bool offscreen;
    GLFWwindow* window;
} glr_glfw_context;

glr_glfw_context glr_build_glfw_context_offscreen(int width, int height);
glr_glfw_context glr_build_glfw_context_onscreen(int width, int height);

void glr_glfw_init(glr_glfw_context* context);

void glr_glfw_terminate(glr_glfw_context* context);

