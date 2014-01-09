#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

typedef struct {
    int WINDOW_WIDTH;
	int WINDOW_HEIGHT;
	int WINDOW_X_POSITION;
	int WINDOW_Y_POSITION;
	const char *title;
	unsigned int display_mode;
    GLFWwindow* window;
} glr_glfw_config;

glr_glfw_config glr_build_glfw_config(int width, int height);

void glr_glfw_init(glr_glfw_config config);

void glr_glfw_set_callbacks(void);

void glr_glfw_display(void);

void glr_glfw_reshape(int width, int height);

void glr_glfw_cleanup(void);
