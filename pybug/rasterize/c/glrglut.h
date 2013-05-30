#pragma once

typedef struct {
    int WINDOW_WIDTH;
	int WINDOW_HEIGHT;
	int WINDOW_X_POSITION;
	int WINDOW_Y_POSITION;
	const char *title;
	unsigned int display_mode;
} glr_glut_config;

glr_glut_config glr_build_glut_config(int width, int height);

void glr_glut_init(glr_glut_config config);

void glr_glut_set_callbacks(void);

void glr_glut_display(void);

void glr_glut_reshape(int width, int height);

void glr_glut_cleanup(void);
