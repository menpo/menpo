#include <stdio.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "glrglut.h"
#include "glrasterizer.h"

glr_glut_config glr_build_glut_config(int width, int height){
	glr_glut_config config;
	config.title = "Generic Viewer";
    config.WINDOW_WIDTH = width;
    config.WINDOW_HEIGHT = height;
    config.WINDOW_X_POSITION = 0;
    config.WINDOW_Y_POSITION = 0;
    config.display_mode = GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH;
    return config;
}

void glr_glut_init(glr_glut_config config)
{
	printf("glr_glut_init(...)\n");
	// Fire up GLUT
	int argc = 1;
	char *argv = "dummy";
	glutInit(&argc, &argv);
	glutInitDisplayMode(config.display_mode);
	//glutInitContextVersion(4, 0);
	//glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitWindowPosition(config.WINDOW_X_POSITION, config.WINDOW_Y_POSITION);
	glutInitWindowSize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT);
	glutCreateWindow(config.title);
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

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	fprintf(stdout,"  - OpenGL Version: %s\n",glGetString(GL_VERSION));
	glr_glut_set_callbacks();
}

void glr_glut_set_callbacks(void)
{
	glutReshapeFunc(glr_glut_reshape);
	glutDisplayFunc(display);
	glutCloseFunc(grab_framebuffer_and_cleanup);
}

void glr_glut_reshape(int width, int height) {
	printf("glr_glut_reshape(...)\n");
//	if(!PERSPECTIVE) {
//		float fFrustumScale = 1.4;
//		_m_perspective[0] = fFrustumScale / (width / (float)height);
//		_m_perspective[5] = fFrustumScale;
//		glUseProgram(_the_program);
//		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, _m_perspective);
//		glUseProgram(0);
//	}
	glViewport(0, 0, (GLsizei) width, (GLsizei) height);
}


//void Rasterizer::render(int argc, char *argv[])
//{
//	std::cout << "Rasterizer::render()" << std::endl;
//	if(!RETURN_FRAMEBUFFER) {
//		float fFrustumScale = 1.0f; float fzNear = 0.5f; float fzFar = 10.0f;
//		memset(_m_perspective,0, sizeof(float) * 16);
//		_m_perspective[0] = fFrustumScale;
//		_m_perspective[5] = fFrustumScale;
//		_m_perspective[10] = (fzFar + fzNear) / (fzNear - fzFar);
//		_m_perspective[14] = (2 * fzFar * fzNear) / (fzNear - fzFar);
//		_m_perspective[11] = -1.0;
//		memset(_v_translation,0, sizeof(float) * 4);
//		_v_translation[2] = -2.0;
//		this->go();
//	} else {
//		std::cout << "Trying to render a RETURN_FRAMEBUFFER object!"
//				  << std::endl;
//	}
//}

void glr_glut_cleanup(void) {
	printf("glr_glut_cleanup()\n");
	grab_framebuffer_and_cleanup();
}
