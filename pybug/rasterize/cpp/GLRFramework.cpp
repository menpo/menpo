#include "GLRFramework.h"
#include <fstream>

GLRFramework *GLRFramework::instance = NULL;

GLRFramework::GLRFramework() { 
	std::cout << "GLRFramework::GLRFramework()" << std::endl;
	title = "Generic Viewer";
	WINDOW_WIDTH = 768;
	WINDOW_HEIGHT = 768;
	WINDOW_X_POSITION = 100;
	WINDOW_Y_POSITION = 100;
	// set the perspective matrix to be identity by default
	perspectiveMatrix = new float[16];
	rotationMatrix = new float[16];
	translationVector = new float[4];

	memset(perspectiveMatrix,0, sizeof(float) * 16);

	memset(translationVector,0, sizeof(float) * 4);
	perspectiveMatrix[0] = 1.0;
	perspectiveMatrix[5] = 1.0;
	perspectiveMatrix[10] = 1.0;
	perspectiveMatrix[15] = 1.0;

	memset(rotationMatrix, 0, sizeof(float) * 16);
	rotationMatrix[0] = 1.0;
	rotationMatrix[5] = 1.0;
	rotationMatrix[10] = 1.0;
	rotationMatrix[15] = 1.0;
}

GLRFramework::~GLRFramework() {
	std::cout << "GLRFramework::~GLRFramework()" << std::endl;
	delete[] perspectiveMatrix;
	delete[] translationVector;
	delete[] rotationMatrix;
}

void GLRFramework::start_framework(int argc, char *argv[])  {
	std::cout << "GLRFramework::start_framework()" << std::endl;
	setInstance();	// Sets the instance to self, used in the callback wrapper functions
	// Fire up GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	//glutInitContextVersion(4, 0);
	//glutInitContextProfile(GLUT_CORE_PROFILE);
	glutInitWindowPosition(WINDOW_X_POSITION, WINDOW_Y_POSITION);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow(title.c_str()); 
	// Fire up GLEW
	GLenum status = glewInit();
	if (status != GLEW_OK) {
	  fprintf(stderr, "GLEW Failed to start! Error: %s\n", glewGetErrorString(status));
	  exit(EXIT_FAILURE);
	}
	fprintf(stdout, "  - Using GLEW %s\n", glewGetString(GLEW_VERSION));
	// Set up function callbacks with wrapper functions
	glutReshapeFunc(reshapeWrapper);
	glutMouseFunc(mouseButtonPressWrapper);
	glutMotionFunc(mouseMoveWrapper);
	glutDisplayFunc(displayWrapper);
	glutKeyboardFunc(keyboardDownWrapper);
	glutKeyboardUpFunc(keyboardUpWrapper);
	glutSpecialFunc(specialKeyboardDownWrapper);
	glutSpecialUpFunc(specialKeyboardUpWrapper);
	glutCloseFunc(cleanupWrapper);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	fprintf(stdout,"  - OpenGL Version: %s\n",glGetString(GL_VERSION));

	// Call subclasses init
	init();

	// Start the main GLUT thread
	glutMainLoop();
}

void GLRFramework::display() {
	std::cout << "GLRFramework::display()" << std::endl;
}

void GLRFramework::reshape(int width, int height) {
	glViewport(0,0,(GLsizei)width,(GLsizei)height);
}

void GLRFramework::mouseButtonPress(int button, int state, int x, int y) {
	printf("MouseButtonPress: x: %d y: %d\n", x, y);
		
}

void GLRFramework::mouseMove(int x, int y) {
	printf("MouseMove: x: %d y: %d\n", x, y);
}

void GLRFramework::keyboardDown( unsigned char key, int x, int y ) 
{
	printf( "KeyboardDown: %c = %d\n", key, (int)key );
	if (key==27) { // ESC key
        glutLeaveMainLoop ();
	}
}

void GLRFramework::keyboardUp( unsigned char key, int x, int y ) 
{
	printf( "KeyboardUp: %c \n", key );
}

void GLRFramework::specialKeyboardDown( int key, int x, int y ) 
{
	printf( "SpecialKeyboardDown: %d\n", key );
}

void GLRFramework::specialKeyboardUp( int key, int x, int y ) 
{
	printf( "SpecialKeyboardUp: %d \n", key );
}

void GLRFramework::init() {
	std::cout << "GLRFramework::init()" << std::endl;
}

void GLRFramework::setInstance() {
	std::cout << "GLRFramework::setInstance()" << std::endl;
	instance = this;
}

void GLRFramework::cleanup() {
	std::cout << "GLRFramework::cleanup()" << std::endl;
}

// ******************************************************************
// ** Static functions which are passed to Glut function callbacks **
// ******************************************************************

void GLRFramework::displayWrapper() {
	instance->display(); 
}
	
void GLRFramework::reshapeWrapper(int width, int height) {
	instance->reshape(width, height);
}

void GLRFramework::mouseButtonPressWrapper(int button, int state, int x, int y) {
	instance->mouseButtonPress(button, state, x, y);
}

void GLRFramework::mouseMoveWrapper(int x, int y) {
	instance->mouseMove(x, y);
}
									 
void GLRFramework::keyboardDownWrapper(unsigned char key, int x, int y) {
	instance->keyboardDown(key,x,y);
}

void GLRFramework::keyboardUpWrapper(unsigned char key, int x, int y) {
	instance->keyboardUp(key,x,y);
}

void GLRFramework::specialKeyboardDownWrapper(int key, int x, int y) {
	instance->specialKeyboardDown(key,x,y);
}

void GLRFramework::specialKeyboardUpWrapper(int key, int x, int y) {
	instance->specialKeyboardUp(key,x,y);
}

void GLRFramework::cleanupWrapper() {
	instance->cleanup();
}
