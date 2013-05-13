#include "GLRFramework.h"
#include <fstream>

GLRFramework *GLRFramework::instance = NULL;

GLRFramework::GLRFramework() { 
    std::cout << "GLRFramework::GLRFramework()" << std::endl;
    _title = "Generic Viewer";
    WINDOW_WIDTH = 768;
    WINDOW_HEIGHT = 768;
    WINDOW_X_POSITION = 100;
    WINDOW_Y_POSITION = 100;
    // set the perspective matrix to be identity by default
    _perspective_matrix = new float[16];
    _rotation_matrix = new float[16];
    _translation_vector = new float[4];

    memset(_perspective_matrix,0, sizeof(float) * 16);

    memset(_translation_vector,0, sizeof(float) * 4);
    _perspective_matrix[0] = 1.0;
    _perspective_matrix[5] = 1.0;
    _perspective_matrix[10] = 1.0;
    _perspective_matrix[15] = 1.0;

    memset(_rotation_matrix, 0, sizeof(float) * 16);
    _rotation_matrix[0] = 1.0;
    _rotation_matrix[5] = 1.0;
    _rotation_matrix[10] = 1.0;
    _rotation_matrix[15] = 1.0;
}

GLRFramework::~GLRFramework() {
    std::cout << "Calling the GLRFramework destructor" << std::endl;
    delete[] _perspective_matrix;
    delete[] _translation_vector;
    delete[] _rotation_matrix;
}

void GLRFramework::startFramework(int argc, char *argv[]) 
{
    setInstance();	// Sets the instance to self, used in the callback wrapper functions
    std::cout << "Calling startFramework in GLRFramework" << std::endl;
    // Fire up GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    //glutInitContextVersion(4, 0);
    //glutInitContextProfile(GLUT_CORE_PROFILE);
    glutInitWindowPosition(WINDOW_X_POSITION, WINDOW_Y_POSITION);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow(_title.c_str()); 

    // Fire up GLEW
    GLenum status = glewInit();
    if (status != GLEW_OK)
    {
        fprintf(stderr, "GLEW Failed to start! Error: %s\n", glewGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    fprintf(stdout, "Using GLEW %s\n", glewGetString(GLEW_VERSION));

    // Set up function callbacks with wrapper functions
    glutCloseFunc(cleanupWrapper);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
    fprintf(stdout,"OpenGL Version: %s\n",glGetString(GL_VERSION));

    // Call subclasses init
    init();

    // Start the main GLUT thread
    glutMainLoop();
}

void GLRFramework::init() {
    std::cout << "GLRFramework::init()" << std::endl;
}

void GLRFramework::setInstance() {
    std::cout << "GLRFramework::setInstance()" << std::endl;
    instance = this;
}

void GLRFramework::display() {
    std::cout << "GLRFramework::display()" << std::endl;
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

void GLRFramework::cleanupWrapper() {
    instance->cleanup();
}

GLuint GLRFramework::createProgram(const std::vector<GLuint> &shaderList)
{
    GLuint program = glCreateProgram();

    for(size_t i = 0; i < shaderList.size(); i++)
        glAttachShader(program, shaderList[i]);

    glLinkProgram(program);

    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        delete[] strInfoLog;
    }
    for(size_t i = 0; i < shaderList.size(); i++)
        glDetachShader(program, shaderList[i]);

    return program;
}

GLuint GLRFramework::createShader(GLenum eShaderType,  std::string &strShaderFilename)
{
    GLuint shader = glCreateShader(eShaderType);

    std::ifstream shaderFile(strShaderFilename.c_str(), std::ifstream::in);

    GLchar strFileData[10000];
    unsigned int i  = 0;
    while(shaderFile.good())
    {
        strFileData[i] = shaderFile.get();
        std::cout << strFileData[i];
        i++;
    }
    strFileData[i-1] = '\0';
    const GLchar* constStrFileData = (const char*)strFileData;
    std::cout << constStrFileData << std::endl;

    glShaderSource(shader, 1, &constStrFileData, NULL);
    glCompileShader(shader);

    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

        GLchar *strInfoLog = new GLchar[infoLogLength + 1];
        glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

        const char *strShaderType = NULL;
        switch (eShaderType)
        {
            case GL_VERTEX_SHADER:   strShaderType = "vertex";   break;
            case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
            case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
        }

        fprintf(stderr, "Compile failure in %s shader: \n%s\n", 
                strShaderType, strInfoLog);
        delete[] strInfoLog;
        exit(EXIT_FAILURE);
    }

    return shader;
}

void GLRFramework::checkError()
{
    GLenum err;
    err = glGetError();
    if (err != GL_NO_ERROR)
    {
        printf("Error. glError: 0x%04X", err);
        std::cout << " - " << gluErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

