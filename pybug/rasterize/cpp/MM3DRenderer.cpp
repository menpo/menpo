#include "MM3DRenderer.h"
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <cmath>

/*
MM3DRenderer::MM3DRenderer(double* tpsCoord_in, float* coord_in, size_t numCoords_in, double* textureVector_in, 
						   unsigned int* coordIndex_in, size_t numTriangles_in)
{
	title = "MM3D Viewer";
	TEXTURE_IMAGE = false;
	std::cout << "MM3DRenderer::MM3DRenderer(TextureVector)" << std::endl;
	tpsCoord = tpsCoord_in;
	coord = coord_in;
	coordIndex = coordIndex_in;
	numCoord = numCoords_in;
	numTriangles = numTriangles_in;
	textureVector = textureVector_in;
	// start viewing straight on
	lastAngleX = 0.0;
	lastAngleY = 0.0;
}
*/

MM3DRenderer::MM3DRenderer(double* tpsCoord_in, float* coord_in,  size_t numCoords_in, 
		unsigned int* coordIndex_in, size_t numTriangles_in, 
		float* texCoord_in, uint8_t* textureImage_in, 
		size_t textureWidth_in, size_t textureHeight_in, bool INTERACTIVE_MODE)
{
	lightVector = new float[3];
	memset(lightVector,0,3);
	lightVector[2] = 1.0;

	title = "MM3D Viewer";
	TEXTURE_IMAGE = true;
	std::cout << "MM3DRenderer::MM3DRenderer(TextureImage)" << std::endl;
	tpsCoord = tpsCoord_in;
	coord = coord_in;
	coordIndex = coordIndex_in;
	numCoord = numCoords_in;
	numTriangles = numTriangles_in;
	texCoord = texCoord_in;
	textureImage = textureImage_in;
	textureWidth = textureWidth_in;
	textureHeight = textureHeight_in;
	// start viewing straight on
	lastAngleX = 0.0;
	lastAngleY = 0.0;
	if(INTERACTIVE_MODE)
		RETURN_FRAMEBUFFER = false;
	else
		RETURN_FRAMEBUFFER = true;
}

MM3DRenderer::~MM3DRenderer()
{
	std::cout << "MM3DRenderer::~MM3DRenderer()" << std::endl;
	delete [] lightVector;
}

void MM3DRenderer::init()
{
	std::cout << "MM3DRenderer::init()" << std::endl;
	checkError();
	glEnable (GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);
	glGenVertexArrays(1, &vao);
	checkError();
	glBindVertexArray(vao);
	checkError();
	initializeProgram();
	glUseProgram(theProgram);
	checkError();
	initializeVertexBuffer();
	checkError();
	if(TEXTURE_IMAGE)
		initializeTexture();
	checkError();
	if(RETURN_FRAMEBUFFER)
	{
		glDepthFunc(GL_LEQUAL);
		initializeFrameBuffer();
	}
	checkError();
}

void MM3DRenderer::initializeVertexBuffer()
{
	// --- SETUP TPSCOORDBUFFER (0)
	glGenBuffers(1, &tpsCoordBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, tpsCoordBuffer);
	// allocate enough memory to store tpsCoord to the GL_ARRAY_BUFFER
	// target (which due to the above line is tpsCoordBuffer) and store it
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLdouble)*numCoord*4, 
		tpsCoord, GL_STATIC_DRAW);
	// enable the coord array (will be location = 0 in shader)
	glEnableVertexAttribArray(0);
	//prescribe how the data is stored
	glVertexAttribPointer(0, 4, GL_DOUBLE, GL_FALSE, 0, 0);
	// detatch from GL_ARRAY_BUFFER (good practice)
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	if(!TEXTURE_IMAGE)
	{
		// --- SETUP TEXTUREVECTORBUFFER (1)
		glGenBuffers(1, &textureVectorBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, textureVectorBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLdouble)*numCoord*4, 
			textureVector, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 4, GL_DOUBLE, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
	else
	{
		// --- SETUP TEXCOORDBUFFER (1)
		glGenBuffers(1, &texCoordBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*numCoord*2, 
			texCoord, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// --- SETUP COORDBUFFER (2)
	glGenBuffers(1, &coordBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, coordBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*numCoord*3, 
		coord, GL_STATIC_DRAW);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*numTriangles*3, 
		coordIndex, GL_STATIC_DRAW);
}

void MM3DRenderer::initializeTexture()
{
	// choose which unit to use and activate it
	textureImageUnit = 1;
	glActiveTexture(GL_TEXTURE0 + textureImageUnit);
	// specify the data storage and actually get OpenGL to 
	// store our textureImage
	glGenTextures(1, &textureImageID);
	glBindTexture(GL_TEXTURE_2D, textureImageID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
		textureWidth, textureHeight, 0, GL_RGBA, 
		GL_UNSIGNED_BYTE, textureImage);

	// Create the description of the texture (sampler) and bind it to the 
	// correct texture unit
	glGenSamplers(1, &textureSampler);
	glSamplerParameteri(textureSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glSamplerParameteri(textureSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glSamplerParameteri(textureSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glBindSampler(textureImageUnit, textureSampler);
    // bind the texture to a uniform called "textureImage" which can be
	// accessed from shaders
	textureUniform = glGetUniformLocation(theProgram, "textureImage");
	glUniform1i(textureUniform, textureImageUnit);

	// set the active Texture to 0 - as long as this is not changed back
	// to textureImageUnit, we know our shaders will find textureImage bound to
	// GL_TEXTURE_2D when they look in textureImageUnit
	glActiveTexture(GL_TEXTURE0);
	// note now we are free to unbind GL_TEXTURE_2D
	// on unit 0 - the state of our textureUnit is safe.
	glBindTexture(GL_TEXTURE_2D, 0);
}

void MM3DRenderer::initializeFrameBuffer()
{
	checkError();

	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	fbTextureUnit = 2;
	glActiveTexture(GL_TEXTURE0 + fbTextureUnit);
	glGenTextures(1, &fbTexture);
	glBindTexture(GL_TEXTURE_2D, fbTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
		GL_TEXTURE_2D, fbTexture, 0);
	checkError();
	glBindTexture(GL_TEXTURE_2D, 0);
	glGenTextures(1, &fbCoord);
	glBindTexture(GL_TEXTURE_2D, fbCoord);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	checkError();
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 
		GL_RGB, GL_FLOAT, NULL);
	checkError();
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, 
		GL_TEXTURE_2D, fbCoord, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	checkError();
	const GLenum buffs[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	GLsizei buffsSize = 2;
	glDrawBuffers(buffsSize, buffs);
		// now, the depth buffer
	GLuint depthBuffer;
	glGenRenderbuffers(1,  &depthBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER,depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, WINDOW_WIDTH, WINDOW_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,depthBuffer);
	GLenum status;
	status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if(status != GL_FRAMEBUFFER_COMPLETE)
	{
		printf("Framebuffer error: 0x%04X\n", status);
		//std::exit(EXIT_FAILURE);
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void MM3DRenderer::display() 
{
	//std::cout << "Calling the MM3DRenderer display method" << std::endl;
	if(RETURN_FRAMEBUFFER)
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	else
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(theProgram);
	if(!RETURN_FRAMEBUFFER)
	{
		perspectiveMatrixUnif = glGetUniformLocation(theProgram, "perspectiveMatrix");
		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, perspectiveMatrix);
		rotationMatrixUinf = glGetUniformLocation(theProgram, "rotationMatrix");
		glUniformMatrix4fv(rotationMatrixUinf, 1, GL_FALSE, rotationMatrix);
		translationVectorUnif = glGetUniformLocation(theProgram, "translationVector");
		glUniform4fv(translationVectorUnif, 1, translationVector);
		GLuint lightDirectionUnif = glGetUniformLocation(theProgram, "lightDirection");
		glUniform3fv(lightDirectionUnif, 1, lightVector);
		printUnitTests();
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glActiveTexture(GL_TEXTURE0 + textureImageUnit);
	glBindTexture(GL_TEXTURE_2D, textureImageID);
	glDrawElements(GL_TRIANGLES, numTriangles*3, GL_UNSIGNED_INT, 0);
	glutSwapBuffers();
	if(RETURN_FRAMEBUFFER)
		glutLeaveMainLoop();
}

void MM3DRenderer::printUnitTests()
{
	float* input = new float[4];
	memset(input,0.,4);
	input[0] = 0;
	input[1] = 0;
	input[2] = 1;
	input[3] = 1;

	float * result = new float[4];
	float * tempResult = new float[4];	
	matrixTimesVector(rotationMatrix,input,tempResult);
	for(int i = 0; i < 4; i++)
		tempResult[i] += translationVector[i];
	matrixTimesVector(perspectiveMatrix,tempResult,result);
	for(int i = 0; i < 4; i ++)
		printf("%2.2f\t%2.2f\t%2.2f\n",input[i],tempResult[i]-translationVector[i],result[i]);
	std::cout << std::endl;
	delete [] input;
	delete [] tempResult;
	delete [] result;
}

void MM3DRenderer::matrixTimesVector(float* matrix, float* vector, float*result)
{
	result[0] = 0;
	result[1] = 0;
	result[2] = 0;
	result[3] = 0;
	for(int i = 0; i < 4; i++){
		for(int j = 0; j < 4; j++){
			result[i] += matrix[4*i+ j]*vector[j];
		}
	}
}

void MM3DRenderer::initializeProgram()
{
	std::cout << "initializeProgram()...";
	std::vector<GLuint> shaderList;
	std::string strVertexShader;
	std::string strFragmentShader;
	if(!RETURN_FRAMEBUFFER){
		strVertexShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/interactive.vert";
		strFragmentShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/interactive.frag";
	}	
	else if(TEXTURE_IMAGE){	
		strVertexShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/textureImage.vert";
		strFragmentShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/textureImage.frag";
	}
	else
	{
		strVertexShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/shader.vert";
		strFragmentShader = "/home/jab08/.virtualenvs/pybug/src/pybug/pybug/rasterize/cpp/shader.frag";
	}
	shaderList.push_back(createShader(GL_VERTEX_SHADER,   strVertexShader  ));
	shaderList.push_back(createShader(GL_FRAGMENT_SHADER, strFragmentShader));

	theProgram = createProgram(shaderList);

	std::for_each(shaderList.begin(), shaderList.end(), glDeleteShader);
	std::cout << "done." << std::endl;
}

void MM3DRenderer::cleanup()
{
	std::cout << "MM3DRenderer::cleanup()" << std::endl;
	if(RETURN_FRAMEBUFFER)
		grabFrameBufferData();
	destroyShaders();
	destroyVBO();
}

void MM3DRenderer::grabFrameBufferData()
{
	if(RETURN_FRAMEBUFFER)
	{
		glActiveTexture(GL_TEXTURE0 + fbTextureUnit);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, fboPixels);
		glActiveTexture(GL_TEXTURE0 + fbCoordUnit);
		glBindTexture(GL_TEXTURE_2D, fbCoord);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, fboCoords);
	}
	else
		std::cout << "Trying to return FBO on an interactive session!" << std::endl;
}

void MM3DRenderer::destroyShaders()
{
	glUseProgram(0);
}

void MM3DRenderer::destroyVBO()
{
	GLenum errorCheckValue = glGetError();

	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glDeleteBuffers(1, &textureVectorBuffer);
	glDeleteBuffers(1, &coordBuffer);
	glDeleteBuffers(1, &tpsCoordBuffer);
	glDeleteBuffers(1, &indexBuffer);
	

	glBindVertexArray(0);
	glDeleteVertexArrays(1, &vao);
	errorCheckValue = glGetError();
    if (errorCheckValue != GL_NO_ERROR){
        fprintf(stderr,
            "ERROR: Could not destroy the VBO: %s \n",
            gluErrorString(errorCheckValue)
        );
        exit(-1);
    }
}

void MM3DRenderer::render(int argc, char *argv[])
{
	if(!RETURN_FRAMEBUFFER)
	{
		float fFrustumScale = 1.0f; float fzNear = 0.5f; float fzFar = 10.0f;
		memset(perspectiveMatrix,0, sizeof(float) * 16);
		perspectiveMatrix[0] = fFrustumScale;
		perspectiveMatrix[5] = fFrustumScale;
		perspectiveMatrix[10] = (fzFar + fzNear) / (fzNear - fzFar);
		perspectiveMatrix[14] = (2 * fzFar * fzNear) / (fzNear - fzFar);
		perspectiveMatrix[11] = -1.0;

		memset(translationVector,0, sizeof(float) * 4);
		translationVector[2] = -2.0;

		startFramework(argc, argv);
	}
	else
		std::cout << "Trying to render a RETURN_FRAMEBUFFER object!" << std::endl;
}

void MM3DRenderer::returnFBPixels(int argc, char *argv[], uint8_t *fboPixels_in, float *fboCoords_in, int width, int height)
{
	fboPixels = fboPixels_in;
	fboCoords = fboCoords_in;
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
	RETURN_FRAMEBUFFER = true;

	// set the rotation, perspective, and translation objects to unitary (we just want orthogonal projection)
	memset(translationVector,0, sizeof(float) * 4);
	memset(perspectiveMatrix,0, sizeof(float) * 16);
	perspectiveMatrix[0]  = 1.0;
	perspectiveMatrix[5]  = 1.0;
	perspectiveMatrix[10] = 1.0;
	perspectiveMatrix[15] = 1.0;
	memset(rotationMatrix,0, sizeof(float) * 16);
	rotationMatrix[0]  = 1.0;
	rotationMatrix[5]  = 1.0;
	rotationMatrix[10] = 1.0;
	rotationMatrix[15] = 1.0;
	startFramework(argc, argv);
}

void MM3DRenderer::reshape(int width, int height)
{
	// if in interactive mode -> adjust perspective matrix
	if(!RETURN_FRAMEBUFFER)
	{
		float fFrustumScale = 1.4;
		perspectiveMatrix[0] = fFrustumScale / (width / (float)height);
		perspectiveMatrix[5] = fFrustumScale;
    
		glUseProgram(theProgram);
		glUniformMatrix4fv(perspectiveMatrixUnif, 1, GL_FALSE, perspectiveMatrix);
		glUseProgram(0);
	}
    
    glViewport(0, 0, (GLsizei) width, (GLsizei) height);

}

void MM3DRenderer::mouseMove(int x, int y)
{
	// if in interactive mode
	if(!RETURN_FRAMEBUFFER)
	{
		int width = glutGet(GLUT_WINDOW_WIDTH);
		int height = glutGet(GLUT_WINDOW_HEIGHT);
		float pi = atan2f(0.0,-1.0);
		//std::cout << "width: " << width << "\theight : " << height << std::endl;
		int deltaX = lastX - x;
		int deltaY = lastY - y;
		//std::cout << "dX: " << deltaX << "\tdY: " << deltaY << std::endl;

		angleX = lastAngleX + (1.0*deltaY)*pi/height;
		angleY = lastAngleY + (1.0*deltaX)*pi/width;
	
		if(angleX < -pi/2)
			angleX = -pi/2;
		if(angleX > pi/2)
			angleX = pi/2;
		if(angleY < -pi/2)
			angleY = -pi/2;
		if(angleX > pi/2)
			angleX = pi/2;
		setRotationMatrixForAngleXAngleY(angleX,angleY);
		glutPostRedisplay();
	}
}

void MM3DRenderer::setRotationMatrixForAngleXAngleY(float angleX,float angleY)
{
	rotationMatrix[5]  =  cos(angleX);
	rotationMatrix[6]  = -sin(angleX);
	rotationMatrix[9]  =  sin(angleX);
	rotationMatrix[10] =  cos(angleX);

	rotationMatrix[0]  =  cos(angleY);
	rotationMatrix[2]  =  sin(angleY);
	rotationMatrix[8] = -sin(angleY);
	rotationMatrix[10] =  cos(angleY);
}

void MM3DRenderer::mouseButtonPress(int button, int state, int x, int y)
{
	
	if(state)
	{
		std::cout << "Released"  << std::endl;
		// button let go - remember current angle
		lastAngleX = angleX;
		lastAngleY = angleY;
	}
	else
	{
		std::cout << "Pressed" << std::endl;
		// button pressed - remember starting position
		lastX = x;
		lastY = y;
	}
}

void MM3DRenderer::keyboardDown( unsigned char key, int x, int y )
{
	float pi = atan2f(0.0,-1.0);
	if(key == 32) //space bar
	{
		// reset the rotation to centre
		memset(rotationMatrix, 0, sizeof(float) * 16);
		rotationMatrix[0] = 1.0;
		rotationMatrix[5] = 1.0;
		rotationMatrix[10] = 1.0;
		rotationMatrix[15] = 1.0;
		glutPostRedisplay();
	}
	else if (key==27)// ESC key
        glutLeaveMainLoop ();
	else if (key == 'p')
	{		
		setRotationMatrixForAngleXAngleY(-0.10,pi/9.0);
		glutPostRedisplay();

	}
	else if (key == 's')
	{
		setRotationMatrixForAngleXAngleY(0,pi/2.);
		glutPostRedisplay();
	}
	else
		std::cout << "Keydown: " << key << std::endl;
}

