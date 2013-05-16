#pragma once
#include "GLRFramework.h"
#include <vector>
#include <stdint.h>

GLuint create_shader_from_filepath(GLenum shader_type, const char* filepath);
GLuint glr_create_program(const std::vector<GLuint> &shaderList);
void glr_check_error();
void matrix_x_vector(float* matrix, float* vector, float*result);

typedef struct {
	unsigned int unit;
	GLint internal_format;
	GLsizei width;
	GLsizei height;
	GLenum format;
	GLenum type;
	GLvoid* data;
	// texture binding variables
	GLuint texture_ID;
	GLuint texture_unit;
	GLuint sampler; // stores texture traits.
	GLuint uniform;
} glr_texture;

typedef struct {
	void* vectors;
	unsigned int n_vectors;
	unsigned int n_dims;
	unsigned int size;
	GLenum datatype;
	GLuint vbo;
	unsigned int attribute_pointer;
} glr_vectorset;

typedef struct {
	glr_vectorset h_points;
	glr_vectorset tcoords;
	glr_vectorset trilist;
	glr_texture texture;
	GLuint vao;
} glr_textured_mesh;

class Rasterizer : public GLRFramework {

private:
	glr_textured_mesh _textured_mesh;
	float* _color;

	// vector to the direction of light
	float* _light_vector;

	size_t  _n_points;

	// --- Handles to GL objects ---
	GLuint _the_program;

	GLuint perspectiveMatrixUnif;
	GLuint translationVectorUnif;
	GLuint rotationMatrixUinf;

	GLuint _color_buffer;

	glr_texture  _texture_fb;
	glr_texture  _texture_fb_color;
	//fbo parameters
	GLuint _fbo;
	GLuint _fb_texture;
	GLuint _fb_color;
	int _fb_texture_unit;
	int _fb_color_unit;
	GLubyte* _fbo_pixels;
	GLfloat* _fbo_color_pixels;

	// if true we are rendering to just return the framebuffer.
	bool RETURN_FRAMEBUFFER;

	// variables tracking last place pressed
	int lastX, lastY;
	float _last_angle_X, _last_angle_Y;
    float angleX, angleY;
public:
	Rasterizer(double* tpsCoord_in, float* coord_in, size_t numCoords_in, 
		unsigned int* coordIndex_in, size_t numTriangles_in, 
		float* texCoord_in, uint8_t* textureImage_in, 
		size_t textureWidth_in, size_t textureHeight_in, bool INTERACTIVE_MODE);
	void return_FB_pixels(int argc, char *argv[], uint8_t* pixels, float* coords, int width, int height);
	void render(int argc, char *argv[]);
	~Rasterizer();

private:
	void init();
	void init_buffers();
	void display();
	void init_program();
	void init_frame_buffer();
	void cleanup();
	void grab_framebuffer_data();
	void reshape(int width, int height);
	void mouseMove(int x, int y);
	void mouseButtonPress(int button, int state, int x, int y);
	void keyboardDown( unsigned char key, int x, int y );

	//void printUnitTests();
	void setRotationMatrixForAngleXAngleY(float angleX,float angleY);
};



