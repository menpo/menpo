#pragma once
#include "GLRFramework.h"
#include <vector>
#include <stdint.h>

GLuint create_shader_from_filepath(GLenum shader_type, const char* filepath);
GLuint glr_create_program(const std::vector<GLuint> &shaderList);
void glr_check_error();
void matrix_x_vector(float* matrix, float* vector, float*result);

typedef struct {
	unsigned int unit_offset;
	GLuint framebuffer;
	GLenum specification;
	GLenum datatype;
	void* pixels;
	unsigned int texture_width;
	unsigned int texture_height;
	unsigned int n_channels;
} glr_texture;

typedef struct {
	void* vectors;
	unsigned int n_vectors;
	unsigned int n_dims;
	GLenum datatype;
} glr_vectorset;

typedef struct {
	glr_vectorset* points;
	glr_vectorset* tcoords;
	glr_vectorset* trilist;
	glr_texture* texture;
} glr_textured_mesh;

class Rasterizer : public GLRFramework
{

private:
	// stores (X,Y,Z,W) position tpsCoordinates
	double*  _h_points;
	// (X,Y,Z) procrustes alligned coords (non tps'd)
	float* _color;
	// provides an (R,G,B,A) color for each coord
	// (redundent if textureImage is used)
	double*  textureVector;
	// (s,t) texture coords
	float* _tcoords;
	uint8_t* _texture;
	// index into coord/textureVector
	GLuint* _trilist;

	// vector to the direction of light
	float* _light_vector;

	size_t  _n_points;
	size_t  _n_tris;
	size_t _texture_width;
	size_t _texture_height;
	// what Texture Unit we wish to bind the texture to
	int _texture_unit;

	// --- Handles to GL objects ---
	GLuint _vao;
	GLuint _the_program;
	GLuint perspectiveMatrixUnif;
	GLuint translationVectorUnif;
	GLuint rotationMatrixUinf;

	GLuint _points_buffer;
	GLuint _color_buffer;
	GLuint _textureVectorBuffer;
	GLuint _tcoord_buffer;

	GLuint _trilist_buffer;

	GLuint _texture_ID;
	// object storing texture traits.
	GLuint _texture_sampler;

	//fbo parameters
	GLuint _fbo;
	GLuint _fb_texture;
	GLuint _fb_color;
	int _fb_texture_unit;
	int _fb_color_unit;
	GLubyte* _fbo_pixels;
	GLfloat* _fbo_color_pixels;

	// Handles to uniforms
	GLuint _texture_uniform;

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
	void init_vertex_buffer();
	void display();
	void init_program();
	void init_texture();
	void init_frame_buffer();
	void cleanup();
	void destroy_program();
	void destroy_VBO();
	void grab_framebuffer_data();
	void reshape(int width, int height);
	void mouseMove(int x, int y);
	void mouseButtonPress(int button, int state, int x, int y);
	void keyboardDown( unsigned char key, int x, int y );

	void printUnitTests();
	void setRotationMatrixForAngleXAngleY(float angleX,float angleY);
};
