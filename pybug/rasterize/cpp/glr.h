#pragma once

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


/**
 * Checks the global OpenGL state to see if an error has occurred.
 *
 * If an error is encountered, the program is exited, and a log to stderr
 * of the problem made.
 */
void glr_check_error();

/**
 * Builds an OpenGL shader of the type shader_type, where shader_type can be:
 * 	- GL_VERTEX_SHADER
 * 	- GL_GEOMETRY_SHADER
 * 	- GL_FRAGMENT_SHADER
 *
 * 	Returns the GLuint that points to the shader. If there is an error, the
 * 	program is exited, and the error message produced.
 */
GLuint glr_create_shader_from_string(GLenum shader_type, const GLchar* string);

/*
 * Taking a vector of shaders, links the shaders into a single program,
 * returning the programs binding. If there is an error in the link,
 * an error message is produced and the program is terminated. If this program
 * successfully returns, it is safe to delete the individual shaders themselves
 * immediately - the program is built as is kept in the OpenGL state for us.
 */
GLuint glr_create_program(const std::vector<GLuint> &shaders);

/*
 * Builds a glr_vectorset storing the pertinent information about a set of
 * homogeneous (X, Y, Z, W) points. Returns the vectorset ready for use.
 */
glr_vectorset glr_build_h_points(double* points, size_t n_points);

/*
 * Builds a glr_vectorset storing the pertinent information about a set of
 * (S,T) texture coordinates. Returns the vectorset ready for use.
 */
glr_vectorset glr_build_tcoords(float* tcoords, size_t n_points);

/*
 * Builds a glr_vectorset storing the pertinent information about a set of
 * (v1, v2, v3) triangluation indices (a triangle list). Returns the vectorset
 * ready for use.
 */
glr_vectorset glr_build_trilist(unsigned int* trilist, size_t n_tris);



glr_texture glr_build_rgba_texture(uint8_t* texture, size_t width,
								   size_t height);


glr_texture glr_build_rgb_float_texture(float* texture, size_t width,
								   	    size_t height);


glr_textured_mesh glr_build_textured_mesh(double* points, size_t n_points,
		unsigned int* trilist, size_t n_tris, float* tcoords,
		uint8_t* texture, size_t texture_width, size_t texture_height);


void glr_init_array_buffer_from_vectorset(glr_vectorset& vector);


void glr_init_element_buffer_from_vectorset(glr_vectorset& vector);


void glr_setup_buffers_on_textured_mesh(glr_textured_mesh& mesh);


void glr_init_texture(glr_texture& texture);


void glr_bind_texture_to_program(glr_texture& texture, GLuint program);


void glr_global_state_settings();


void glr_get_framebuffer(unsigned int texture_unit_offset,
		             GLuint texture_framebuffer, GLenum texture_specification,
		             GLenum texture_datatype, void* texture);


void glr_destroy_program();


void glr_destroy_vbos_on_trianglar_mesh(glr_textured_mesh mesh);
