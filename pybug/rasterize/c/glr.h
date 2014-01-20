#pragma once
#include "glrglfw.h"


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


typedef struct {
	float projectionMatrix [16];  // how the camera projects (ortho, persp)
    float viewMatrix [16];  // how the camera is positioned in world space
} glr_camera;


typedef struct {
	float position [4];
} glr_light;


typedef struct {
	glr_textured_mesh mesh;
	glr_camera camera;
	glr_light light;
    float modelMatrix [16];  // adjust the model in the world
	glr_glfw_context* context;
	GLuint program;
	GLuint fbo;
	// texture that is to be rendered to on the FrameBuffer
	glr_texture fb_texture;
} glr_scene;



/**
 * Checks the global OpenGL state to see if an error has occurred.
 *
 * If an error is encountered, the program is exited, and a log to stderr
 * of the problem made.
 */
void glr_check_error(void);

/**
 * Builds an OpenGL shader of the type shader_type, where shader_type can be:
 * 	- GL_VERTEX_SHADER
 * 	- GL_GEOMETRY_SHADER
 * 	- GL_FRAGMENT_SHADER
 *
 * 	Returns the GLuint that points to the shader. If there is an error, the
 * 	program is exited, and the error message produced.
 */
GLuint glr_create_shader_from_string(GLenum shader_type, const GLchar *string);

/*
 * Taking a vector of shaders, links the shaders into a single program,
 * returning the programs binding. If there is an error in the link,
 * an error message is produced and the program is terminated. If this program
 * successfully returns, it is safe to delete the individual shaders themselves
 * immediately - the program is built as is kept in the OpenGL state for us.
 */
GLuint glr_create_program(GLuint *shaders, size_t n_shaders);

/*
 * Returns a glr_vectorset configured for a set of homogeneous (X, Y, Z, W)
 * points of type double.
 * Note that the attribute_pointer needs to be set before being bound to the
 * OpenGL context.
 */
glr_vectorset glr_build_h_points(double *points, size_t n_points);

/*
 * Returns a glr_vectorset configured for a set of float (S,T) texture
 * coordinates.
 * Note that the attribute_pointer needs to be set before being bound to the
 * OpenGL context.
 */
glr_vectorset glr_build_tcoords(float *tcoords, size_t n_points);

/*
 * Returns a glr_vectorset configured for a set of (v1, v2, v3) triangluation
 * indices (a triangle list).
 */
glr_vectorset glr_build_trilist(unsigned int *trilist, size_t n_tris);

/*
 * Returns a glr_texture configured for an 8-bit RGBA texture. Note that the
 * texture unit is initialised to 999, and should be changed before attempting
 * to bind to the OpenGL context.
 */
glr_texture glr_build_rgba_texture(uint8_t* texture, size_t width,
								   size_t height);

/*
 * Returns a glr_texture configured for a float RGB texture. Note that the
 * texture unit is initialised to 999, and should be changed before attempting
 * to bind to the OpenGL context.
 */
glr_texture glr_build_rgb_float_texture(float* texture, size_t width,
								   	    size_t height);

/*
 * Returns a glr_textured_mesh configured for a mesh with:
 * - a set of double points @ .points (attribute_pointer needs to be set after)
 * - an unsigned int triangle list @ .trilist
 * - an 8-bit RGBA texture @ .texture (unit need to be set)
 * - a set of float texture coords @ .tcoords (attribute_pointer as points)
 */
glr_textured_mesh glr_build_textured_mesh(double* points, size_t n_points,
		unsigned int* trilist, size_t n_tris, float* tcoords,
		uint8_t* texture, size_t texture_width, size_t texture_height);


/*
 * Return an orthographic glr_camera at the origin
 */
glr_camera glr_build_othographic_camera_at_origin(void);


/*
 * Return a new glr_camera with a projection_matrix and model matrix
 */
glr_camera glr_build_camera(float* projectionMatrix, float* modelMatrix);


/*
 * Return a scene struct with a default camera and no mesh 
 */
glr_scene glr_build_scene(void);


void glr_init_array_buffer_from_vectorset(glr_vectorset* vector);


void glr_init_element_buffer_from_vectorset(glr_vectorset* vector);


void glr_init_buffers_from_textured_mesh(glr_textured_mesh* mesh);


void glr_init_texture(glr_texture *texture);


void glr_bind_texture_to_program(glr_texture *texture, GLuint program);


void glr_init_framebuffer(GLuint* fbo, glr_texture* texture, GLuint attachment);


void glr_register_draw_framebuffers(GLuint fbo, size_t n_attachments,
		GLenum* attachments);


void glr_global_state_settings(void);


void glr_render_scene(glr_scene* scene);


void glr_get_framebuffer(unsigned int texture_unit_offset,
		             GLuint texture_framebuffer, GLenum texture_specification,
		             GLenum texture_datatype, void* texture);


void glr_destroy_program(void);


void glr_destroy_vbos_on_trianglar_mesh(glr_textured_mesh mesh);


void glr_math_float_matrix_eye(float *matrix);

/*
 * Set the float vector to
 * [0, 0, 0, 1]
 */
void glr_math_float_vector4_0001(float *matrix);


void glr_math_float_matrix_rotation_for_angles(float *matrix, float angle_x,
                                               float angle_y);


void glr_print_matrix(float* matrix);
