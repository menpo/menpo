#pragma once
#include "glrglfw.h"


typedef struct {
	GLint internal_format;
	GLsizei width;
	GLsizei height;
	GLenum format;
	GLenum type;
	GLvoid* data;
	// texture binding variables
	GLenum unit;
	GLuint id;
	GLuint sampler; // stores texture traits.
	GLuint uniform;
} glr_texture;


typedef struct {
	GLvoid* vectors;
    unsigned n_vectors;
	unsigned n_dims;
	unsigned size;
	GLenum datatype;
	GLuint vbo;
	GLuint attribute_pointer;
} glr_vectorset;


typedef struct {
	glr_vectorset vertices; // float vec4 (homogeneous vertex data)
	glr_vectorset f3v_data; // float vec3 - arbitray per-vertex data. Written out to fb_f3v_target
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
    // RGB texture FB target for colour values
	glr_texture fb_rgb_target;
    // 3 channel float FB target (writes out f3v_data)
    glr_texture fb_f3v_target;
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
 * VECTORSET CONSTRUCTORS
 *
 * The following methods build glr_vectorset's of 2,3,4 vecs of floats/doubles.
 *
 * Note that the attribute_pointer needs to be set before being bound to the
 * OpenGL context.
 */
glr_vectorset glr_build_double_3v(double* vectors, size_t n_vectors);
glr_vectorset glr_build_double_4v(double* vectors, size_t n_vectors);
glr_vectorset glr_build_float_2v(float* vectors, size_t n_vectors);
glr_vectorset glr_build_float_3v(float* vectors, size_t n_vectors);
glr_vectorset glr_build_float_4v(float* vectors, size_t n_vectors);
// final unsigned varient useful for triangle list
glr_vectorset glr_build_unsigned_3v(unsigned* vectors, size_t n_vectors);

/*
 * TEXTURE CONSTRUCTORS
 *
 * The following methods build glr_textures's of RGB(A) uints/floats.
 *
 * Returns a glr_texture configured for an 8-bit RGBA texture.
 * Note that the texture unit is initialised to 999, and should be changed
 * before attempting to bind to the OpenGL context.
 */
glr_texture glr_build_uint_rgba_texture(uint8_t* texture, size_t w, size_t h);
glr_texture glr_build_uint_rgb_texture(uint8_t* texture, size_t w, size_t h);
glr_texture glr_build_float_rgb_texture(float* texture, size_t w, size_t h);
glr_texture glr_build_float_rgba_texture(float* texture, size_t w, size_t h);

/*
 * Returns a glr_textured_mesh configured for a mesh with:
 * - a set of double points @ .points (attribute_pointer needs to be set after)
 * - an unsigned triangle list @ .trilist
 * - an 8-bit RGBA texture @ .texture (unit need to be set)
 * - a set of float texture coords @ .tcoords (attribute_pointer as points)
 */
glr_textured_mesh glr_build_d4_f3_rgba_uint8_mesh(double* vertices, float* f3v_data,
        size_t n_points, unsigned* trilist, size_t n_tris, float* tcoords,
		uint8_t* texture, size_t tex_width, size_t tex_height);

glr_textured_mesh glr_build_f3_f3_rgb_uint8_mesh(float* vertices, float* f3v_data,
        size_t n_points, unsigned* trilist, size_t n_tris, float* tcoords,
		uint8_t* texture, size_t tex_width, size_t tex_height);

glr_textured_mesh glr_build_f3_f3_rgb_float_mesh(float* vertices, float* f3v_data,
        size_t n_points, unsigned* trilist, size_t n_tris, float* tcoords,
		float* texture, size_t tex_width, size_t tex_height);

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


void glr_init_and_bind_array_buffer(glr_vectorset* vector);


void glr_init_and_bind_element_buffer(glr_vectorset* vector);


void glr_init_vao(glr_textured_mesh* mesh);


void glr_init_texture(glr_texture *texture);


void glr_init_framebuffer(GLuint* fbo, glr_texture* texture, GLuint attachment);


void glr_register_draw_framebuffers(GLuint fbo, size_t n_attachments,
		                            GLenum* attachments);


void glr_set_global_settings(void);


void glr_render_scene(glr_scene* scene);


void glr_render_to_framebuffer(glr_scene* scene);


void glr_get_framebuffer(glr_texture* texture);


void glr_destroy_vbos_on_trianglar_mesh(glr_textured_mesh* mesh);

/*
 * set the float matrix to
 * [1, 0, 0, 1]
 * [0, 1, 0, 0]
 * [0, 0, 1, 0]
 * [0, 0, 0, 1]
 */
void glr_math_float_matrix_eye(float *matrix);

/*
 * Set the float vector to
 * [0, 0, 0, 1]
 */
void glr_math_float_vector4_0001(float *matrix);


void glr_print_matrix(float* matrix);

// set the clear colour to a new value
// (takes four value float array)
void glr_set_clear_color(float* clear_color_4_vec);

// get the clear colour
// (fills the four value float array passed i)
void glr_get_clear_color(float* clear_color_4_vec);

