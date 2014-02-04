#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glr.h"

void glr_check_error(void) {
	GLenum err;
	err = glGetError();
	if (err != GL_NO_ERROR) {
		printf("Error. glError: 0x%04X", err);
		printf(" - %s\n", gluErrorString(err));
		exit(EXIT_FAILURE);
	}
}

GLuint glr_create_shader_from_string(GLenum shader_type,
								     const GLchar* string) {
	GLuint shader = glCreateShader(shader_type);
	glShaderSource(shader, 1, &string, NULL);
	glCompileShader(shader);
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		GLint info_log_length;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
		GLchar str_info_log [info_log_length + 1];
		glGetShaderInfoLog(shader, info_log_length, NULL, str_info_log);
		const char *strShaderType = NULL;
		switch (shader_type) {
			case GL_VERTEX_SHADER:   strShaderType = "vertex";   break;
			case GL_GEOMETRY_SHADER: strShaderType = "geometry"; break;
			case GL_FRAGMENT_SHADER: strShaderType = "fragment"; break;
		}
		fprintf(stderr, "Compile failure in %s shader: \n%s\n",
				strShaderType, str_info_log);
		exit(EXIT_FAILURE);
	}
	return shader;
}

GLuint glr_create_program(GLuint *shaders, size_t n_shaders) {
	GLuint program = glCreateProgram();
	for(size_t i = 0; i < n_shaders; i++)
		glAttachShader(program, shaders[i]);
	glLinkProgram(program);
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		GLint info_log_length;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
		GLchar str_info_log [info_log_length + 1];
		glGetProgramInfoLog(program, info_log_length, NULL, str_info_log);
		fprintf(stderr, "Linker failure: %s\n", str_info_log);
	}
	for(size_t i = 0; i < n_shaders; i++)
		glDetachShader(program, shaders[i]);
	return program;
}

glr_texture glr_build_uint_rgb_texture(uint8_t* texture, size_t w, size_t h)
{
	glr_texture texture_tmp;
	texture_tmp.unit = 999; // the texture unit this texture binds to. Set to
	// 999 as a safety - must be changed!
	texture_tmp.internal_format = GL_RGB8;
	texture_tmp.width = w;
	texture_tmp.height = h;
	texture_tmp.format = GL_RGB;
	texture_tmp.type = GL_UNSIGNED_BYTE;
	texture_tmp.data = texture;
	return texture_tmp;
}

glr_texture glr_build_uint_rgba_texture(uint8_t* texture, size_t w, size_t h)
{
	glr_texture texture_tmp;
	texture_tmp.unit = 999; // the texture unit this texture binds to. Set to
	// 999 as a safety - must be changed!
	texture_tmp.internal_format = GL_RGBA8;
	texture_tmp.width = w;
	texture_tmp.height = h;
	texture_tmp.format = GL_RGBA;
	texture_tmp.type = GL_UNSIGNED_BYTE;
	texture_tmp.data = texture;
	return texture_tmp;
}


glr_texture glr_build_float_rgb_texture(float* texture, size_t w, size_t h)
{
	glr_texture texture_tmp;
	texture_tmp.unit = 999; // the texture unit this texture binds to. Set to
	// 999 as a safety - must be changed!
	texture_tmp.internal_format = GL_RGB32F;
	texture_tmp.width = w;
	texture_tmp.height = h;
	texture_tmp.format = GL_RGB;
	texture_tmp.type = GL_FLOAT;
	texture_tmp.data = texture;
	return texture_tmp;
}

glr_texture glr_build_float_rgba_texture(float* texture, size_t w, size_t h)
{
	glr_texture texture_tmp;
	texture_tmp.unit = 999; // the texture unit this texture binds to. Set to
	// 999 as a safety - must be changed!
	texture_tmp.internal_format = GL_RGBA32F;
	texture_tmp.width = w;
	texture_tmp.height = h;
	texture_tmp.format = GL_RGBA;
	texture_tmp.type = GL_FLOAT;
	texture_tmp.data = texture;
	return texture_tmp;
}

glr_vectorset glr_build_double_3v(double* vectors, size_t n_vectors) {
	glr_vectorset vector_tmp;
	vector_tmp.datatype = GL_DOUBLE;
	vector_tmp.n_dims = 3;
	vector_tmp.n_vectors = n_vectors;
	vector_tmp.size = sizeof(GLdouble);
	vector_tmp.vectors = vectors;
	return vector_tmp;
}

glr_vectorset glr_build_double_4v(double* vectors, size_t n_vectors) {
	glr_vectorset vector_tmp;
	vector_tmp.datatype = GL_DOUBLE;
	vector_tmp.n_dims = 4;
	vector_tmp.n_vectors = n_vectors;
	vector_tmp.size = sizeof(GLdouble);
	vector_tmp.vectors = vectors;
	return vector_tmp;
}

glr_vectorset glr_build_float_2v(float* vectors, size_t n_vectors) {
	glr_vectorset vector_tmp;
	vector_tmp.datatype = GL_FLOAT;
	vector_tmp.n_dims = 2;
	vector_tmp.n_vectors = n_vectors;
	vector_tmp.size = sizeof(GLfloat);
	vector_tmp.vectors = vectors;
	return vector_tmp;
}

glr_vectorset glr_build_float_3v(float* vectors, size_t n_vectors) {
	glr_vectorset vector_tmp;
	vector_tmp.datatype = GL_FLOAT;
	vector_tmp.n_dims = 3;
	vector_tmp.n_vectors = n_vectors;
	vector_tmp.size = sizeof(GLfloat);
	vector_tmp.vectors = vectors;
	return vector_tmp;
}

glr_vectorset glr_build_float_4v(float* vectors, size_t n_vectors) {
	glr_vectorset vector_tmp;
	vector_tmp.datatype = GL_FLOAT;
	vector_tmp.n_dims = 4;
	vector_tmp.n_vectors = n_vectors;
	vector_tmp.size = sizeof(GLfloat);
	vector_tmp.vectors = vectors;
	return vector_tmp;
}

glr_vectorset glr_build_unsigned_3v(unsigned* vectors, size_t n_vectors) {
	glr_vectorset vector_tmp;
	vector_tmp.datatype = GL_UNSIGNED_INT;
	vector_tmp.n_dims = 3;
	vector_tmp.n_vectors = n_vectors;
	vector_tmp.size = sizeof(GLuint);
	vector_tmp.vectors = vectors;
	return vector_tmp;
}

glr_textured_mesh glr_build_d4_f3_rgba_uint8_mesh(double* vertices, float* f3v_data,
        size_t n_points, unsigned* trilist, size_t n_tris, float* tcoords,
		uint8_t* texture, size_t tex_width, size_t tex_height) {
	glr_textured_mesh mesh;
	mesh.vertices = glr_build_double_4v(vertices, n_points);
	mesh.f3v_data = glr_build_float_3v(f3v_data, n_points);
	mesh.tcoords = glr_build_float_2v(tcoords, n_points);
	mesh.trilist = glr_build_unsigned_3v(trilist, n_tris);
	mesh.texture = glr_build_uint_rgba_texture(texture, tex_width, tex_height);
	return mesh;
}

glr_textured_mesh glr_build_f3_f3_rgb_uint8_mesh(float* vertices, float* f3v_data,
        size_t n_points, unsigned* trilist, size_t n_tris, float* tcoords,
		uint8_t* texture, size_t tex_width, size_t tex_height) {
	glr_textured_mesh mesh;
	mesh.vertices = glr_build_float_3v(vertices, n_points);
	mesh.f3v_data = glr_build_float_3v(f3v_data, n_points);
	mesh.tcoords = glr_build_float_2v(tcoords, n_points);
	mesh.trilist = glr_build_unsigned_3v(trilist, n_tris);
	mesh.texture = glr_build_uint_rgb_texture(texture, tex_width, tex_height);
	return mesh;
}

glr_textured_mesh glr_build_f3_f3_rgb_float_mesh(float* vertices, float* f3v_data,
        size_t n_points, unsigned* trilist, size_t n_tris, float* tcoords,
		float* texture, size_t tex_width, size_t tex_height) {
	glr_textured_mesh mesh;
	mesh.vertices = glr_build_float_3v(vertices, n_points);
	mesh.f3v_data = glr_build_float_3v(f3v_data, n_points);
	mesh.tcoords = glr_build_float_2v(tcoords, n_points);
	mesh.trilist = glr_build_unsigned_3v(trilist, n_tris);
	mesh.texture = glr_build_float_rgb_texture(texture, tex_width, tex_height);
	return mesh;
}

glr_camera glr_build_othographic_camera_at_origin(void)
{
	glr_camera camera;
	// set the camera's matrices to I
	glr_math_float_matrix_eye(camera.projectionMatrix);
    glr_math_float_matrix_eye(camera.viewMatrix);
	return camera;
}

glr_camera glr_build_camera(float* projectionMatrix, float* viewMatrix) {
    glr_camera camera;
    memcpy(camera.projectionMatrix, projectionMatrix,
           sizeof(camera.projectionMatrix));
    // copy the modelMatrix
    memcpy(camera.viewMatrix, viewMatrix,
           sizeof(camera.viewMatrix));
    return camera;
}

glr_scene glr_build_scene(void)
{
	glr_scene scene;
	glr_math_float_matrix_eye(scene.modelMatrix);
    glr_math_float_vector4_0001(scene.light.position);
    scene.camera = glr_build_othographic_camera_at_origin();
    return scene;
}

void glr_init_and_bind_array_buffer(glr_vectorset *vector) {
	glGenBuffers(1, &(vector->vbo));
	glBindBuffer(GL_ARRAY_BUFFER, vector->vbo);
	glBufferData(GL_ARRAY_BUFFER,
				 (vector->size) * (vector->n_vectors) * (vector->n_dims),
				 vector->vectors, GL_STATIC_DRAW);
	glEnableVertexAttribArray(vector->attribute_pointer);
	glVertexAttribPointer(vector->attribute_pointer, vector->n_dims,
						  vector->datatype, GL_FALSE, 0, 0);
}

void glr_init_and_bind_element_buffer(glr_vectorset *vector) {
	glGenBuffers(1, &(vector->vbo));
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vector->vbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
			(vector->size) * (vector->n_vectors) * (vector->n_dims),
			vector->vectors, GL_STATIC_DRAW);
}

// TODO make the texture sampler a seperate customizable thing.
void glr_init_texture(glr_texture *texture) {
	printf("glr_init_texture(...)\n");
    // OpenGL texturing works as follows.
    //
    // a. Many textures can be stored in memory, I just need to use glGenTextures
    // to get some handles that I am allowed to store textures in. Call one of these
    // handles a texture *id*.
    //
    // b. To fill one of these out, I BIND the texture id I got from glGenTextures
    // to a system texture type (like GL_TEXTURE_2D). Then I'm free to actually
    // store some pixels and metadata by the glTexImage call.
    //
    // c. At this point I could unbind the textureID from GL_TEXTURE_2D. If in the
    // future I want to use this texture in shaders, or change it's state, I would
    // just have to rebind it - then subsequent calls using GL_TEXTURE_2D would
    // change this texture.
    //
    // d. We also have to worry about metadata (how the texture should be sampled).
    // This is dictated by calls to glSampler*. If I wanted different sampling
    // behavior around different texture sets, I'd have to keep flicking all this
    // state on and off around the correct rendering calls.
    //
    // d. If I was writing a game, this could be a challanging task. I may have
    // many different types of textures on the go, and I'd have to manage all this
    // state. To make things a little easier, TEXTURE UNITS were introduced. A
    // TEXTURE UNIT just holds a set of currently bound textures - so, on a unit,
    // you can leave your texture id bound to GL_TEXTURE_2D for instance. All
    // sampling calls are also bound to a unit - so making a unit active sets
    // up all the sampler state as it last was when the unit was actice.
    //
    // Now the usage pattern is something like:
    //
    //   - use glActiveTexture to set my texture->unit as the active one. Set an
    //   GL_TEXTURE_2D texture, my mipmap and normal textures...everything. Also
    //   set all my sampler state for these family of textures.
    //
    //   - unblind the texture unit, and know all my binds won't be disturbed.
    //     Do whatever else we need to with textures (binding to GL_TEXTURE_2D,
    //     changing sampler state - not of it will affect the texture unit you
    //     have).
    //
    //   - before rendering use glActiveTexture to set my texture->unit as active.
    //     I previously panstakingly setup all my textures just so on this unit
    //     so I'm good to go.
    //
    // In order then, the first thing to do is choose our texture unit
    //
    // 1. Set the unit to texture->unit
	glActiveTexture(GL_TEXTURE0 + texture->unit);
    // 2. Get a handle on a piece of OpenGL memory where we can store our
    // texture
	glGenTextures(1, &(texture->id));
    // 3. Set the currently active GL_TEXTURE_2D to the texture->id
	glBindTexture(GL_TEXTURE_2D, texture->id);
    // 4. fill the currently active GL_TEXTURE_2D (texture->id thanks to 3.)
    // with our actual pixels
	glTexImage2D(GL_TEXTURE_2D, 0, texture->internal_format,
		texture->width, texture->height, 0, texture->format,
		texture->type, texture->data);
	// Create the description of the texture (sampler)
	glGenSamplers(1, &(texture->sampler));
	glSamplerParameteri(texture->sampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glSamplerParameteri(texture->sampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glSamplerParameteri(texture->sampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    // Bind this metadata to the unit
	glBindSampler(texture->unit, texture->sampler);
    // UNBIND THE TEXTURE UNIT. Now all our texture information is safe! Just
    // bind the right unit before rendering and we are good to go.
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void glr_init_vao(glr_textured_mesh *mesh) {
	printf("glr_init_vao(...)\n");
    // for simplicity, all our VBO/attribute bindings are wrapped in a
    // Vertex Array object.
    // 1. Generate and bind a VAO.
	glGenVertexArrays(1, &(mesh->vao));
	glBindVertexArray(mesh->vao);
    // 2. Make all our intialization code run. The VAO will track buffer
    // attribute bindings for us.
	glr_init_and_bind_array_buffer(&mesh->vertices);
	glr_init_and_bind_array_buffer(&mesh->f3v_data);
	glr_init_and_bind_array_buffer(&mesh->tcoords);
	glr_init_and_bind_element_buffer(&mesh->trilist);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->trilist.vbo);
    // 3. Unbind the VAO.
	glBindVertexArray(0);
    // now before rendering we only need to glBindVertexArray(mesh->vao)
    // - all the above attributes will be set for us.
}

void glr_init_framebuffer(GLuint* fbo, glr_texture* texture, GLuint attachment)
{
	glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D,
			texture->id, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glr_check_error();
}

void glr_register_draw_framebuffers(GLuint fbo, size_t n_attachments,
		GLenum* attachments)
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glDrawBuffers(n_attachments, attachments);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glr_check_error();
}

void glr_set_global_settings(void) {
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CW);  // as we do a flip in the fragment shader!
	glDepthFunc(GL_LEQUAL);
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
}

void glr_set_clear_color(float* cv) {
    glClearColor(cv[0], cv[1], cv[2], cv[3]);
}

void glr_get_clear_color(float* cv) {
    glGetFloatv(GL_COLOR_CLEAR_VALUE, cv);
}

void glr_render_scene(glr_scene* scene) {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(scene->program);

    // 1. SETUP UNIFORMS
    GLuint uniform;
	// a. camera uniforms
    uniform = glGetUniformLocation(scene->program, "viewMatrix");
    glUniformMatrix4fv(uniform, 1, GL_FALSE, scene->camera.viewMatrix);

	uniform = glGetUniformLocation(scene->program, "projectionMatrix");
	glUniformMatrix4fv(uniform, 1, GL_FALSE, scene->camera.projectionMatrix);

	// b. modelMatrix
    uniform = glGetUniformLocation(scene->program, "modelMatrix");
    glUniformMatrix4fv(uniform, 1, GL_FALSE, scene->modelMatrix);

	// c. lightPosition
    uniform = glGetUniformLocation(scene->program, "lightPosition");
    glUniform4fv(uniform, 1, scene->light.position);

    // d. textureImage
	uniform = glGetUniformLocation(scene->program, "textureImage");
	glUniform1i(uniform, scene->mesh.texture.unit);

	// 2. BIND VAO AND DRAW
    // this call to bind vertex array means trilist, points,
    // and tcoords are all bound to the attributes and ready to go
	glBindVertexArray(scene->mesh.vao);
	glDrawElements(GL_TRIANGLES, scene->mesh.trilist.n_vectors * 3,
			GL_UNSIGNED_INT, 0);
    // 3. DETATCH + SWAP BUFFERS
    // now we're done, can disable the vertex array (for safety)
	glBindVertexArray(0);
	glfwSwapBuffers(scene->context->window);
}

void glr_render_to_framebuffer(glr_scene* scene)
{
	// bind the framebuffer and render as normal
	glBindFramebuffer(GL_FRAMEBUFFER, scene->fbo);
	glr_render_scene(scene);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // grab the framebuffer content
	glr_get_framebuffer(&scene->fb_rgb_target);
	glr_get_framebuffer(&scene->fb_f3v_target);
}

void glr_get_framebuffer(glr_texture* texture)
{
	glActiveTexture(GL_TEXTURE0 + texture->unit);
    glBindTexture(GL_TEXTURE_2D, texture->id);
	glGetTexImage(GL_TEXTURE_2D, 0, texture->format, texture->type,
                  texture->data);
	glActiveTexture(GL_TEXTURE0);
}

void glr_destroy_vbos_on_trianglar_mesh(glr_textured_mesh* mesh) {
    // ensure the VAO is unbound.
	glBindVertexArray(0);
    // delete our buffers
	glDeleteBuffers(1, &(mesh->vertices.vbo));
	glDeleteBuffers(1, &(mesh->f3v_data.vbo));
	glDeleteBuffers(1, &(mesh->trilist.vbo));
	glDeleteBuffers(1, &(mesh->tcoords.vbo));
	// now the buffers are all cleared, we can unbind and delete the vao
	glDeleteVertexArrays(1, &(mesh->vao));
}

void glr_math_float_matrix_eye(float* matrix) {
    memset(matrix, 0, sizeof(float) * 16);
    matrix[0] = 1.0;
    matrix[5] = 1.0;
    matrix[10] = 1.0;
    matrix[15] = 1.0;
}

void glr_math_float_vector4_0001(float* vector) {
    memset(vector, 0, sizeof(float) * 3);
    vector[3] = 1.0;
}

void glr_print_matrix(float* matrix) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f\t", matrix[i *4 + j]);
        }
        printf("\n");
    }
}

