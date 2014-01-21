# distutils: language = c
# distutils: sources = ./pybug/rasterize/c/glrasterizer.c ./pybug/rasterize/c/glr.c ./pybug/rasterize/c/glrglfw.c
# distutils: libraries = m GLEW
# distutils: extra_compile_args = -std=c99

from libc.stdint cimport uint8_t
from libcpp cimport bool
cimport numpy as np
import numpy as np


# we need to be able to hold onto a context reference
cdef extern from "./c/glrglfw.h":
    ctypedef struct glr_glfw_context:
        int window_width
        int window_height
        const char*title
        bool offscreen
        void* window

    cdef glr_glfw_context glr_build_glfw_context_offscreen(int width,
                                                           int height)
    cdef void glr_glfw_init(glr_glfw_context* context)
    cdef void glr_glfw_terminate(glr_glfw_context* context)


# we need to be able to hold onto a scene reference
cdef extern from "./c/glr.h":

    ctypedef struct glr_texture:
        int internal_format
        int width
        int height
        int format
        int type
        void* data
        unsigned unit
        unsigned texture_ID
        unsigned sampler
        unsigned uniform

    ctypedef struct glr_vectorset:
        void* vectors
        unsigned n_vectors
        unsigned n_dims
        unsigned size
        int datatype
        unsigned vbo
        unsigned attribute_pointer

    ctypedef struct glr_textured_mesh:
        glr_vectorset vertices
        glr_vectorset f3v_data
        glr_vectorset tcoords
        glr_vectorset trilist
        glr_vectorset texture
        unsigned vao

    ctypedef struct glr_camera:
        float projectionMatrix [16]
        float viewMatrix [16]

    ctypedef struct glr_light:
        float position [4]

    ctypedef struct glr_scene:
        glr_textured_mesh mesh
        glr_camera camera
        glr_light light
        float modelMatrix [16]
        glr_glfw_context* context
        unsigned program
        unsigned fbo
        glr_texture fb_rgb_target
        glr_texture fb_f3v_target

    glr_textured_mesh glr_build_d4_f3_rgba_uint8_mesh(
            double* points, float* f3v_data,
            size_t n_points, unsigned* trilist,
            size_t n_tris, float* tcoords, uint8_t* texture,
            size_t texture_width, size_t texture_height)

    glr_scene glr_build_scene()


cdef extern from "./c/glrasterizer.h":
    void render_texture_shader_to_fb(glr_scene* scene)
    void init_program_to_texture_shader(glr_scene* scene)
    void init_frame_buffer(glr_scene* scene,
                           uint8_t* rgb_pixels, float* f3v_pixels)


cdef class OpenGLRasterizer:
    cdef unsigned t_width
    cdef unsigned t_height
    cdef unsigned n_points
    cdef unsigned n_tris
    cdef glr_glfw_context context
    cdef glr_scene scene
    cdef int width
    cdef int height
    # store the pixels perminantly
    cdef uint8_t[:, :, ::1] rgb_pixels
    cdef float[:, :, ::1] f3v_pixels

    def __cinit__(self, int width, int height):
        self.scene = glr_build_scene()
        self.context = glr_build_glfw_context_offscreen(width, height)
        # init our context
        glr_glfw_init(&self.context)
        self.scene.context = &self.context
        # build the program and set it
        init_program_to_texture_shader(&self.scene)
        self.width = width
        self.height = height
        # store out the FB pixels and wire up the Framebuffer
        self.rgb_pixels = np.empty((self.height, self.width, 4),
                                   dtype=np.uint8)
        self.f3v_pixels = np.empty((self.height, self.width, 3),
                                   dtype=np.float32)
        init_frame_buffer(&self.scene, &self.rgb_pixels[0, 0, 0],
                          &self.f3v_pixels[0, 0, 0])

    def render_offscreen_rgb(self,
            np.ndarray[double, ndim=2, mode="c"] points not None,
            np.ndarray[float, ndim=2, mode="c"] f3v_data not None,
            np.ndarray[unsigned, ndim=2, mode="c"] trilist not None,
            np.ndarray[float, ndim=2, mode="c"] tcoords not None,
            np.ndarray[uint8_t, ndim=3, mode="c"] texture not None):
        self.scene.mesh = glr_build_d4_f3_rgba_uint8_mesh(
            &points[0, 0], &f3v_data[0, 0], points.shape[0],
            &trilist[0, 0], trilist.shape[0], &tcoords[0, 0],
            &texture[0, 0, 0], texture.shape[1], texture.shape[0])
        render_texture_shader_to_fb(&self.scene)
        return np.array(self.rgb_pixels), np.array(self.f3v_pixels, dtype=np
        .float32)

    cpdef get_model_matrix(self):
        return _copy_float_mat4(self.scene.modelMatrix)

    cpdef get_view_matrix(self):
        return _copy_float_mat4(self.scene.camera.viewMatrix)

    cpdef get_projection_matrix(self):
        return _copy_float_mat4(self.scene.camera.projectionMatrix)

    cpdef set_model_matrix(self,
                           np.ndarray[float, ndim=2, mode="c"] m):
        return _set_float_mat4(m, self.scene.modelMatrix)

    cpdef set_view_matrix(self,
                          np.ndarray[float, ndim=2, mode="c"] m):
        return _set_float_mat4(m, self.scene.camera.viewMatrix)

    cpdef set_projection_matrix(self,
                    np.ndarray[float, ndim=2, mode="c"] m):
        return _set_float_mat4(m, self.scene.camera.projectionMatrix)

    def __del__(self):
        glr_glfw_terminate(&self.context)

cdef _copy_float_mat4(float* src):
    r"""Copy a 4x4 float OpenGL matrix from C to Numpy

    Deals with transposing the array so as to be correct in Numpy
    """
    cdef np.ndarray[float, ndim=2, mode='c'] tgt
    tgt = np.empty((4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            # note that we transpose the matrix in and out of OpenGL, see
            # http://stackoverflow.com/a/17718408
            tgt[j, i] = src[i * 4 + j]
    return tgt

cdef _set_float_mat4(np.ndarray[float, ndim=2, mode="c"] src, float* tgt):
    r"""Set a 4x4 float OpenGL matrix from numpy array.

    Deals with transposing the array so as to be correct in OpenGL
    """
    for i in range(4):
        for j in range(4):
            # note that we transpose the matrix in and out of OpenGL, see
            # http://stackoverflow.com/a/17718408
            tgt[i * 4 + j] = src[j, i]

