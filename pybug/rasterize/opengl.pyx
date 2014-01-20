# distutils: language = c
# distutils: sources = ./pybug/rasterize/c/glrasterizer.c ./pybug/rasterize/c/glr.c ./pybug/rasterize/c/glrglfw.c
# distutils: libraries = m GLU GL glfw GLEW
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
        void*window

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
        unsigned texture_ID
        unsigned texture_unit
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
        glr_vectorset h_points
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
        glr_texture fb_texture

    glr_textured_mesh glr_build_textured_mesh(
            double* points, size_t n_points, unsigned* trilist,
            size_t n_tris, float* tcoords, uint8_t* texture,
            size_t texture_width, size_t texture_height)

    glr_scene glr_build_scene()

# externally declare the C structs we need
cdef extern from "./c/glrasterizer.h":
    void return_FB_pixels(glr_scene*scene, uint8_t*pixels)


cdef class OpenGLRasterizer:
    cdef unsigned t_width
    cdef unsigned t_height
    cdef unsigned n_points
    cdef unsigned n_tris
    cdef glr_glfw_context context
    cdef glr_scene scene
    cdef int width
    cdef int height

    def __cinit__(self, int width, int height):
        self.scene = glr_build_scene()
        self.context = glr_build_glfw_context_offscreen(width, height)
        # init our context
        glr_glfw_init(&self.context)
        self.scene.context = &self.context
        self.width = width
        self.height = height

    def render_offscreen_rgb(self,
            np.ndarray[double, ndim=2, mode="c"] points not None,
            np.ndarray[unsigned, ndim=2, mode="c"] trilist not None,
            np.ndarray[float, ndim=2, mode="c"] tcoords not None,
            np.ndarray[uint8_t, ndim=3, mode="c"] texture not None):
        cdef np.ndarray[uint8_t, ndim=3, mode='c'] pixels = \
            np.empty((self.width, self.height, 4), dtype=np.uint8)
        self.scene.mesh = glr_build_textured_mesh(
            &points[0, 0], points.shape[0], &trilist[0, 0], trilist.shape[0],
            &tcoords[0, 0], &texture[0, 0, 0], texture.shape[1],
            texture.shape[0])
        return_FB_pixels(&self.scene, &pixels[0, 0, 0])
        return pixels

    cpdef get_model_matrix(self):
        return _copy_float_mat4(self.scene.modelMatrix)

    cpdef get_view_matrix(self):
        return _copy_float_mat4(self.scene.camera.viewMatrix)

    cpdef get_projection_matrix(self):
        return _copy_float_mat4(self.scene.camera.projectionMatrix)

    def set_model_matrix(self,
                           np.ndarray[float, ndim=2, mode="c"] m not None):
        return _set_float_mat4(m, self.scene.modelMatrix)

    def set_view_matrix(self,
                          np.ndarray[float, ndim=2, mode="c"] m not None):
        return _set_float_mat4(m, self.scene.camera.viewMatrix)

    def set_projection_matrix(self,
                    np.ndarray[float, ndim=2, mode="c"] m not None):
        return _set_float_mat4(m, self.scene.camera.projectionMatrix)

    def __del__(self):
        glr_glfw_terminate(&self.context)

cdef _copy_float_mat4(float* src):
    cdef np.ndarray[float, ndim=2, mode='c'] tgt = np.empty((4, 4),
                                                            dtype=np.float32)
    for i in range(4):
        for j in range(4):
            tgt[i, j] = src[i * 4 + j]
    return tgt

cdef _set_float_mat4(np.ndarray[float, ndim=2, mode="c"] src, float* tgt):
    for i in range(4):
        for j in range(4):
            tgt[i * 4 + j] = src[i, j]
