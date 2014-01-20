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

    cdef void glr_glfw_terminate(glr_glfw_context*context)


# we need to be able to hold onto a scene reference
cdef extern from "./c/glr.h":
    ctypedef struct glr_textured_mesh:
        pass

    ctypedef struct glr_camera:
        pass

    ctypedef struct glr_light:
        pass

    ctypedef struct glr_texture:
        pass

    ctypedef struct glr_scene:
        glr_textured_mesh mesh
        glr_camera camera
        glr_light light
        glr_glfw_context*context
        unsigned int program
        unsigned int fbo
        glr_texture fb_texture

# externally declare the C structs we need
cdef extern from "./c/glrasterizer.h":
    glr_glfw_context init_offscreen_context(int width, int height)

    cdef glr_scene init_scene(double*points, size_t n_points,
                              unsigned int*trilist, size_t n_tris,
                              float*tcoords, uint8_t*texture,
                              size_t texture_width,
                              size_t texture_height)

    cdef void return_FB_pixels(glr_scene*scene, uint8_t*pixels)


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
        self.context = init_offscreen_context(width, height)
        self.width = width
        self.height = height
        print self.scene.fbo

    cdef _setup_scene(self,
                      np.ndarray[double, ndim=2, mode="c"] points,
                      np.ndarray[unsigned, ndim=2, mode="c"] trilist,
                      np.ndarray[float, ndim=2, mode="c"] tcoords,
                      np.ndarray[uint8_t, ndim=3, mode="c"] texture):
        self.scene = init_scene(
            &points[0, 0], points.shape[0], &trilist[0, 0], trilist.shape[0],
            &tcoords[0, 0], &texture[0, 0, 0],
            texture.shape[1], texture.shape[0])
        self.scene.context = &self.context

    def render_offscreen_rgb(self,
                             np.ndarray[double, ndim=2,
                                        mode="c"] points not None,
                             np.ndarray[unsigned, ndim=2,
                                        mode="c"] trilist not None,
                             np.ndarray[float, ndim=2,
                                        mode="c"] tcoords not None,
                             np.ndarray[uint8_t, ndim=3,
                                        mode="c"] texture not None):
        cdef np.ndarray[uint8_t, ndim=3, mode='c'] pixels = \
            np.empty((self.width, self.height, 4), dtype=np.uint8)
        self._setup_scene(points, trilist, tcoords, texture)
        return_FB_pixels(&self.scene, &pixels[0, 0, 0])
        return pixels

    def __del__(self):
        glr_glfw_terminate(&self.context)
