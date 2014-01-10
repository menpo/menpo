# distutils: language = c
# distutils: sources = ./pybug/rasterize/c/glrasterizer.c ./pybug/rasterize/c/glr.c ./pybug/rasterize/c/glrglfw.c
# distutils: libraries = m GLU GL glfw GLEW
from libc.stdint cimport uint8_t
import numpy as np
cimport numpy as np



# externally declare the C++ classes
cdef extern from "./c/glrasterizer.h":

    cdef void init_scene(double* points, size_t n_points,
                     unsigned int* trilist, size_t n_tris,
                     float* tcoord, uint8_t* texture, size_t texture_width,
                     size_t texture_height)
    cdef void return_FB_pixels(uint8_t* pixels, int width, int height)


cdef class OpenGLRasterizer:
    cdef unsigned t_width
    cdef unsigned t_height
    cdef unsigned n_points
    cdef unsigned n_tris

    def __cinit__(self,
                  np.ndarray[double, ndim=2, mode="c"] points not None ,
                  np.ndarray[unsigned, ndim=2, mode="c"] trilist not None,
                  np.ndarray[float, ndim=2, mode="c"] tcoords not None,
                  np.ndarray[uint8_t, ndim=3, mode="c"] texture not None):
        self.t_height = texture.shape[0]
        self.t_width = texture.shape[1]
        self.n_points = points.shape[0]
        self.n_tris = trilist.shape[0]
        init_scene(&points[0,0], self.n_points, &trilist[0,0], self.n_tris,
                   &tcoords[0,0], &texture[0,0,0], self.t_width, self.t_height)

    def pixels(self, render_width, render_height):
        cdef np.ndarray[uint8_t, ndim=3, mode='c'] pixels = \
            np.empty((render_width, render_height, 4), dtype=np.uint8)
        return_FB_pixels(&pixels[0,0,0], render_width, render_height)
        return pixels
