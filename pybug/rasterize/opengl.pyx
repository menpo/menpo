# distutils: language = c++
# distutils: sources = ./pybug/rasterize/cpp/MM3DRenderer.cpp ./pybug/rasterize/cpp/GLRFramework.cpp
# distutils: libraries = GLU GL glut GLEW
from libcpp.vector cimport vector
from libc.stdint cimport uint8_t
from libcpp cimport bool
import numpy as np
cimport numpy as np


# externally declare the C++ classes
cdef extern from "./cpp/MM3DRenderer.h":

    cdef cppclass MM3DRenderer:
        MM3DRenderer(double* tpsCoord_in, float* coord_in, size_t numCoords_in,
                     unsigned int* coordIndex_in, size_t numTriangles_in,
                     float* texCoord_in, uint8_t* textureImage_in,
                     size_t textureWidth_in, size_t textureHeight_in,
                     bool INTERACTIVE_MODE)
        void returnFBPixels(int argc, char **argv, uint8_t* pixels,
                            float* coords, int width,
                            int height)


cdef class OpenGLRasterizer:
    cdef MM3DRenderer* thisptr
    cdef unsigned t_width
    cdef unsigned t_height
    cdef unsigned n_points
    cdef unsigned n_tris

    def __cinit__(self,
                  np.ndarray[double, ndim=2, mode="c"] points not None ,
                  np.ndarray[float, ndim=2, mode="c"] color not None,
                  np.ndarray[unsigned, ndim=2, mode="c"] trilist not None,
                  np.ndarray[float, ndim=2, mode="c"] tcoords not None,
                  np.ndarray[uint8_t, ndim=3, mode="c"] texture not None):
        self.t_height = texture.shape[0]
        self.t_width = texture.shape[1]
        self.n_points = color.shape[0]
        self.n_tris = trilist.shape[0]
        self.thisptr = new MM3DRenderer(&points[0,0], &color[0,0],
                                         self.n_points, &trilist[0,0],
                                         self.n_tris, &tcoords[0,0],
                                         &texture[0,0,0],
                                         self.t_width, self.t_height,
                                         False)

    def __dealloc__(self):
       del self.thisptr

    def pixels(self, render_width, render_height):
        cdef np.ndarray[uint8_t, ndim=3, mode='c'] pixels = \
            np.empty((render_width, render_height, 4), dtype=np.uint8)
        cdef np.ndarray[float, ndim=3, mode='c'] coords = \
            np.empty((render_width, render_height, 3), dtype=np.float32)
        cdef pystring = 'hello'.encode('UTF-8')
        cdef char* b = pystring
        self.thisptr.returnFBPixels(1, &b, &pixels[0,0,0], &coords[0,0,0],
                                    render_width, render_height)
        return pixels, coords
