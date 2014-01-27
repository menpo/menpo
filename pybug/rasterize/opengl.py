import numpy as np
from pybug.rasterize.copengl import COpenGLRasterizer
from pybug.image import MaskedImage


# noinspection PyProtectedMember
class GLRasterizer(object):

    def __init__(self, width, height):
        self._opengl = COpenGLRasterizer(width, height)

    @property
    def model_matrix(self):
        return self._opengl.get_model_matrix()

    @property
    def view_matrix(self):
        return self._opengl.get_view_matrix()

    @property
    def projection_matrix(self):
        return self._opengl.get_projection_matrix()

    def set_model_matrix(self, value):
        value = _verify_opengl_homogeneous_matrix(value)
        self._opengl.set_model_matrix(value)

    def set_view_matrix(self, value):
        value = _verify_opengl_homogeneous_matrix(value)
        self._opengl.set_view_matrix(value)

    def set_projection_matrix(self, value):
        value = _verify_opengl_homogeneous_matrix(value)
        self._opengl.set_projection_matrix(value)

    def rasterize(self, rasterizable):
        if rasterizable._rasterize_type_texture:
            # request the
            r = rasterizable._rasterize_generate_textured_mesh()
        elif rasterizable._rasterize_type_color():
            raise ValueError("Color Mesh rasterizations are not supported "
                             "yet")

    def rasterize_with_f3v_interpolant(self, rasterizable,
                                       f3v_interpolant):
        if rasterizable._rasterize_type_texture:
            # request the textured info for rasterizing
            r = rasterizable._rasterize_generate_textured_mesh()
            return self._rasterize_texture_with_interp(r, f3v_interpolant)
        elif rasterizable._rasterize_type_color():
            raise ValueError("Color Mesh rasterizations are not supported "
                             "yet")

    def rasterize_with_shape_image(self, rasterizable):
        return self.rasterize_with_f3v_interpolant(rasterizable, None)

    def rasterize(self, rasterizable):
        return self.rasterize_with_shape_image(rasterizable)[0]

    def _rasterize_texture_with_interp(self, r, per_vertex_f3v=None):
        r"""Rasterizes a textured mesh along with it's interpolant data
        through OpenGL.

        Parameters
        ----------
        r : object
            Any object with fields named 'points', 'trilist', 'texture' and
            'tcoords' specifying the data that will be used to render. Such
            objects are handed out by the
            _rasterize_generate_textured_mesh method on Rasterizable
            subclasses
        per_vertex_f3v: ndarray, shape (n_points, 3)
            A matrix specifying arbitrary 3 floating point numbers per
            vertex. This data will be linearly interpolated across triangles
            and returned in the f3v image. If none, the shape information is used

        Returns
        -------
        image : MaskedImage
            The rasterized image returned from OpenGL. Note that the
            behavior of the rasterization is governed by the projection,
            rotation and view matrices that may be set on this class,
            as well as the width and height of the rasterization, which is
            determined on the creation of this class. The mask is True if a
            triangle is visable at that pixel in the output, and False if not.

        f3v_image : MaskedImage
            The rasterized image returned from OpenGL. Note that the
            behavior of the rasterization is governed by the projection,
            rotation and view matrices that may be set on this class,
            as well as the width and height of the rasterization, which is
            determined on the creation of this class.

        """
        points = np.require(r.points, dtype=np.float32, requirements='c')
        trilist = np.require(r.trilist, dtype=np.uint32, requirements='c')
        texture = np.require(r.texture, dtype=np.float32, requirements='c')
        tcoords = np.require(r.tcoords, dtype=np.float32, requirements='c')
        if per_vertex_f3v is None:
            per_vertex_f3v = points
        interp = np.require(per_vertex_f3v, dtype=np.float32, requirements='c')
        rgb_fb, f3v_fb = self._opengl.render_offscreen_rgb(
            points, interp, trilist, tcoords, texture)
        mask_array = ~rgb_fb[..., 3].astype(np.bool)
        return (MaskedImage(rgb_fb[..., :3].copy(), mask=mask_array),
                MaskedImage(f3v_fb.copy(), mask=mask_array))


def _verify_opengl_homogeneous_matrix(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("OpenGL matrices must have shape (4,4)")
    return np.require(matrix, dtype=np.float32, requirements='C')
