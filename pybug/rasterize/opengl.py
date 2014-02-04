import numpy as np
from pybug.rasterize.copengl import COpenGLRasterizer
from pybug.image import MaskedImage


# noinspection PyProtectedMember
from pybug.rasterize.base import TextureRasterInfo


class GLRasterizer(object):

    def __init__(self, width=1024, height=768, model_matrix=None,
                 view_matrix=None, projection_matrix=None):
        r"""Offscreen OpenGL rasterize of fixed width and height.

        Parameters
        ----------

        width : int
            The width of the rasterize target

        height: int
            The height of hte rasterize target


        Notes
        -----

        For a given vertex v = (x, y, z, 1), it's position in image space
        v' = (s, t) is calculated from

        v' = P * V * M * v

        where:

        M is the model matrix
        V is the view matrix (view the world from the position of the camera)
        P is the projection matrix (usually an orthographic or perspective
        matrix)

        All matrices are 4x4 floats, as in OpenGL all points are treated as
        homogeneous.

        Note that this is the raw code written in the shader. The usual
        pipeline of OpenGL applies - perspective division is performed to
        form a clip space, and z-buffering is used to mask pixels
        appropriately.

        Texture information in the form of a texture map and normalized
        per-vertex texture coordinates) are used to source colour values.

        An arbitrary float 3-vector (f3v) can also be set on each vertex.
        This value is passed through the same pipeline and interpolated but
        note that the MATRICES ABOVE ARE NOT APPLIED TO THIS DATA.

        This can be useful for example for passing through the shape
        information of
        the object into the rendered image domain. Note that because of the
        above statement, the shape information rendered would be in the
        objects original space, not in camera space (i.e. the z value will
        not correlate to a depth buffer).

        """
        self._opengl = COpenGLRasterizer(width, height)
        if model_matrix is not None:
            self.set_model_matrix(model_matrix)
        if view_matrix is not None:
            self.set_view_matrix(view_matrix)
        if projection_matrix is not None:
            self.set_projection_matrix(projection_matrix)

    @property
    def width(self):
        return self._opengl.get_width()

    @property
    def height(self):
        return self._opengl.get_height()

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

    def rasterize_mesh_with_f3v_interpolant(self, rasterizable,
                                            per_vertex_f3v=None):
        r"""Rasterize the object to an image and generate an interpolated
        3-float image from a per vertex float 3 vector.

        If no per_vertex_f3v is provided, the model's shape is used (making
        this method equivalent to rasterize_mesh_with_shape_image)

        Parameters
        ----------
        rasterizable : object implementing the Rasterizable interface.
            Will be queried for some state to rasterize via the Rasterizable
            interface. Note that currently, color mesh rasterizations are
            not supported.

        per_vertex_f3v : optional, ndarray (n_points, 3)
            A per-vertex 3 vector of floats that will be interpolated across
            the image.


        Returns
        -------
        rgb_image : 3 channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.

        interp_image: 3 channel MaskedImage of shape (width, height)
            The result of interpolating the per_vertex_f3v across the
            visible primitives.

        """
        if rasterizable._rasterize_type_texture:
            # request the textured info for rasterizing
            r = rasterizable._rasterize_generate_textured_mesh()
            return self._rasterize_texture_with_interp(
                r, per_vertex_f3v=per_vertex_f3v)
        elif rasterizable._rasterize_type_colour:
            #TODO: This should use a different shader!
            # But I'm hacking it here to work quickly.
            colour_r = rasterizable._rasterize_generate_color_mesh()

            # Fake some texture coordinates and a texture as required by the
            # shader
            fake_tcoords = np.random.randn(colour_r.points.shape[0], 2)
            fake_texture = np.zeros([2, 2, 3])
            r = TextureRasterInfo(colour_r.points, colour_r.trilist,
                              fake_tcoords, fake_texture)

            # The RGB image is going to be broken due to the fake texture
            # information we passed in
            _, rgb_image = self._rasterize_texture_with_interp(
                r, per_vertex_f3v=colour_r.colours)
            _, f3v_image = self._rasterize_texture_with_interp(
                r, per_vertex_f3v=per_vertex_f3v)

            return rgb_image, f3v_image

    def rasterize_mesh_with_shape_image(self, rasterizable):
        r"""Rasterize the object to an image and generate an interpolated
        3-float image from the shape information on the rasterizable object.

        Parameters
        ----------
        rasterizable : object implementing the Rasterizable interface.
            Will be queried for some state to rasterize via the Rasterizable
            interface. Note that currently, color mesh rasterizations are
            not supported.


        Returns
        -------
        rgb_image : 3 channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.

        shape_image: 3 channel MaskedImage of shape (width, height)
            The result of interpolating the spatial information of each vertex
            across the visible primitives. Note that the shape information
            is *NOT* adjusted by the P,V,M matrices, and so the resulting
            shape image is always in the original objects reference shape
            (i.e. the z value will not necessarily correspond to a depth
            buffer).

        """
        return self.rasterize_mesh_with_f3v_interpolant(rasterizable)

    def rasterize_mesh(self, rasterizable):
        r"""Rasterize the object to an image and generate an interpolated
        3-float image from the shape information on the rasterizable object.

        Parameters
        ----------
        rasterizable : object implementing the Rasterizable interface.
            Will be queried for some state to rasterize via the Rasterizable
            interface. Note that currently, color mesh rasterizations are
            not supported.


        Returns
        -------
        rgb_image : 3 channel MaskedImage of shape (width, height)
            The result of the rasterization. Mask is true iff the pixel was
            rendered to by OpenGL.

        shape_image: 3 channel MaskedImage of shape (width, height)
            The result of interpolating the spatial information of each vertex
            across the visible primitives. Note that the shape information
            is *NOT* adjusted by the P,V,M matrices, and so the resulting
            shape image is always in the original objects reference shape
            (i.e. the z value will not necessarily correspond to a depth
            buffer).

        """
        return self.rasterize_mesh_with_shape_image(rasterizable)[0]

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
            and returned in the f3v image. If none, the shape information is
            used

        Returns
        -------
        image : MaskedImage
            The rasterized image returned from OpenGL. Note that the
            behavior of the rasterization is governed by the projection,
            rotation and view matrices that may be set on this class,
            as well as the width and height of the rasterization, which is
            determined on the creation of this class. The mask is True if a
            triangle is visible at that pixel in the output, and False if not.

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
        mask_array = rgb_fb[..., 3].astype(np.bool)
        return (MaskedImage(rgb_fb[..., :3].copy(), mask=mask_array),
                MaskedImage(f3v_fb.copy(), mask=mask_array))


def _verify_opengl_homogeneous_matrix(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("OpenGL matrices must have shape (4,4)")
    return np.require(matrix, dtype=np.float32, requirements='C')
