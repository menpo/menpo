import numpy as np
from cyrasterize.base import CyRasterizerBase
from menpo.image import MaskedImage


# noinspection PyProtectedMember
from menpo.rasterize.base import TextureRasterInfo

# Subclass the CyRasterizerBase class to add Menpo-specific features
class GLRasterizer(CyRasterizerBase):

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
        # make a call out to the CyRasterizer _rasterize method
        rgb_pixels, f3v_pixels, mask = self._rasterize(
            r.points, r.trilist, r.texture, r.tcoords,
            per_vertex_f3v=per_vertex_f3v)
        return (MaskedImage(rgb_pixels, mask=mask),
                MaskedImage(f3v_pixels, mask=mask))
