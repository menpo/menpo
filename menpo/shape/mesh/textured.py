import copy
import numpy as np

from menpo.rasterize.base import TextureRasterInfo
from menpo.shape import PointCloud
from menpo.visualize import TexturedTriMeshViewer3d
from menpo.transform import Scale
from menpo.rasterize import Rasterizable
from menpo.exception import DimensionalityError

from .base import TriMesh


class TexturedTriMesh(TriMesh, Rasterizable):
    r"""
    Combines a :class:`menpo.shape.mesh.base.TriMesh` with a texture. Also
    encapsulates the texture coordinates required to render the texture on the
    mesh.

    Parameters
    ----------
    points : (N, D) ndarray
        The coordinates of the mesh.
    trilist : (M, 3) ndarray
        The triangle list for the mesh
    tcoords : (N, 2) ndarray
        The texture coordinates for the mesh.
    texture : :class:`menpo.image.base.Image`
        The texture for the mesh.
    """

    def __init__(self, points, trilist, tcoords, texture):
        super(TexturedTriMesh, self).__init__(points, trilist)
        self.tcoords = PointCloud(tcoords)
        self.texture = copy.deepcopy(texture)

    def tcoords_pixel_scaled(self):
        r"""
        Returns a PointCloud that is modified to be suitable for directly
        indexing into the pixels of the texture (e.g. for manual mapping
        operations). The resulting tcoords behave just like image landmarks
        do:

         >>> texture = texturedtrimesh.texture
         >>> tc_ps = texturedtrimesh.tcoords_pixel_scaled()
         >>> pixel_values_at_tcs = texture[tc_ps[: ,0], tc_ps[:, 1]]

        The operations that are performed are:

        - Flipping the origin from bottom-left to top-left
        - Scaling the tcoords by the image shape (denormalising them)
        - Permuting the axis so that

        Returns
        -------
        tcoords_scaled : :class:`menpo.shape.PointCloud`
            A copy of the tcoords that behave like Image landmarks
        """
        scale = Scale(np.array(self.texture.shape)[::-1])
        tcoords = self.tcoords.points.copy()
        # flip the 'y' st 1 -> 0 and 0 -> 1, moving the axis to upper left
        tcoords[:, 1] = 1 - tcoords[:, 1]
        # apply the scale to get the units correct
        tcoords = scale.apply(tcoords)
        # flip axis 0 and axis 1 so indexing is as expected
        tcoords = tcoords[:, ::-1]
        return PointCloud(tcoords)

    def _view(self, figure_id=None, new_figure=False, textured=True, **kwargs):
        r"""
        Visualize the :class:`TexturedTriMesh`. Only 3D objects are currently
        supported.

        Parameters
        ----------
        textured : bool, optional
            If ``True``, render the texture.

            Default: ``True``

        Returns
        -------
        viewer : :class:`menpo.visualize.base.Renderer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If ``self.n_dims != 3``.
        """
        if textured:
            if self.n_dims == 3:
                return TexturedTriMeshViewer3d(
                    figure_id, new_figure, self.points,
                    self.trilist, self.texture,
                    self.tcoords.points).render(**kwargs)
            else:
                raise DimensionalityError("Only viewing of 3D textured meshes"
                                          "is currently supported.")
        else:
            return super(TexturedTriMesh, self)._view(figure_id=figure_id,
                                                      new_figure=new_figure,
                                                      **kwargs)

    def __str__(self):
        return '{}\ntexture_shape: {}, n_texture_channels: {}'.format(
            TriMesh.__str__(self), self.texture.shape, self.texture.n_channels)

    @property
    def _rasterize_type_texture(self):
        return True  # TexturedTriMesh can specify texture rendering params

    def _rasterize_generate_textured_mesh(self):
        return TextureRasterInfo(self.points, self.trilist,
                                 self.tcoords.points,
                                 self.texture.pixels)
