import copy
import numpy as np

from menpo.rasterize.base import TextureRasterInfo
from menpo.shape import PointCloud
from menpo.visualize import TexturedTriMeshViewer3d
from menpo.transform import Scale
from menpo.rasterize import Rasterizable

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
    tcoords : (N, 2) ndarray
        The texture coordinates for the mesh.
    texture : :class:`menpo.image.base.Image`
        The texture for the mesh.
    trilist : (M, 3) ndarray, optional
        The triangle list for the mesh. If `None`, a Delaunay triangulation
        will be performed.

        Default: `None`
    copy: bool, optional
        If `False`, the points, trilist and texture will not be copied on
        assignment.
        In general this should only be used if you know what you are doing.

        Default: `False`
    """

    def __init__(self, points, tcoords, texture, trilist=None, copy=True):
        super(TexturedTriMesh, self).__init__(points, trilist=trilist,
                                              copy=copy)
        self.tcoords = PointCloud(tcoords, copy=copy)

        if not copy:
            self.texture = texture
        else:
            self.texture = texture.copy()

    def copy(self):
        r"""
        An efficient copy of this TexturedTriMesh.

        Only landmarks and points will be transferred. For a full copy consider
        using `deepcopy()`.

        Returns
        -------
        texturedtrimesh: :map:`TexturedTriMesh`
            A TexturedTriMesh with the same points, trilist, tcoords, texture
            and landmarks as this one.
        """
        new_ttm = TexturedTriMesh(self.points, self.tcoords, self.texture,
                                  trilist=self.trilist, copy=True)
        new_ttm.landmarks = self.landmarks
        return new_ttm

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

    def tojson(self):
        r"""
        Convert this `TriMesh` to a dictionary JSON representation.

        Returns
        -------
        dictionary with 'points', 'trilist' and 'tcoords' keys. Both are lists
        suitable for use in the by the `json` standard library package.

        Note that textures are best transmitted in a native format like jpeg
        rather that in a JSON format. For this reason the texture itself is
        not encoded. Consumers of this method (e.g. a web server serving
        Menpo TexturedTriMeshes) could use the ioinfo property to locate the
        original texture on disk for clients and serve this directly.
        """
        json_dict = TriMesh.tojson(self)
        json_dict['tcoords'] = self.tcoords.tojson()['points']
        return json_dict

    def _view(self, figure_id=None, new_figure=False, textured=True, **kwargs):
        r"""
        Visualize the :class:`TexturedTriMesh`. Only 3D objects are currently
        supported.

        Parameters
        ----------
        textured : bool, optional
            If `True`, render the texture.

            Default: `True`

        Returns
        -------
        viewer : :class:`menpo.visualize.base.Renderer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If `self.n_dims != 3`.
        """
        if textured:
            if self.n_dims == 3:
                return TexturedTriMeshViewer3d(
                    figure_id, new_figure, self.points,
                    self.trilist, self.texture,
                    self.tcoords.points).render(**kwargs)
            else:
                raise ValueError("Only viewing of 3D textured meshes"
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
