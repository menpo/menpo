import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Scale

from ..adjacency import mask_adjacency_array, reindex_adjacency_array
from .base import TriMesh


class TexturedTriMesh(TriMesh):
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

    def from_vector(self, flattened):
        r"""
        Builds a new :class:`TexturedTriMesh` given then `flattened` vector.
        Note that the trilist, texture, and tcoords will be drawn from self.

        Parameters
        ----------
        flattened : (N,) ndarray
            Vector representing a set of points.

        Returns
        --------
        trimesh : :class:`TriMesh`
            A new trimesh created from the vector with self's trilist.
        """
        return TexturedTriMesh(flattened.reshape([-1, self.n_dims]),
                               self.tcoords.points, self.texture,
                               trilist=self.trilist)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the TexturedTriMesh. This is then broadcast across the
        dimensions of the mesh and returns a new mesh containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        mesh : :map:`TexturedTriMesh`
            A new mesh that has been masked.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this '
                             'TexturedTriMesh.')

        ttm = self.copy()
        if np.all(mask):  # Fast path for all true
            return ttm
        else:
            # Recalculate the mask to remove isolated vertices
            isolated_mask = self._isolated_mask(mask)
            # Recreate the adjacency array with the updated mask
            masked_adj = mask_adjacency_array(isolated_mask, self.trilist)
            ttm.trilist = reindex_adjacency_array(masked_adj)
            ttm.points = ttm.points[isolated_mask, :]
            ttm.tcoords.points = ttm.tcoords.points[isolated_mask, :]
            return ttm

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
        Menpo TexturedTriMeshes) could use the path property to locate the
        original texture on disk for clients and serve this directly.
        """
        json_dict = TriMesh.tojson(self)
        json_dict['tcoords'] = self.tcoords.tojson()['points']
        return json_dict

    def view(self, figure_id=None, new_figure=False, textured=True, **kwargs):
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
                try:
                    from menpo3d.visualize import TexturedTriMeshViewer3d
                    return TexturedTriMeshViewer3d(
                        figure_id, new_figure, self.points,
                        self.trilist, self.texture,
                        self.tcoords.points).render(**kwargs)
                except ImportError:
                    from menpo.visualize import Menpo3dErrorMessage
                    raise ImportError(Menpo3dErrorMessage)
            else:
                raise ValueError("Only viewing of 3D textured meshes"
                                 "is currently supported.")
        else:
            return super(TexturedTriMesh, self).view(figure_id=figure_id,
                                                     new_figure=new_figure,
                                                     **kwargs)

    def __str__(self):
        return '{}\ntexture_shape: {}, n_texture_channels: {}'.format(
            TriMesh.__str__(self), self.texture.shape, self.texture.n_channels)
