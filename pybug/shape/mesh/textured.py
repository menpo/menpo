import numpy as np
from pybug.exception import DimensionalityError
from pybug.shape import PointCloud
from pybug.shape.mesh import TriMesh
from pybug.visualize import TexturedTriMeshViewer3d
from pybug.transform.affine import Scale


class TexturedTriMesh(TriMesh):
    r"""
    Combines a :class:`pybug.shape.mesh.base.TriMesh` with a texture. Also
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
    texture : :class:`pybug.image.base.Image`
        The texture for the mesh.
    """

    def __init__(self, points, trilist, tcoords, texture):
        super(TexturedTriMesh, self).__init__(points, trilist)
        self.tcoords = PointCloud(tcoords)
        self.texture = texture

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
        tcoords_scaled : :class:`pybug.shape.PointCloud`
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
        viewer : :class:`pybug.visualize.base.Renderer`
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