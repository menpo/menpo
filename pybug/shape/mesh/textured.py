from pybug.exceptions import DimensionalityError
from pybug.shape import PointCloud
from pybug.shape.mesh import TriMesh
from pybug.visualize import TexturedTriMeshViewer3d


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

    # def attach_texture(self, texture, tcoords, tcoords_trilist=None):
    #     """Attaches a trifield or pointfield called 'tcoords' depending
    #     on whether the tcoords given are per vertex or per triangle.
    #     kwargs:
    #        tcoords_trilist: a texture specific trilist used to index into
    #        the tcoords. In this case tcoords will be converted to a trifield
    #        removing the dependency on the texture specific trilist.
    #        This comes at a memory cost (there will be many repeated tcoords in
    #        the constructed trifield), but allows for a consistent processing
    #        of texture coords as just another field instance.
    #     """
    #     self.texture = texture
    #     if tcoords_trilist is not None:
    #         # looks like we have tcoords that are referenced into
    #         # by a trilist in the same way points are. As it becomes messy to
    #         # maintain different texturing options, we just turn this indexing
    #         # scheme into (repeated) values stored explicitly as a trifield.
    #         self.add_trifield('tcoords', tcoords[tcoords_trilist])
    #     elif tcoords.shape == (self.n_points, 2):
    #         # tcoords are just per vertex
    #         self.add_pointfield('tcoords', tcoords)
    #     elif tcoords.shape == (self.n_tris, 3, 2):
    #         # explicitly given per triangle vertex
    #         self.add_trifield('tcoords', tcoords)
    #     else:
    #         raise TextureError(
    #             "Don't understand how to deal with these tcoords.")

    def view(self, textured=True):
        """
        Visualize the :class:`TexturedTriMesh`. Only 3D objects are currently
        supported.

        Parameters
        ----------
        textured : bool, optional
            If ``True``, render the textur.

            Default: ``True``

        Returns
        -------
        viewer : :class:`pybug.visualize.base.Viewer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If ``self.n_dims != 3``.
        """
        if textured:
            if self.n_dims == 3:
                viewer = TexturedTriMeshViewer3d(
                    self.points, self.trilist, self.texture,
                    tcoords_per_point=self.tcoords.points)
                return viewer.view()
            else:
                raise DimensionalityError("Only viewing of 3D textured meshes"
                                          "is currently supported.")
        else:
            return super(TexturedTriMesh, self).view()