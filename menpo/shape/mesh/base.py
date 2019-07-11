# coding=utf-8
from collections import Counter
import numpy as np
from warnings import warn

from .. import PointCloud
from ..adjacency import mask_adjacency_array, reindex_adjacency_array
from .normals import compute_vertex_normals, compute_face_normals


Delaunay = None  # expensive, from scipy.spatial


def grid_tcoords(shape):
    r"""
    Return texture coordinates laid out on a grid. This is useful for creating
    a textured version of an image whereby the underlying mesh maps
    1-1 with a texture. Therefore, the provided shape should be the shape
    of the texture.

    Parameters
    ----------
    shape : `tuple` of 2 `int`
        The size of the grid to create, this defines the number of points
        across each dimension in the grid. The first element is the number
        of rows and the second is the number of columns.

    Returns
    -------
    tcoords : ``(M, 2)`` `ndarray`
        The texture coordinates of a uniform grid. The origin will be
        at the image origin (appropriate for viewing texture mapped planes
        such as viewing image height maps).
    """
    # Default tcoords are just a grid, which assumes the input texture
    # is an image the same size as the input grid. The meshgrid is made in
    # an ordering that attempts to reduce the amount of copying required but
    # places the texture coordinates in the correct arrangement.
    tcoords = np.meshgrid(np.linspace(0, 1, num=shape[1]),
                          np.linspace(1, 0, num=shape[0]),
                          indexing='xy')
    tcoords = np.stack(tcoords, axis=2).reshape([-1, 2])
    tcoords = np.require(tcoords, requirements=['C'])
    return tcoords


def trilist_to_adjacency_array(trilist):
    r"""
    Turn an ``(M, 3)`` trilist into an adjacency array suitable for building
    graphs.

    Parameters
    ----------
    trilist : ``(M, 3)`` `ndarray`
        The trilist to transform into an adjacency array

    Returns
    -------
    adj_array : ``(M * 3, 2)`` `ndarray`
        The adjacency array including the edges that complete the triangle
        which are implicit in a trilist.
    """
    wrap_around_adj = np.hstack([trilist[:, -1][..., None],
                                 trilist[:, 0][..., None]])
    # Build the array of all pairs
    return np.concatenate([trilist[:, :2],
                           trilist[:, 1:],
                           wrap_around_adj])


def subsampled_grid_triangulation(shape, subsampling=1):
    r"""
    Create a triangulation based on a regular grid. This will be a right
    handed triangulation with the separating triangle edge going from
    the top left of a grid point to the bottom right.

    Optionally, the triangulation can be subsampled which has the effect
    of skipping points. This is useful for subsampling a dense pointcloud.

    Parameters
    ----------
    shape : `tuple` of 2 `int`
        The size of the grid to assume, this defines the number of points
        across each dimension in the grid. The first element is the number
        of rows and the second is the number of columns.
    subsampling : `int`, optional
        Will be used to index into the implicit grid and has the effect
        of subsampling the grid (every subsampling'th vertex is chosen).

    Returns
    -------
    trilist : ``(M, 3)`` `ndarray`
        The triangle list created on an implicit regular grid.
    """
    # Quickly create the indices in a grid
    indices_grid = np.zeros(shape)
    flat_vals_grid = indices_grid.ravel()
    flat_vals_grid[:] = np.arange(np.prod(shape))

    # Subsample the grid if necessary - useful for making very dense grids
    # much sparser
    indices_grid = indices_grid[::subsampling, ::subsampling]

    # Bottom-left triangles (right handed)
    tri_down_left = np.concatenate(
        [indices_grid[:-1, :-1].ravel()[..., None],
         indices_grid[1:, :-1].ravel()[..., None],
         indices_grid[1:, 1:].ravel()[..., None]], axis=-1)

    # Top-right triangles (right handed)
    tri_up_right = np.concatenate(
        [indices_grid[:-1, :-1].ravel()[..., None],
         indices_grid[1:, 1:].ravel()[..., None],
         indices_grid[:-1, 1:].ravel()[..., None]], axis=-1)

    return np.vstack([tri_down_left, tri_up_right]).astype(np.uint32)


class TriMesh(PointCloud):
    r"""
    A :map:`PointCloud` with a connectivity defined by a triangle list. These
    are designed to be explicitly 2D or 3D.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The array representing the points.
    trilist : ``(M, 3)`` `ndarray` or ``None``, optional
        The triangle list. If `None`, a Delaunay triangulation of
        the points will be used instead.
    copy: `bool`, optional
        If ``False``, the points will not be copied on assignment.
        Any trilist will also not be copied.
        In general this should only be used if you know what you are doing.
    """
    def __init__(self, points, trilist=None, copy=True):
        super(TriMesh, self).__init__(points, copy=copy)
        if trilist is None:
            global Delaunay
            if Delaunay is None:
                from scipy.spatial import Delaunay  # expensive
            trilist = Delaunay(points).simplices
        if not copy:
            if not trilist.flags.c_contiguous:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
                trilist = np.array(trilist, copy=True, order='C')
        else:
            trilist = np.array(trilist, copy=True, order='C')
        self.trilist = trilist

    @classmethod
    def init_2d_grid(cls, shape, spacing=None):
        r"""
        Create a TriMesh that exists on a regular 2D grid. The first
        dimension is the number of rows in the grid and the second dimension
        of the shape is the number of columns. ``spacing`` optionally allows
        the definition of the distance between points (uniform over points).
        The spacing may be different for rows and columns.

        The triangulation will be right-handed and the diagonal will go from
        the top left to the bottom right of a square on the grid.

        Parameters
        ----------
        shape : `tuple` of 2 `int`
            The size of the grid to create, this defines the number of points
            across each dimension in the grid. The first element is the number
            of rows and the second is the number of columns.
        spacing : `int` or `tuple` of 2 `int`, optional
            The spacing between points. If a single `int` is provided, this
            is applied uniformly across each dimension. If a `tuple` is
            provided, the spacing is applied non-uniformly as defined e.g.
            ``(2, 3)`` gives a spacing of 2 for the rows and 3 for the
            columns.

        Returns
        -------
        trimesh : :map:`TriMesh`
            A TriMesh arranged in a grid.
        """
        pc = PointCloud.init_2d_grid(shape, spacing=spacing)
        points = pc.points
        return cls(points, trilist=subsampled_grid_triangulation(
            shape, subsampling=1), copy=False)

    @classmethod
    def init_from_depth_image(cls, depth_image):
        r"""
        Return a 3D triangular mesh from the given depth image. The depth image
        is assumed to represent height/depth values and the XY coordinates
        are assumed to unit spaced and represent image coordinates. This is
        particularly useful for visualising depth values that have been
        recovered from images.

        Parameters
        ----------
        depth_image : :map:`Image` or subclass
            A single channel image that contains depth values - as commonly
            returned by RGBD cameras, for example.

        Returns
        -------
        depth_cloud : ``type(cls)``
            A new 3D TriMesh with unit XY coordinates and the given depth
            values as Z coordinates. The trilist is constructed as in
            :meth:`init_2d_grid`.
        """
        from menpo.image import MaskedImage

        new_tmesh = cls.init_2d_grid(depth_image.shape)
        if isinstance(depth_image, MaskedImage):
            new_tmesh = new_tmesh.from_mask(depth_image.mask.as_vector())
        return cls(np.hstack([new_tmesh.points,
                              depth_image.as_vector(keep_channels=True).T]),
                   trilist=new_tmesh.trilist,
                   copy=False)

    def __str__(self):
        return '{}, n_tris: {}'.format(PointCloud.__str__(self),
                                       self.n_tris)

    @property
    def n_tris(self):
        r"""
        The number of triangles in the triangle list.

        :type: `int`
        """
        return len(self.trilist)

    def tojson(self):
        r"""
        Convert this :map:`TriMesh` to a dictionary representation suitable
        for inclusion in the LJSON landmark format. Note that this enforces a
        simpler representation, and as such is not suitable for
        a permanent serialization of a :map:`TriMesh` (to be clear,
        :map:`TriMesh`'s serialized as part of a landmark set will be rebuilt
        as a :map:`PointUndirectedGraph`).

        Returns
        -------
        json : `dict`
            Dictionary with ``points`` and ``connectivity`` keys.
        """
        return self.as_pointgraph().tojson()

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the TriMesh. This is then broadcast across the dimensions
        of the mesh and returns a new mesh containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        mesh : :map:`TriMesh`
            A new mesh that has been masked.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this TriMesh.')

        tm = self.copy()
        if np.all(mask):  # Fast path for all true
            return tm
        else:
            # Recalculate the mask to remove isolated vertices
            isolated_mask = self._isolated_mask(mask)
            # Recreate the adjacency array with the updated mask
            masked_adj = mask_adjacency_array(isolated_mask, self.trilist)
            tm.trilist = reindex_adjacency_array(masked_adj)
            tm.points = tm.points[isolated_mask, :]
            return tm

    def from_tri_mask(self, tri_mask):
        """
        A 1D boolean array with the same number of elements as the number of
        triangles in the TriMesh. This is then broadcast across the dimensions
        of the mesh and returns a new mesh containing only those
        triangles that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_tris,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        mesh : :map:`TriMesh`
            A new mesh that has been masked by triangles.
        """
        # start with an all False point mask.
        point_mask = np.zeros(self.n_points, dtype=np.bool)
        # find all points that are involved in the triangles we wish to
        # retain and set their mask to True.
        point_mask[np.unique(self.trilist[tri_mask].ravel())] = True
        return self.from_mask(point_mask)

    def _isolated_mask(self, mask):
        # Find the triangles we need to keep
        masked_adj = mask_adjacency_array(mask, self.trilist)
        # Find isolated vertices (vertices that don't exist in valid
        # triangles)
        isolated_indices = np.setdiff1d(np.nonzero(mask)[0], masked_adj)

        # Create a 'new mask' that contains the points the use asked
        # for MINUS the points that we can't create triangles for
        new_mask = mask.copy()
        new_mask[isolated_indices] = False
        return new_mask

    def as_pointgraph(self, copy=True, skip_checks=False):
        """
        Converts the TriMesh to a :map:`PointUndirectedGraph`.

        Parameters
        ----------
        copy : `bool`, optional
            If ``True``, the graph will be a copy.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.

        Returns
        -------
        pointgraph : :map:`PointUndirectedGraph`
            The point graph.
        """
        from .. import PointUndirectedGraph
        from ..graph import _convert_edges_to_symmetric_adjacency_matrix
        # Since we have triangles we need the last connection
        # that 'completes' the triangle
        adjacency_matrix = _convert_edges_to_symmetric_adjacency_matrix(
            trilist_to_adjacency_array(self.trilist), self.points.shape[0])
        pg = PointUndirectedGraph(self.points, adjacency_matrix, copy=copy,
                                  skip_checks=skip_checks)
        # This is always a copy
        pg.landmarks = self.landmarks
        return pg

    def vertex_normals(self):
        r"""
        Compute the per-vertex normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Returns
        -------
        normals : ``(n_points, 3)`` `ndarray`
            Normal at each point.

        Raises
        ------
        ValueError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise ValueError("Normals are only valid for 3D meshes")
        return compute_vertex_normals(self.points, self.trilist)

    def tri_normals(self):
        r"""
        Compute the triangle face normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Returns
        -------
        normals : ``(n_tris, 3)`` `ndarray`
            Normal at each triangle face.

        Raises
        ------
        ValueError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise ValueError("Normals are only valid for 3D meshes")
        return compute_face_normals(self.points, self.trilist)

    def tri_areas(self):
        r"""The area of each triangle face.

        Returns
        -------
        areas : ``(n_tris,)`` `ndarray`
            Area of each triangle, ordered as the trilist is

        Raises
        ------
        ValueError
            If mesh is not 2D or 3D
        """
        t = self.points[self.trilist]
        ij, ik = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        if self.n_dims == 2:
            return np.abs(np.cross(ij, ik) * 0.5)
        elif self.n_dims == 3:
            return np.linalg.norm(np.cross(ij, ik), axis=1) * 0.5
        else:
            raise ValueError('tri_areas can only be calculated on a 2D or '
                             '3D mesh')

    def mean_tri_area(self):
        r"""The mean area of each triangle face in this :map:`TriMesh`.

        Returns
        -------
        mean_tri_area : ``float``
            The mean area of each triangle face in this :map:`TriMesh`

        Raises
        ------
        ValueError
            If mesh is not 3D
        """
        return np.mean(self.tri_areas())

    def boundary_tri_index(self):
        r"""Boolean index into triangles that are at the edge of the TriMesh

        Returns
        -------
        boundary_tri_index : ``(n_tris,)`` `ndarray`
            For each triangle (ABC), returns whether any of it's edges is not
            also an edge of another triangle (and so this triangle exists on
            the boundary of the TriMesh)
        """
        trilist = self.trilist
        # Get a sorted list of edge pairs
        edge_pairs = np.sort(np.vstack((trilist[:, [0, 1]],
                                        trilist[:, [0, 2]],
                                        trilist[:, [1, 2]])))

        # convert to a tuple per edge pair
        edges = [tuple(x) for x in edge_pairs]
        # count the occurrences of the ordered edge pairs - edge pairs that
        # occur once are at the edge of the whole mesh
        mesh_edges = (e for e, i in Counter(edges).items() if i == 1)
        # index back into the edges to find which triangles contain these edges
        return np.array(list(set(edges.index(e) % trilist.shape[0]
                                 for e in mesh_edges)))

    def edge_vectors(self):
        r"""A vector of edges of each triangle face.

        Note that there will be two edges present in cases where two triangles
        'share' an edge. Consider :meth:`unique_edge_vectors` for a
        single vector for each physical edge on the :map:`TriMesh`.

        Returns
        -------
        edges : ``(n_tris * 3, n_dims)`` `ndarray`
            For each triangle (ABC), returns the edge vectors AB, BC, CA. All
            edges are concatenated for a total of ``n_tris * 3`` edges. The
            ordering is done so that all AB vectors are first in the returned
            list, followed by BC, then CA.
        """
        t = self.points[self.trilist]
        return np.vstack((t[:, 1] - t[:, 0],
                          t[:, 2] - t[:, 1],
                          t[:, 2] - t[:, 0]))

    def edge_indices(self):
        r"""An unordered index into points that rebuilds the edges of this
        :map:`TriMesh`.

        Note that there will be two edges present in cases where two triangles
        'share' an edge. Consider :meth:`unique_edge_indices` for a single index
        for each physical edge on the :map:`TriMesh`.

        Returns
        -------
        edge_indices : ``(n_tris * 3, 2)`` `ndarray`
            For each triangle (ABC), returns the pair of point indices that
            rebuild AB, AC, BC. All edge indices are concatenated for a total
            of ``n_tris * 3`` edge_indices. The ordering is done so that all
            AB vectors are first in the returned list, followed by BC, then CA.
        """
        tl = self.trilist
        return np.vstack((tl[:, [0, 1]],
                          tl[:, [1, 2]],
                          tl[:, [2, 0]]))

    def unique_edge_indices(self):
        r"""An unordered index into points that rebuilds the unique edges of
        this :map:`TriMesh`.

        Note that each physical edge will only be counted once in this method
        (i.e. edges shared between neighbouring triangles are only counted once
        not twice). The ordering should be considered random.

        Returns
        -------
        unique_edge_indices : ``(n_unique_edges, 2)`` `ndarray`
            Return a point index that rebuilds all edges present in this
            :map:`TriMesh` only once.
        """
        # Get a sorted list of edge pairs. sort ensures that each edge is
        # ordered from lowest index to highest.
        edge_pairs = np.sort(self.edge_indices())

        # We want to remove duplicates - this is a little hairy: basically we
        # get a view on the array where each pair is considered by numpy to be
        # one item
        edge_pair_view = np.ascontiguousarray(edge_pairs).view(
            np.dtype((np.void, edge_pairs.dtype.itemsize * edge_pairs.shape[1])))
        # Now we can use this view to ask for only unique edges...
        unique_edge_index = np.unique(edge_pair_view, return_index=True)[1]
        # And use that to filter our original list down
        return edge_pairs[unique_edge_index]

    def unique_edge_vectors(self):
        r"""An unordered vector of unique edges for the whole :map:`TriMesh`.

        Note that each physical edge will only be counted once in this method
        (i.e. edges shared between neighbouring triangles are only counted once
        not twice). The ordering should be considered random.

        Returns
        -------
        unique_edge_vectors : ``(n_unique_edges, n_dims)`` `ndarray`
            Vectors for each unique edge in this :map:`TriMesh`.
        """
        x = self.points[self.unique_edge_indices()]
        return x[:, 1] - x[:, 0]

    def edge_lengths(self):
        r"""The length of each edge in this :map:`TriMesh`.

        Note that there will be two edges present in cases where two triangles
        'share' an edge. Consider :meth:`unique_edge_indices` for a single
        index for each physical edge on the :map:`TriMesh`. The ordering
        matches the case for edges and edge_indices.

        Returns
        -------
        edge_lengths : ``(n_tris * 3, )`` `ndarray`
            Scalar euclidean lengths for each edge in this :map:`TriMesh`.
        """
        return np.linalg.norm(self.edge_vectors(), axis=1)

    def unique_edge_lengths(self):
        r"""The length of each edge in this :map:`TriMesh`.

        Note that each physical edge will only be counted once in this method
        (i.e. edges shared between neighbouring triangles are only counted once
        not twice). The ordering should be considered random.

        Returns
        -------
        edge_lengths : ``(n_tris * 3, )`` `ndarray`
            Scalar euclidean lengths for each edge in this :map:`TriMesh`.
        """
        return np.linalg.norm(self.unique_edge_vectors(), axis=1)

    def mean_edge_length(self, unique=True):
        r"""The mean length of each edge in this :map:`TriMesh`.

        Parameters
        ----------
        unique : `bool`, optional
            If ``True``, each shared edge will only be counted once towards
            the average. If false, shared edges will be counted twice.

        Returns
        -------
        mean_edge_length : ``float``
            The mean length of each edge in this :map:`TriMesh`
        """
        return np.mean(self.unique_edge_lengths() if unique
                       else self.edge_lengths())

    def _view_2d(self, figure_id=None, new_figure=False, image_view=True,
                 render_lines=True, line_colour='r', line_style='-',
                 line_width=1., render_markers=True, marker_style='o',
                 marker_size=5, marker_face_colour='k', marker_edge_colour='k',
                 marker_edge_width=1., render_numbering=False,
                 numbers_horizontal_align='center',
                 numbers_vertical_align='bottom',
                 numbers_font_name='sans-serif', numbers_font_size=10,
                 numbers_font_style='normal', numbers_font_weight='normal',
                 numbers_font_colour='k', render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                 axes_y_ticks=None, figure_size=(7, 7), label=None, **kwargs):
        r"""
        Visualization of the TriMesh in 2D.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True`` the TriMesh will be viewed as if it is in the image
            coordinate system.
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : ``{-, --, -., :}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : ``{center, right, left}``, optional
            The horizontal alignment of the numbers' texts.
        numbers_vertical_align : ``{center, top, bottom, baseline}``, optional
            The vertical alignment of the numbers' texts.
        numbers_font_name : See Below, optional
            The font of the numbers. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : See Below, optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the TriMesh as a percentage of the TriMesh's
            width. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the TriMesh as a percentage of the TriMesh's
            height. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.
        label : `str`, optional
            The name entry in case of a legend.

        Returns
        -------
        viewer : :map:`PointGraphViewer2d`
            The viewer object.
        """
        from menpo.visualize import PointGraphViewer2d

        return PointGraphViewer2d(
            figure_id, new_figure, self.points,
            trilist_to_adjacency_array(self.trilist)).render(
                image_view=image_view, render_lines=render_lines,
                line_colour=line_colour, line_style=line_style,
                line_width=line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width,
                render_numbering=render_numbering,
                numbers_horizontal_align=numbers_horizontal_align,
                numbers_vertical_align=numbers_vertical_align,
                numbers_font_name=numbers_font_name,
                numbers_font_size=numbers_font_size,
                numbers_font_style=numbers_font_style,
                numbers_font_weight=numbers_font_weight,
                numbers_font_colour=numbers_font_colour, render_axes=render_axes,
                axes_font_name=axes_font_name, axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
                axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
                axes_y_ticks=axes_y_ticks, figure_size=figure_size, label=label)

    def _view_landmarks_2d(self, group=None, with_labels=None,
                           without_labels=None, figure_id=None,
                           new_figure=False, image_view=True,
                           render_lines=True, line_colour='k',
                           line_style='-', line_width=2,
                           render_markers=True, marker_style='s', marker_size=7,
                           marker_face_colour='k', marker_edge_colour='k',
                           marker_edge_width=1., render_lines_lms=True,
                           line_colour_lms=None, line_style_lms='-',
                           line_width_lms=1, render_markers_lms=True,
                           marker_style_lms='o', marker_size_lms=5,
                           marker_face_colour_lms=None,
                           marker_edge_colour_lms=None,
                           marker_edge_width_lms=1., render_numbering=False,
                           numbers_horizontal_align='center',
                           numbers_vertical_align='bottom',
                           numbers_font_name='sans-serif', numbers_font_size=10,
                           numbers_font_style='normal',
                           numbers_font_weight='normal',
                           numbers_font_colour='k', render_legend=False,
                           legend_title='', legend_font_name='sans-serif',
                           legend_font_style='normal', legend_font_size=10,
                           legend_font_weight='normal',
                           legend_marker_scale=None, legend_location=2,
                           legend_bbox_to_anchor=(1.05, 1.),
                           legend_border_axes_pad=None, legend_n_columns=1,
                           legend_horizontal_spacing=None,
                           legend_vertical_spacing=None, legend_border=True,
                           legend_border_padding=None, legend_shadow=False,
                           legend_rounded_corners=False, render_axes=False,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           axes_x_limits=None, axes_y_limits=None,
                           axes_x_ticks=None, axes_y_ticks=None,
                           figure_size=(7, 7)):
        """
        Visualize the landmarks. This method will appear on the `TriMesh` as
        ``view_landmarks``.

        Parameters
        ----------
        group : `str` or``None`` optional
            The landmark group to be visualized. If ``None`` and there are more
            than one landmark groups, an error is raised.
        with_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True`` the PointCloud will be viewed as if it is in the image
            coordinate system.
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : ``{-, --, -., :}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_lines_lms : `bool`, optional
            If ``True``, the edges of the landmarks will be rendered.
        line_colour_lms : See Below, optional
            The colour of the lines of the landmarks.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style_lms : ``{-, --, -., :}``, optional
            The style of the lines of the landmarks.
        line_width_lms : `float`, optional
            The width of the lines of the landmarks.
        render_markers : `bool`, optional
            If ``True``, the markers of the landmarks will be rendered.
        marker_style : See Below, optional
            The style of the markers of the landmarks. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers of the landmarks in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers of the landmarks.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers of the landmarks.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The width of the markers' edge of the landmarks.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : ``{center, right, left}``, optional
            The horizontal alignment of the numbers' texts.
        numbers_vertical_align : ``{center, top, bottom, baseline}``, optional
            The vertical alignment of the numbers' texts.
        numbers_font_name : See Below, optional
            The font of the numbers. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : See Below, optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See below, optional
            The font of the legend. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : ``{normal, italic, oblique}``, optional
            The font style of the legend.
        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : See Below, optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold,demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the PointCloud as a percentage of the PointCloud's
            width. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the PointCloud as a percentage of the PointCloud's
            height. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        ValueError
            If the landmark manager doesn't contain the provided group label.
        """
        if not self.has_landmarks:
            raise ValueError('PointGraph does not have landmarks attached, '
                             'unable to view landmarks.')
        self_view = self.view(figure_id=figure_id, new_figure=new_figure,
                              image_view=image_view, figure_size=figure_size,
                              render_markers=render_markers,
                              marker_style=marker_style,
                              marker_size=marker_size,
                              marker_face_colour=marker_face_colour,
                              marker_edge_colour=marker_edge_colour,
                              marker_edge_width=marker_edge_width,
                              render_lines=render_lines,
                              line_colour=line_colour, line_style=line_style,
                              line_width=line_width)
        # correct group label in legend
        if group is None:
            group = self.landmarks.group_labels[0]
        landmark_view = self.landmarks[group].view(
            with_labels=with_labels, without_labels=without_labels,
            figure_id=self_view.figure_id, new_figure=False, group=group,
            image_view=image_view, render_lines=render_lines_lms,
            line_colour=line_colour_lms, line_style=line_style_lms,
            line_width=line_width_lms, render_markers=render_markers_lms,
            marker_style=marker_style_lms, marker_size=marker_size_lms,
            marker_face_colour=marker_face_colour_lms,
            marker_edge_colour=marker_edge_colour_lms,
            marker_edge_width=marker_edge_width_lms,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour,
            render_legend=render_legend, legend_title=legend_title,
            legend_font_name=legend_font_name,
            legend_font_style=legend_font_style,
            legend_font_size=legend_font_size,
            legend_font_weight=legend_font_weight,
            legend_marker_scale=legend_marker_scale,
            legend_location=legend_location,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            legend_border_axes_pad=legend_border_axes_pad,
            legend_n_columns=legend_n_columns,
            legend_horizontal_spacing=legend_horizontal_spacing,
            legend_vertical_spacing=legend_vertical_spacing,
            legend_border=legend_border,
            legend_border_padding=legend_border_padding,
            legend_shadow=legend_shadow,
            legend_rounded_corners=legend_rounded_corners,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

        return landmark_view

    def _view_3d(self, figure_id=None, new_figure=True, mesh_type='wireframe',
                 line_width=2, colour='r', marker_style='sphere',
                 marker_size=None, marker_resolution=8, normals=None,
                 normals_colour='k', normals_line_width=2,
                 normals_marker_style='2darrow', normals_marker_resolution=8,
                 normals_marker_size=None, step=None, alpha=1.0):
        r"""
        Visualization of the TriMesh in 3D.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        mesh_type : `str`, optional
            The representation type to be used for the mesh.
            Example options ::

                {surface, wireframe, points, mesh, fancymesh}

        line_width : `float`, optional
            The width of the lines, if there are any.
        colour : See Below, optional
            The colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_style : `str`, optional
            The style of the markers.
            Example options ::

                {2darrow, 2dcircle, 2dcross, 2ddash, 2ddiamond, 2dhooked_arrow,
                 2dsquare, 2dthick_arrow, 2dthick_cross, 2dtriangle, 2dvertex,
                 arrow, axes, cone, cube, cylinder, point, sphere}

        marker_size : `float` or ``None``, optional
            The size of the markers. This size can be seen as a scale factor
            applied to the size markers, which is by default calculated from
            the inter-marker spacing. If ``None``, then an optimal marker size
            value will be set automatically. It only applies for the
            'fancymesh'.
        marker_resolution : `int`, optional
            The resolution of the markers. For spheres, for instance, this is
            the number of divisions along theta and phi. It only applies for
            the 'fancymesh'.
        normals : ``(n_points, 3)`` `ndarray` or ``None``, optional
            If ``None``, then the normals will not be rendered. If `ndarray`,
            then the provided normals will be rendered as well. Note that a
            normal must be provided for each point in the TriMesh.
        normals_colour : See Below, optional
            The colour of the normals.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        normals_line_width : `float`, optional
            The width of the lines of the normals. It only applies if `normals`
            is not ``None``.
        normals_marker_style : `str`, optional
            The style of the markers of the normals. It only applies if `normals`
            is not ``None``.
            Example options ::

                {2darrow, 2dcircle, 2dcross, 2ddash, 2ddiamond, 2dhooked_arrow,
                 2dsquare, 2dthick_arrow, 2dthick_cross, 2dtriangle, 2dvertex,
                 arrow, axes, cone, cube, cylinder, point, sphere}

        normals_marker_resolution : `int`, optional
            The resolution of the markers of the normals. For spheres, for
            instance, this is the number of divisions along theta and phi. It
            only applies if `normals` is not ``None``.
        normals_marker_size : `float` or ``None``, optional
            The size of the markers. This size can be seen as a scale factor
            applied to the size markers, which is by default calculated from
            the inter-marker spacing. If ``None``, then an optimal marker size
            value will be set automatically. It only applies if `normals` is not
            ``None``.
        step : `int` or ``None``, optional
            If `int`, then one every `step` markers will be rendered.
            If ``None``, then all vertexes will be rendered. It only applies for
            the 'fancymesh' and if `normals` is not ``None``.
        alpha : `float`, optional
            Defines the transparency (opacity) of the object.

        Returns
        -------
        renderer : `menpo3d.visualize.TriMeshViewer3D`
            The Menpo3D rendering object.
        """
        try:
            from menpo3d.visualize import TriMeshViewer3d
            renderer = TriMeshViewer3d(figure_id, new_figure, self.points,
                                       self.trilist)
            renderer.render(
                mesh_type=mesh_type, line_width=line_width, colour=colour,
                marker_style=marker_style, marker_size=marker_size,
                marker_resolution=marker_resolution, normals=normals,
                normals_colour=normals_colour,
                normals_line_width=normals_line_width,
                normals_marker_style=normals_marker_style,
                normals_marker_resolution=normals_marker_resolution,
                normals_marker_size=normals_marker_size, step=step, alpha=alpha)
            return renderer
        except ImportError as e:
            from menpo.visualize import Menpo3dMissingError
            raise Menpo3dMissingError(e)
