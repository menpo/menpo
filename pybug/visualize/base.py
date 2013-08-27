# This has to go above the default importers to prevent cyclical importing
from pybug.exceptions import DimensionalityError
import abc


class Renderer(object):
    """
    Abstract class for rendering visualizations. Framework specific
    implementations of these classes are made in order to separate
    implementation cleanly from the rest of the code.

    Parameters
    ----------
    figure_id : object
        A figure id. Could be any valid object that identifies
        a figure in a given framework (string, int, etc)
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, figure_id, new_figure):
        if figure_id is not None and new_figure:
            raise ValueError("Conflicting arguments. figure_id cannot be "
                             "specified if the new_figure flag is True")

        self.figure_id = figure_id
        self.new_figure = new_figure
        self.figure = self.get_figure()

    def render(self, **kwargs):
        r"""
        Render the object on the figure given at instantiation.

        Parameters
        ----------
        kwargs : dict
            Passed through to specific rendering engine.

        Returns
        -------
        viewer : :class:`Renderer`
            Pointer to ``self``.
        """
        return self._render(**kwargs)

    @abc.abstractmethod
    def _render(self, **kwargs):
        pass

    @abc.abstractmethod
    def get_figure(self):
        pass


class Viewable(object):
    """
    Abstract interface for objects that can visualize themselves.
    """

    __metaclass__ = abc.ABCMeta

    def view_on(self, figure_id, **kwargs):
        r"""
        View the object on a a specific figure specified by the given id.

        Parameters
        ----------
        figure_id : object
            A unique identifier for a figure.
        kwargs : dict
            Passed through to specific rendering engine.
        """
        return self._view(figure_id=figure_id, **kwargs)

    def view_new(self, **kwargs):
        r"""
        View the object on a new figure.

        Parameters
        ----------
        kwargs : dict
            Passed through to specific rendering engine.
        """
        return self._view(new_figure=True, **kwargs)

    def view(self, **kwargs):
        r"""
        View the object using the default rendering engine figure handling.
        For example, the default behaviour for Matplotlib is that all draw
        commands are applied to the same ``figure`` object.

        Parameters
        ----------
        kwargs : dict
            Passed through to specific rendering engine.
        """
        return self._view(**kwargs)

    @abc.abstractmethod
    def _view(self, figure_id=None, new_figure=False, **kwargs):
        pass

from pybug.visualize.viewmayavi import MayaviPointCloudViewer3d, \
    MayaviTriMeshViewer3d, MayaviTexturedTriMeshViewer3d, \
    MayaviLandmarkViewer3d
from pybug.visualize.viewmatplotlib import MatplotlibImageViewer2d, \
    MatplotlibPointCloudViewer2d, MatplotlibLandmarkViewer2d, \
    MatplotlibLandmarkViewer2dImage, MatplotlibTriMeshViewer2d

# Default importer types
PointCloudViewer2d = MatplotlibPointCloudViewer2d
PointCloudViewer3d = MayaviPointCloudViewer3d
TriMeshViewer2d = MatplotlibTriMeshViewer2d
TriMeshViewer3d = MayaviTriMeshViewer3d
TexturedTriMeshViewer3d = MayaviTexturedTriMeshViewer3d
LandmarkViewer3d = MayaviLandmarkViewer3d
LandmarkViewer2d = MatplotlibLandmarkViewer2d
LandmarkViewer2dImage = MatplotlibLandmarkViewer2dImage
ImageViewer2d = MatplotlibImageViewer2d


class LandmarkViewer(object):
    """
    Base Landmark viewer that abstracts away dimensionality

    Parameters
    ----------
    label : string
        The main label of the landmark set.
    landmark_dict : dict (string, :class:`pybug.shape.pointcloud.PointCloud`)
        The landmark dictionary containing pointclouds.
    parent_shape : :class:`pybug.base.Shape`
        The parent shape that we are drawing the landmarks for.
    """
    def __init__(self, figure_id, new_figure,
                 label, landmark_dict, parent_shape):
        if landmark_dict is None:
            landmark_dict = {}
        self.landmark_dict = landmark_dict
        self.label = label
        self.shape = parent_shape
        self.figure_id = figure_id
        self.new_figure = new_figure

    def render(self, **kwargs):
        r"""
        Select the correct type of landmark viewer for the given parent shape.

        Parameters
        ----------
        kwargs : dict
            Passed through to landmark viewer.

        Returns
        -------
        viewer : :class:`Renderer`
                Pointer to ``self``.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.landmark_dict:
            item = self.landmark_dict.values()[0]
            if item.n_dims == 2:
                from pybug.image import Image
                if type(self.shape) is Image:
                    return LandmarkViewer2dImage(
                        self.figure_id, self.new_figure,
                        self.label, self.landmark_dict).render(**kwargs)
                else:
                    return LandmarkViewer2d(self.figure_id, self.new_figure,
                                            self.label,
                                            self.landmark_dict).render(**kwargs)
            elif item.n_dims == 3:
                return LandmarkViewer3d(self.figure_id, self.new_figure,
                                        self.label,
                                        self.landmark_dict).render(**kwargs)
            else:
                raise DimensionalityError("Only 2D and 3D landmarks are "
                                          "currently supported")


class PointCloudViewer(object):
    r"""
    Base PointCloud viewer that abstracts away dimensionality.

    Parameters
    ----------
    points : (N, D) ndarray
        The points to render.
    """
    def __init__(self, figure_id, new_figure, points):
        self.figure_id = figure_id
        self.new_figure = new_figure
        self.points = points

    def render(self, **kwargs):
        r"""
        Select the correct type of pointcloud viewer for the given
        pointcloud dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to pointcloud viewer.

        Returns
        -------
        viewer : :class:`Renderer`
                Pointer to ``self``.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.points.shape[1] == 2:
            return PointCloudViewer2d(self.figure_id, self.new_figure,
                                      self.points).render(**kwargs)
        elif self.points.shape[1] == 3:
            return PointCloudViewer3d(self.figure_id, self.new_figure,
                                      self.points).render(**kwargs)
        else:
            raise DimensionalityError("Only 2D and 3D pointclouds are "
                                      "currently supported")


class ImageViewer(object):
    r"""
    Base Image viewer that abstracts away dimensionality.

    Parameters
    ----------
    points : (N, D) ndarray
        The points to render.
    """
    def __init__(self, figure_id, new_figure, dimensions, pixels):
        self.figure_id = figure_id
        self.new_figure = new_figure
        self.pixels = pixels
        self.dimensions = dimensions

    def render(self, **kwargs):
        r"""
        Select the correct type of image viewer for the given
        image dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to image viewer.

        Returns
        -------
        viewer : :class:`Renderer`
                Pointer to ``self``.

        Raises
        ------
        DimensionalityError
            Only 2D images are supported.
        """
        if self.dimensions == 2:
            return ImageViewer2d(self.figure_id, self.new_figure,
                                 self.pixels).render(**kwargs)
        else:
            raise DimensionalityError("Only 2D images are currently supported")


class TriMeshViewer(object):
    """
    Base TriMesh viewer that abstracts away dimensionality.

    Parameters
    ----------
    points : (N, D) ndarray
        The points to render.
    trilist : (M, 3) ndarray
        The triangulation for the points.
    """
    def __init__(self, figure_id, new_figure, points, trilist):
        self.figure_id = figure_id
        self.new_figure = new_figure
        self.points = points
        self.trilist = trilist

    def render(self, **kwargs):
        r"""
        Select the correct type of trimesh viewer for the given
        trimesh dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to trimesh viewer.

        Returns
        -------
        viewer : :class:`Renderer`
                Pointer to ``self``.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.points.shape[1] == 2:
            return TriMeshViewer2d(self.figure_id, self.new_figure,
                                   self.points, self.trilist).render(**kwargs)

        elif self.points.shape[1] == 3:
            return TriMeshViewer3d(self.figure_id, self.new_figure,
                                   self.points, self.trilist).render(**kwargs)
        else:
            raise DimensionalityError("Only 2D and 3D TriMeshes are "
                                      "currently supported")
