# This has to go above the default importers to prevent cyclical importing
from pybug.exceptions import DimensionalityError


class Viewer(object):
    """
    Abstract class for performing visualizations. Framework specific
    implementations of these classes are made in order to separate
    implementation cleanly from the rest of the code.
    """

    def __init__(self):
        self.currentfigure = None
        self.currentscene = None

    def view(self, **kwargs):
        r"""
        View the object.

        Parameters
        ----------
        onviewer : figure object, optional
            The figure object to draw this view call on to.
        kwargs : dict
            Passed through to specific viewer.

        Returns
        -------
        viewer : :class:`Viewer`
            Pointer to ``self``.
        """
        figure = kwargs.get('onviewer', None)
        if figure is None:
            figure = self.newfigure()
        else:
            figure = figure.currentfigure
        return self._viewonfigure(figure, **kwargs)

from pybug.visualize.viewmayavi import MayaviPointCloudViewer3d, \
    MayaviTriMeshViewer3d, MayaviTexturedTriMeshViewer3d, \
    MayaviLandmarkViewer3d, MayaviVectorViewer3d
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
VectorViewer3d = MayaviVectorViewer3d


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
    def __init__(self, label, landmark_dict, parent_shape):
        if landmark_dict is None:
            landmark_dict = {}
        self.landmark_dict = landmark_dict
        self.label = label
        self.shape = parent_shape

    def view(self, **kwargs):
        r"""
        Select the correct type of landmark viewer for the given parent shape.

        Parameters
        ----------
        kwargs : dict
            Passed through to landmark viewer.

        Returns
        -------
        viewer : :class:`Viewer`
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
                        self.label, self.landmark_dict).view(**kwargs)
                else:
                    return LandmarkViewer2d(
                        self.label, self.landmark_dict).view(**kwargs)
            elif item.n_dims == 3:
                return LandmarkViewer3d(self.label, self.landmark_dict).view(
                    **kwargs)
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
    def __init__(self, points):
        self.points = points

    def view(self, **kwargs):
        r"""
        Select the correct type of pointcloud viewer for the given
        pointcloud dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to pointcloud viewer.

        Returns
        -------
        viewer : :class:`Viewer`
                Pointer to ``self``.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.points.shape[1] == 2:
            return PointCloudViewer2d(self.points).view(**kwargs)
        elif self.points.shape[1] == 3:
            return PointCloudViewer3d(self.points).view(**kwargs)
        else:
            raise DimensionalityError("Only 2D and 3D pointclouds are "
                                      "currently supported")


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
    def __init__(self, points, trilist):
        self.points = points
        self.trilist = trilist

    def view(self, **kwargs):
        r"""
        Select the correct type of trimesh viewer for the given
        trimesh dimensionality.

        Parameters
        ----------
        kwargs : dict
            Passed through to trimesh viewer.

        Returns
        -------
        viewer : :class:`Viewer`
                Pointer to ``self``.

        Raises
        ------
        DimensionalityError
            Only 2D and 3D viewers are supported.
        """
        if self.points.shape[1] == 2:
            return TriMeshViewer2d(self.points, self.trilist).view(**kwargs)

        elif self.points.shape[1] == 3:
            return TriMeshViewer3d(self.points, self.trilist).view(**kwargs)
        else:
            raise DimensionalityError("Only 2D and 3D TriMeshes are "
                                      "currently supported")
