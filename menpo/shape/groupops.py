from __future__ import division
from menpo.shape import PointCloud


def mean_pointcloud(pointclouds):
    r"""
    Compute the mean of a `list` of :map:`PointCloud` or subclass objects.
    The list is assumed to be homogeneous i.e all elements of the list are
    assumed to belong to the same point cloud subclass just as all elements
    are also assumed to have the same number of points and represent
    semantically equivalent point clouds.

    Parameters
    ----------
    pointclouds: `list` of :map:`PointCloud` or subclass
        List of point cloud or subclass objects from which we want to compute
        the mean.

    Returns
    -------
    mean_pointcloud : :map:`PointCloud` or subclass
        The mean point cloud or subclass.
    """
    # make a temporary PointCloud (with copy=False for low overhead)
    tmp_pc = PointCloud(sum(pc.points for pc in pointclouds) /
                        len(pointclouds), copy=False)
    # use the type of the first element in the list to rebuild from the vector
    return pointclouds[0].from_vector(tmp_pc.as_vector())
