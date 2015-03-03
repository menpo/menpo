from __future__ import division
from .pointcloud import PointCloud


def mean_pointcloud(pointclouds):
    r"""
    Compute the mean of a `list` of :map:`PointCloud` objects.

    Parameters
    ----------
    pointclouds: `list` of :map:`PointCloud`
        List of point cloud objects from which we want to compute the mean.

    Returns
    -------
    mean_pointcloud : :map:`PointCloud`
        The mean point cloud.
    """
    return PointCloud(sum(pc.points for pc in pointclouds) / len(pointclouds))
