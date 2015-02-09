from .pointcloud import PointCloud
import numpy as np


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
    return PointCloud(np.mean([pc.points for pc in pointclouds], axis=0))
