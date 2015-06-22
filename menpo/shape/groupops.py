from __future__ import division


def mean_pointcloud(pointclouds):
    r"""
    Compute the mean of a `list` of :map:`PointCloud` or subclass objects.

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
    return pointclouds[0].from_vector(
        sum(pc.points for pc in pointclouds) / len(pointclouds))
