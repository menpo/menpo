from .pointcloud import PointCloud


def mean_pointcloud(pointclouds):
    r"""
    Parameters
    ----------
    pointclouds: Iterable of PointCloud

    Returns
    -------
    PointCloud: The mean of the pointclouds
    """
    return PointCloud(sum(pointclouds) / len(pointclouds))

