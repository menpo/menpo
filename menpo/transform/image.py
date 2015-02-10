from __future__ import unicode_literals
import numpy as np

from menpo.transform import Transform, Homogeneous, Scale


class TcoordsToPointCloud(Transform):
    r"""
    Converts unitary tcoords into a PointCloud that is suitable
    for directly indexing into the pixels of the texture (e.g. for manual
    mapping operations). The resulting tcoords behave just like image landmarks
    do

    ::

        >>> texture = texturedtrimesh.texture
        >>> t = TcoordsToPointCloud(texture.shape)
        >>> tc_ps = t.apply(texturedtrimesh.tcoords)
        >>> pixel_values_at_tcs = texture[tc_ps[: ,0], tc_ps[:, 1]]

    The operations that are performed are:

    - Flipping the origin from bottom-left to top-left
    - Scaling the tcoords by the image shape (denormalising them)
    - Permuting the axis so that

    Returns
    -------
    tcoords_scaled : :class:`menpo.shape.PointCloud`
        A copy of the tcoords that behave like Image landmarks
    """
    def __init__(self, image_shape):
        # flip axis 0 and axis 1 so indexing is as expected
        flip_xy = Homogeneous(np.array([[0, 1, 0],
                                        [1, 0, 0],
                                        [0, 0, 1]]))
        # scale to get the units correct
        scale = Scale(image_shape)
        self.flip_and_scale = flip_xy.compose_before(scale)

    def _apply(self, tcoords, **kwargs):
        # flip the 'y' st 1 -> 0 and 0 -> 1, moving the axis to upper left
        tcoords[:, 1] = 1 - tcoords[:, 1]
        return self.flip_and_scale.apply(tcoords)


class PointCloudToTcoords(Transform):

    def __init__(self, image_shape):
        # flip axis 0 and axis 1 so indexing is as expected
        flip_xy = Homogeneous(np.array([[0, 1, 0],
                                        [1, 0, 0],
                                        [0, 0, 1]]))
        # scale to get the units correct
        scale = Scale(image_shape).pseudoinverse
        self.flip_and_scale = scale.compose_before(flip_xy)

    def _apply(self, x, **kwargs):
        tcoords = self.flip_and_scale.apply(x)
        # flip the 'y' st 1 -> 0 and 0 -> 1, moving the axis to upper left
        tcoords[:, 1] = 1 - tcoords[:, 1]
        return tcoords
