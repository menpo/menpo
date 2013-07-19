import numpy as np
from pybug.exceptions import DimensionalityError
from pybug.shape import TriMesh
from pybug.transform import AffineTransform, Transform


class PiecewiseAffineTransform(object):

    def __init__(self, source, target, trilist):
        self.source = TriMesh(source, trilist)
        self.target = TriMesh(target, trilist)
        if self.source.n_dims != self.target.n_dims:
            raise DimensionalityError("source and target must have the same "
                                      "dimension")
        if self.source.n_dims != 2:
            raise DimensionalityError("source and target must be 2 "
                                      "dimensional")
        self.transforms = self._produce_affine_transforms_per_tri()

    @property
    def n_tris(self):
        return self.source.n_tris

    def _produce_affine_transforms_per_tri(self):
        # we permute the axes of the indexed point set to have shape
        # [3, n_dims, n_tris] for ease of indexing in.
        s = np.transpose(self.source.points[self.source.trilist],
                         axes=[1, 2, 0])
        t = np.transpose(self.target.points[self.target.trilist],
                         axes=[1, 2, 0])
        # sik
        # ^^^
        # ||\- the k'th point
        # ||
        # |vector between end (j or k) and i
        # source [target]
        # if i is absent, it is the position of the ijk point.
        # (not a _vector_ between points)
        # get vectors ij ik for source and target
        sij, sik = s[1] - s[0], s[2] - s[0]
        tij, tik = t[1] - t[0], t[2] - t[0]

        # source vertex positions
        si, sj, sk = s[0], s[1], s[2]
        ti = t[0]

        d = ((sij[0] - si[0]) * (sik[1] - si[1]) -
             (sij[1] - si[1]) * (sik[0] - si[0]))

        c_x = (sik[1] * tij - sij[1] * tik) / d
        c_y = (sij[0] * tik - sik[0] * tij) / d
        c_t = ti + (tij * (si[1] * sik[0] - si[0] * sik[1]) +
                    tik * (si[0] * sij[1] - si[1] * sij[0])) / d
        ht = np.repeat(np.eye(3)[..., None], self.n_tris, axis=2)
        ht[:2, 0] = c_x
        ht[:2, 1] = c_y
        ht[:2, 2] = c_t
        transforms = []
        for i in range(self.source.n_tris):
            transforms.append(AffineTransform(ht[..., i]))

        # store our state out
        self.transforms = transforms
        self.s, self.t = s, t
        self.sij, self.sik = sij, sik
        self.tij, self.tik = tij, tik
