import numpy as np
from pybug.shape import TriMesh
from pybug.transform import AffineTransform, Transform


class PiecewiseAffineTransform(Transform):

    def __init__(self, source, target, trilist):
        self.source = TriMesh(source, trilist)
        self.target = TriMesh(target, trilist)
        self.transforms = self._produce_affine_transforms_per_tri()

    @property
    def n_tris(self):
        return self.source.n_tris

    def _diff_ijk(self, x):
        di = x[:, 0]
        dj = x[:, 1] - x[:, 0]
        dk = x[:, 2] - x[:, 0]
        return di.T, dj.T, dk.T

    def _produce_affine_transforms_per_tri(self):
        s = self.source.points[self.source.trilist]
        t = self.target.points[self.target.trilist]
        # svk
        # ^^^
        # ||\- the k'th point
        # ||
        # |vector between end (j or k) and i
        # source [target]
        # if v is absent, it is the position of the ijk point.
        # (not a _vector_ between points)
        svi, svj, svk = self._diff_ijk(s)
        tvi, tvj, tvk = self._diff_ijk(t)
        si, sj, sk = s[:, 0].T, s[:, 1].T, s[:, 2].T

        d = ((svj[0] - svi[0]) * (svk[1] - svi[1]) -
             (svj[1] - svi[1]) * (svk[0] - svi[0]))

        c_x = (svk[1] * tvj - svj[1] * tvk) / d
        c_y = (svj[0] * tvk - svk[0] * tvj) / d
        c_t = tvi + (tvj * (si[1] * svk[0] - si[0] * svk[1]) +
                     tvk * (si[0] * svj[1] - si[1] * svj[0])) / d
        ht = np.repeat(np.eye(3)[..., None], s.shape[0], axis=2)
        ht[:2, 0] = c_x
        ht[:2, 1] = c_y
        ht[:2, 2] = c_t
        transforms = []
        for i in range(self.source.n_tris):
            transforms.append(AffineTransform(ht[..., i]))
        return transforms