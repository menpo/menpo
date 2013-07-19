import numpy as np
from pybug.exceptions import DimensionalityError
from pybug.shape import TriMesh
from pybug.transform import AffineTransform, Transform


class PiecewiseAffineTransform(Transform):

    def __init__(self, source, target, trilist):
        self.source = TriMesh(source, trilist)
        self.target = TriMesh(target, trilist)
        if self.source.n_dims != self.target.n_dims:
            raise DimensionalityError("source and target must have the same "
                                      "dimension")
        if self.source.n_dims != 2:
            raise DimensionalityError("source and target must be 2 "
                                      "dimensional")
        self._produce_affine_transforms_per_tri()

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

    def tri_containment(self, points):
        """
        Finds for each input point whether it is contained in a triangle,
        and if so what triangle index it is in.
        :param points: An [n_points, n_dim] input array
        :return: (points_in_tris, tri_index)
        points_in_tris is an index into points, st
        points[points_in_tris] yields only points that are contained within
        a triangle.
        tri_index is the triangle index for each points[points_in_tris],
        assigning each point in a triangle to the triangle index.
        """
        ip, ij, ik = (points[..., None] - self.s[0]), self.sij, self.sik
        # todo this could be cached if tri_containment is being tested at
        # many points
        dot_jj = np.einsum('dt, dt -> t', ij, ij)
        dot_kk = np.einsum('dt, dt -> t', ik, ik)
        dot_jk = np.einsum('dt, dt -> t', ij, ik)
        dot_pj = np.einsum('vdt, dt -> vt', ip, ij)
        dot_pk = np.einsum('vdt, dt -> vt', ip, ik)

        d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
        u = (dot_jj * dot_pk - dot_jk * dot_pj) * d
        v = (dot_kk * dot_pj - dot_jk * dot_pk) * d

        return np.nonzero(np.logical_and(
            np.logical_and(u >= 0, v >= 0), u + v <= 1))

    def _tri_containment_loop(self, points):
        """
        Performs the same operation as tri_containment but in C style.
        Useful as a reference for how to convert C/Matlab style to numpy
        (especially the use of einsum)
        """
        all_ij, all_ik = self.sij.T, self.sik.T
        all_i = self.s[0].T
        output = np.zeros((points.shape[0], self.n_tris), dtype=np.bool)
        for i, p in enumerate(points):
            for t in range(self.n_tris):
                ip = p - all_i[t]
                ij = all_ij[t]
                ik = all_ik[t]
                dot_jj = np.dot(ij, ij)
                dot_kk = np.dot(ik, ik)
                dot_jk = np.dot(ij, ik)
                dot_pj = np.dot(ip, ij)
                dot_pk = np.dot(ip, ik)

                d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
                u = (dot_jj * dot_pk - dot_jk * dot_pj) * d
                v = (dot_kk * dot_pj - dot_jk * dot_pk) * d
                output[i, t] = (u >= 0 and v >= 0) and u + v <= 1
        return np.nonzero(output)

    def _apply(self, x, **kwargs):
        """
        Applies this transform to a new set of vectors
        :param x: A (n_points, n_dims) ndarray to apply this transform to.

        :return: The transformed version of x
        """
        x_in_tris, tri_index = self.tri_containment(x)
        # build a list of points in each triangle for each triangle
        x_per_tri = [x[x_in_tris[tri_index == i]] for i in xrange(self.n_tris)]
        # zip the transforms and the list to apply to make the transformed x
        x_per_tri_tran = [t.apply(p) for p, t in zip(x_per_tri,
                                                     self.transforms)]
        x_transformed = x.copy()
        # loop through each triangle, indexing into the x_transformed array
        # for points in that triangle and replacing the value of x with x_t
        for i, x_t in enumerate(x_per_tri_tran):
            x_transformed[x_in_tris[tri_index == i]] = x_t
        return x_transformed

    def as_vector(self):
        pass

    def compose(self, a):
        pass

    def from_vector(self, flattened):
        pass

    def inverse(self):
        pass

    def jacobian(self, shape):
        pass
