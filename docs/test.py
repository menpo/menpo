from menpo.transform.piecewiseaffine.base import DiscreteAffinePWA
import numpy as np

from menpo.shape import TriMesh, PointCloud
a = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
              [-0.5, -0.7], [0.8, -0.4], [0.9, -2.1]])
b = np.array([[0,0], [2, 0], [-1, 3], [2, 6],
              [-1.0, -0.01], [1.0, -0.4], [0.8, -1.6]])
tl = np.array([[0,2,1], [1,3,2]])

src = TriMesh(a, tl)
src_points = PointCloud(a)
tgt = PointCloud(b)
slow_pwa = DiscreteAffinePWA(src, tgt)

points_s = PointCloud(np.random.rand(10000).reshape([-1, 2]))
t_points_s = slow_pwa.apply(points_s)
print t_points_s.shape
