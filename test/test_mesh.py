import numpy as np
import unittest
from ibugMM.mesh.face     import Face

class LaplacianTest(unittest.TestCase):
  """Unit tests for Laplacian Operator"""

  def test_cotangent_flat_square(self):
    mesh = self.flat_square()
    L_c, A = mesh.laplacian_operator()
    L_c_d  = L_c.todense()
    print L_c_d
    assert np.all(L_c_d == np.array([[ 4., -1., -1., -1., -1.],
                                     [-1.,  1.,  0.,  0.,  0.],
                                     [-1.,  0.,  1.,  0.,  0.],
                                     [-1.,  0.,  0.,  1.,  0.],
                                     [-1.,  0.,  0.,  0.,  1.]]))


  def test_area_flat_square(self):
    mesh = self.flat_square()
    L_c, A = mesh.laplacian_operator()
    areas = A.diagonal()
    assert areas[0] == 2.0/3.0
    assert np.all(areas[1:] == 1.0/3.0)

  def flat_square(self):
    c  = np.array([[ 0., 0., 0.],
                   [ 1., 0., 0.],
                   [ 0., 1., 0.],
                   [-1., 0., 0.],
                   [ 0.,-1., 0.]])
    
    cI = np.array([[0, 4, 1],
                   [3, 0, 2],
                   [1, 2, 0],
                   [0, 3, 4]], dtype=np.uint32)
    return Face(coords=c,coordsIndex=cI)


if __name__ == '__main__':
  unittest.main()


