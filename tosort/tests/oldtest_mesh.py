import numpy as np
import unittest
from menpo.mesh.face import Face
from scipy.sparse import linalg 

class LaplacianTest(unittest.TestCase):
  """Unit tests for Laplacian Operator"""

  def setUp(self):
    """Construct a square mesh and it's cotangent Laplacian/Area matrix
       
       Sparse representations are at self.A and self.L_c, dense versions
       for matrix comparision are at self.A_d. self.L_c_d
    """
    c  = np.array([[ 0., 0., 0.],
                   [ 1., 0., 0.],
                   [ 0., 1., 0.],
                   [-1., 0., 0.],
                   [ 0.,-1., 0.]])
    cI = np.array([[0, 4, 1],
                   [3, 0, 2],
                   [1, 2, 0],
                   [0, 3, 4]], dtype=np.uint32)
    self.square_mesh = Face(coords=c,coordsIndex=cI)
    self.L_c, self.A = self.square_mesh.laplacian_operator()
    self.L_c_d = self.L_c.todense()
    self.A_d   =   self.A.todense()


  def test_cotangent_flat_square_laplacian(self):
    L_c_analytic = np.array([[ 4., -1., -1., -1., -1.],
                             [-1.,  1.,  0.,  0.,  0.],
                             [-1.,  0.,  1.,  0.,  0.],
                             [-1.,  0.,  0.,  1.,  0.],
                             [-1.,  0.,  0.,  0.,  1.]])
    self.assertTrue(matrix_equal(self.L_c_d, L_c_analytic))

  def test_cotangent_flat_square_area(self):
    A_analytic = np.diag([2., 1., 1., 1., 1.])/3.
    self.assertTrue(matrix_equal(self.A_d, A_analytic))

  def test_cotangent_flat_square_euler_step(self):
    u_0 = np.zeros(self.square_mesh.n_coords)
    u_0[0] = 1
    L_c = -1.0*self.L_c
    t = self.square_mesh.mean_edge_length
    u_t = linalg.spsolve(self.A - t*(self.L_c), u_0)
    # heat should spread equally to each corner
    self.assertTrue(np.var(u_t[1:]) < 0.000001)
    # there should be some heat everywhere
    self.assertTrue(np.all(u_t > 0))

  def test_cotangent_flat_square_heat_divergence(self):
    u_0 = np.zeros(self.square_mesh.n_coords)
    u_0[0] = 1
    L_c = -1.0*self.L_c
    t = self.square_mesh.mean_edge_length
    u_t = linalg.spsolve(self.A - t*(self.L_c), u_0)
    grad_u_t = self.square_mesh.gradient(u_t)
    print grad_u_t
    grad_u_t_analytic = np.array([[ 0.15206873, -0.15206873, 0.],
                                  [-0.15206873,  0.15206873, 0.],
                                  [ 0.15206873,  0.15206873, 0.],
                                  [-0.15206873, -0.15206873, 0.]])
    self.assertTrue(matrix_equal(grad_u_t, grad_u_t_analytic))

  def test_flat_square_mean_length(self):
    t = self.square_mesh.mean_edge_length
    self.assertTrue(np.absolute(t - (1 + np.sqrt(2))/2.0) < 0.00000000001)
    

def matrix_equal(A,B):
  """True if A==B for all elements within small error margin, False if not"""
  if A.shape != B.shape:
    return False
  diff = np.absolute(A-B)
  if np.all(diff < 0.00001):
    return True
  else:
    return False

if __name__ == '__main__':
  unittest.main()

