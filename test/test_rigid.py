from pybug.alignment.rigid import Procrustes
import numpy as np
import unittest
import doctest

class ProcrustesTest(unittest.TestCase):
  """Unit tests for Procrustes"""

  def test_invarient_target_source(self):
    source = np.array([[ 0, 0],[ 0, 1],[ 1, 1],[ 1, 0]],dtype=np.float64).T
    proc = Procrustes([source],target=source)
    proc.generalProcrusesAlignment()
    self.assertTrue(np.sum((proc.sources - proc.target)**2) < 0.000000001)

  def test_scale(self):
    source = np.array([[-1,-1],[-1, 1],[ 1, 1],[ 1,-1]],dtype=np.float64).T
    target = source * 2
    proc = Procrustes([source],target=target)
    proc.generalProcrusesAlignment()
    assert(np.sum((proc.translation_vectors[0]     - np.zeros(2))**2) < 0.00000001)
    assert(np.sum((proc.scale_rotation_matrices[0] - 2*np.eye(2))**2) < 0.00000001)

  def test_translate(self):
    source = np.array([[ 0, 0],[ 0, 1],[ 1, 1],[ 1, 0]],dtype=np.float64).T
    target = source - 1
    proc = Procrustes([source],target=target)
    proc.generalProcrusesAlignment()
    assert(np.sum((proc.translation_vectors[0]     + np.ones(2))**2) < 0.00000001)
    assert(np.sum((proc.scale_rotation_matrices[0] -  np.eye(2))**2) < 0.00000001)




if __name__ == '__main__':
  unittest.main()


