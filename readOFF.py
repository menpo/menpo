import os.path
import numpy as np

class OFFImporter(object):
  def __init__(self,pathToFile):
    self._fileHandle = open(pathToFile)
    self.lines = self._fileHandle.readlines()
    self._fileHandle.close()


  def c_c_I(self):
    self.lines = [l[:-2] for l in self.lines]
    self.n_coords = int(self.lines[1].split(' ')[0])
    x = self.n_coords + 2
    coord_lines = self.lines[2:x]
    coord_index_lines = self.lines[x:]
    coords = np.array([[float(x) for x in l.split(' ')] for l in coord_lines])
    coords_index = np.array([[int(x) for x in l.split(' ')[1:]] for l in coord_index_lines],dtype=np.uint32)
    return coords, coords_index

