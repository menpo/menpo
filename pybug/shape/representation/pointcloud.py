import numpy as np
from tvtk.api import tvtk
from tvtk.tools import ivtk
from tvtk.pyface import picker
from mayavi import mlab

class PointCloud(object):

  def __init__(self, coords):
    self.coords = coords

