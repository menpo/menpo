import numpy as np
from . import Alignment
  
class NonRigidAlignment(Alignment):
  """ Abstract class specializing in nonrigid alignments.
  """
  def __init__(self, sources, **kwargs):
    Alignment.__init__(self,sources, **kwargs)


class TPS(NonRigidAlignment):

  def __init__(self, sources, **kwargs):
    NonRigidAlignment.__init__(self,sources,**kwargs)

