import os.path
import numpy as np

class OFFImporter(object):
  def __init__(self,pathToFile):
    self._fileHandle = open(pathToFile)
    self.lines = self._fileHandle.readlines()
    self._fileHandle.close()
