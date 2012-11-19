import sys
import os.path
import Image
from face import Face

class Importer(object):
  def __init__(self,pathToFile):
    self.pathToFile = os.path.abspath(pathToFile)
    self._fileHandle = open(self.pathToFile)
    self.lines = self._fileHandle.readlines()
    self._fileHandle.close()
    self.importGeometry()
    self.importTexture()

  def importGeometry(self):
    raise NotImplimentedException()

  def importTexture(self):
    raise NotImplimentedException()

  def generateFace(self, **kwargs):
      kwargs['coords']             = self.coords
      kwargs['textureCoords']      = self.textureCoords
      kwargs['normals']            = self.normals
      kwargs['coordsIndex']        = self.coordsIndex
      kwargs['normalsIndex']       = self.normalsIndex
      kwargs['textureCoordsIndex'] = self.textureCoordsIndex
      kwargs['texture']            = self.texture
      return Face(**kwargs)


class OBJImporter(Importer):
  def __init__(self,pathToFile):
    Importer.__init__(self,pathToFile)

  def importGeometry(self):
    coordsStr          = self._extractDataType('v')
    textureCoordsStr   = self._extractDataType('vt')
    normalsStr         = self._extractDataType('vn')
    indexStr           = self._extractDataType('f')
    self.coords        = self._stringsToFloats(coordsStr)
    self.textureCoords = self._stringsToFloats(textureCoordsStr)
    self.normals       = self._stringsToFloats(normalsStr)
    self.coordsIndex, self.normalsIndex, self.textureCoordsIndex = [],[],[]
    for indexLine in indexStr:
      cI,tI,nI = [],[],[]
      for indexStr in indexLine.split(' '):
        cIn,nIn,tIn =  [int(x) for x in indexStr.split('/')]
        # take 1 off as we expect indexing to be 0 based
        cI.append(cIn-1)
        tI.append(tIn-1)
        nI.append(nIn-1)
      self.coordsIndex.append(cI)
      self.normalsIndex.append(nI)
      self.textureCoordsIndex.append(tI)

  def importTexture(self):
    pathToJpg = os.path.splitext(self.pathToFile)[0] + '.jpg'
    self.texture = Image.open(pathToJpg)

  def _extractDataType(self,signiture):
    headerLength = len(signiture) + 1
    return [line[headerLength:-1] for line in self.lines 
                                    if line.startswith(signiture + ' ')]
  def _stringsToFloats(self, lines):
    return [[float(x) for x in line.split(' ')] for line in lines]


class WRLImporter(Importer):

  def __init__(self,pathToFile):
    Importer.__init__(self,pathToFile)

  def importGeometry(self):
    self._sectionEnds  = [i for i,line in enumerate(self.lines) 
                              if ']' in line]
    self.coords        = self._getFloatDataForString(' Coordinate')
    self.textureCoords = self._getFloatDataForString('TextureCoordinate')
    textureCoordsIndex = self._getFloatDataForString(
                                  'texCoordIndex',seperator=', ',cast=int)
    self.textureCoordsIndex = [x[:-1] for x in textureCoordsIndex]
    self.coordsIndex  = self.textureCoordsIndex
    self.normalsIndex = None 
    self.normals      = None

  def _getFloatDataForString(self, string, **kwargs):
    sep = kwargs.get('seperator',' ')
    cast = kwargs.get('cast', float)
    start = self._findIndexOfFirstInstanceOfString(string)
    end   = self._findNextSectionEnd(start)
    floatLines = self.lines[start+1:end]
    return [[cast(x) for x in line[5:-3].split(sep)] for line in floatLines]

  def _findIndexOfFirstInstanceOfString(self,string):
    return [i for i,line in enumerate(self.lines) 
                          if string in line][0]

  def _findNextSectionEnd(self,beginningIndex):
    return [i for i in self._sectionEnds if i > beginningIndex][0]

  def importTexture(self):
    imageIndex = self._findIndexOfFirstInstanceOfString('ImageTexture') + 1
    self.imageName = self.lines[imageIndex].split('"')[1]
    pathToTexture = os.path.dirname(self.pathToFile) + '/' + self.imageName 
    self.texture = Image.open(pathToTexture)

