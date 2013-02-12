import sys
import os.path
import Image
from ..mesh.face import Face
import numpy as np
import sys
import os
import commands
import tempfile

def ModelImporterFactory(pathToFile, **kwargs):
  ext = os.path.splitext(pathToFile)[-1]
  if ext == '.off':
    return OFFImporter(pathToFile, **kwargs)
  elif ext == '.wrl':
    return WRLImporter(pathToFile, **kwargs)
  elif ext == '.obj':
    return OBJImporter(pathToFile, **kwargs)
  else:
    raise Exception("I don't understand the file type " + `ext`)

class ModelImporter(object):
  def __init__(self,pathToFile):
    self.pathToFile = os.path.abspath(
                      os.path.expanduser(pathToFile))
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
      coords         = np.array(self.coords)
      tri_index   = np.array(self.coordsIndex,dtype=np.uint32)
      kwargs['texture_coords'] = np.array(self.textureCoords)
      kwargs['texture']        = self.texture
      #kwargs['normals']            = self.normals
      #kwargs['normalsIndex']       = self.normalsIndex
      kwargs['texture_tri_index'] = np.array(self.textureCoordsIndex, dtype=np.uint32)
      return Face(coords, tri_index, **kwargs)

class OBJImporter(ModelImporter):
  def __init__(self, path_to_file, **kwargs):
    clean_up = kwargs.get('clean_up')
    if clean_up:
      print 'clean up of mesh requested'
      path_to_file = self.clean_up_mesh_on_path(path_to_file)
    print 'importing without cleanup'
    ModelImporter.__init__(self, path_to_file)

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
        #cIn,nIn,tIn =  [int(x) for x in indexStr.split('/')]
        coord_normal_texture_i =  [int(x) for x in indexStr.split('/')]
        # take 1 off as we expect indexing to be 0 based
        cI.append(coord_normal_texture_i[0]-1)
        if len(coord_normal_texture_i) > 1:
          # there is texture data as well
          tI.append(coord_normal_texture_i[1]-1)
        if len(coord_normal_texture_i) > 2:
          # there is normal data as well
          nI.append(coord_normal_texture_i[2]-1)
      self.coordsIndex.append(cI)
      self.normalsIndex.append(nI)
      self.textureCoordsIndex.append(tI)

  def clean_up_mesh_on_path(self, path_to_file):
    clean_up_path = os.path.join(os.path.split(os.path.abspath(__file__))[0],'cleanup.mlx')
    tmp_path = tempfile.gettempdir()
    output_file_name = ''
    file_name = os.path.split(path_to_file)[-1]
    output_path = os.path.join(tmp_path, file_name)
    command = 'meshlabserver -i ' + path_to_file + ' -o ' + \
              output_path + ' -s ' + clean_up_path + ' -om wt'
    commands.getoutput(command)
    print 'importing cleaned version of mesh from tmp'
    return output_path

  def importTexture(self):
    pathToJpg = os.path.splitext(self.pathToFile)[0] + '.jpg'
    print pathToJpg
    try:
      Image.open(pathToJpg)
      self.texture = Image.open(pathToJpg)
    except IOError:
      print 'Warning, no texture found'
      if self.textureCoords != []:
        raise Exception('why do we have texture coords but no texture?')
      else:
        print '(there are no texture coordinates anyway so this is expected)'
        self.texture = []

  def _extractDataType(self,signiture):
    headerLength = len(signiture) + 1
    return [line[headerLength:-1] for line in self.lines 
                                    if line.startswith(signiture + ' ')]
  def _stringsToFloats(self, lines):
    return [[float(x) for x in line.split(' ')] for line in lines]


class WRLImporter(ModelImporter):

  def __init__(self,pathToFile):
    ModelImporter.__init__(self,pathToFile)

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


class OFFImporter(ModelImporter):

  def __init__(self,pathToFile):
    ModelImporter.__init__(self,pathToFile)
    #.off files only have geometry info - all other fields None 
    self.textureCoords      = None
    self.normals            = None
    self.normalsIndex       = None
    self.textureCoordsIndex = None
    self.texture            = None

  def importGeometry(self):
    lines = [l.rstrip() for l in self.lines]
    self.n_coords = int(lines[1].split(' ')[0])
    offset = 2
    while lines[offset] == '':
      offset += 1
    x = self.n_coords + offset
    coord_lines = lines[offset:x]
    coord_index_lines = lines[x:]
    self.coords = [[float(x) for x in l.split(' ')] for l in coord_lines]
    self.coordsIndex = [[int(x) for x in l.split(' ')[2:]] for l in coord_index_lines if l != '']

  def importTexture(self):
    pass
