import sys
import os.path
import Image
from ..mesh.face import Face
import numpy as np
import sys
import os
import commands
import tempfile

def import_face(path_to_file, **kwargs):
  ext = os.path.splitext(path_to_file)[-1]
  if ext == '.off':
    importer = OFFImporter(path_to_file, **kwargs)
  elif ext == '.wrl':
    importer = WRLImporter(path_to_file, **kwargs)
  elif ext == '.obj':
    importer = OBJImporter(path_to_file, **kwargs)
  else:
    raise Exception("I don't understand the file type " + `ext`)
    return None
  face = importer.generate_face()
  if kwargs.get('keep_importer', False):
    print 'attaching the importer at face.importer'
    face.importer = importer
  return face

class ModelImporter(object):
  def __init__(self,path_to_file):
    self.path_to_file = os.path.abspath(
                      os.path.expanduser(path_to_file))
    self._file_handle = open(self.path_to_file)
    self.lines = self._file_handle.readlines()
    self._file_handle.close()
    self.parse_geometry()
    self.import_texture()

  def parse_geometry(self):
    raise NotImplimentedException()

  def import_texture(self):
    raise NotImplimentedException()

  def generate_face(self, **kwargs):
      coords = np.array(self.coords)
      tri_index = np.array(self.tri_index, dtype=np.uint32)
      kwargs['texture'] = self.texture
      if self.texture_coords is not None:
        kwargs['texture_coords'] = np.array(self.texture_coords)
      if self.texture_tri_index is not None:
        kwargs['texture_tri_index'] = np.array(self.texture_tri_index, dtype=np.uint32)
      return Face(coords, tri_index, **kwargs)

class OBJImporter(ModelImporter):
  def __init__(self, path_to_file, **kwargs):
    if kwargs.get('clean_up', False):
      print 'clean up of mesh requested'
      path_to_file = self.clean_up_mesh_on_path(path_to_file)
    print 'importing without cleanup'
    ModelImporter.__init__(self, path_to_file)

  def parse_geometry(self):
    coords_str = self._extract_data_type('v')
    texture_coords_str = self._extract_data_type('vt')
    normals_str = self._extract_data_type('vn')
    index_str = self._extract_data_type('f')
    self.coords = self._strings_to_floats(coords_str)
    self.texture_coords = self._strings_to_floats(texture_coords_str)
    self.normals = self._strings_to_floats(normals_str)
    self.tri_index, self.normalsIndex, self.texture_tri_index = [],[],[]
    for indexLine in index_str:
      cI,tI,nI = [],[],[]
      for index_str in indexLine.split(' '):
        #cIn,nIn,tIn =  [int(x) for x in index_str.split('/')]
        coord_normal_texture_i =  [int(x) for x in index_str.split('/')]
        # take 1 off as we expect indexing to be 0 based
        cI.append(coord_normal_texture_i[0]-1)
        if len(coord_normal_texture_i) > 1:
          # there is texture data as well
          tI.append(coord_normal_texture_i[1]-1)
        if len(coord_normal_texture_i) > 2:
          # there is normal data as well
          nI.append(coord_normal_texture_i[2]-1)
      self.tri_index.append(cI)
      self.normalsIndex.append(nI)
      self.texture_tri_index.append(tI)

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

  def import_texture(self):
    # TODO: make this more intelligent in locating the texture
    # (i.e. from the materials file, this can be second guess)
    pathToJpg = os.path.splitext(self.path_to_file)[0] + '.jpg'
    print pathToJpg
    try:
      Image.open(pathToJpg)
      self.texture = Image.open(pathToJpg)
    except IOError:
      print 'Warning, no texture found'
      if self.texture_coords != []:
        raise Exception('why do we have texture coords but no texture?')
      else:
        print '(there are no texture coordinates anyway so this is expected)'
        self.texture = None

  def _extract_data_type(self,signiture):
    header_length = len(signiture) + 1
    return [line[header_length:-1] for line in self.lines 
                                    if line.startswith(signiture + ' ')]

  def _strings_to_floats(self, lines):
    return [[float(x) for x in line.split(' ')] for line in lines]


class WRLImporter(ModelImporter):

  def __init__(self,path_to_file):
    ModelImporter.__init__(self,path_to_file)

  def parse_geometry(self):
    self._sectionEnds  = [i for i,line in enumerate(self.lines) 
                              if ']' in line]
    self.coords        = self._getFloatDataForString(' Coordinate')
    self.texture_coords = self._getFloatDataForString('TextureCoordinate')
    texture_tri_index = self._getFloatDataForString(
                                  'texCoordIndex',seperator=', ',cast=int)
    self.texture_tri_index = [x[:-1] for x in texture_tri_index]
    self.tri_index  = self.texture_tri_index
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

  def import_texture(self):
    imageIndex = self._findIndexOfFirstInstanceOfString('ImageTexture') + 1
    self.imageName = self.lines[imageIndex].split('"')[1]
    pathToTexture = os.path.dirname(self.path_to_file) + '/' + self.imageName 
    self.texture = Image.open(pathToTexture)


class OFFImporter(ModelImporter):

  def __init__(self,path_to_file):
    ModelImporter.__init__(self,path_to_file)
    #.off files only have geometry info - all other fields None 
    self.texture_coords      = None
    self.normals            = None
    self.normalsIndex       = None
    self.texture_tri_index = None
    self.texture            = None

  def parse_geometry(self):
    lines = [l.rstrip() for l in self.lines]
    self.n_coords = int(lines[1].split(' ')[0])
    offset = 2
    while lines[offset] == '':
      offset += 1
    x = self.n_coords + offset
    coord_lines = lines[offset:x]
    coord_index_lines = lines[x:]
    self.coords = [[float(x) for x in l.split(' ')] for l in coord_lines]
    self.tri_index = [[int(x) for x in l.split(' ')[2:]] for l in coord_index_lines if l != '']

  def import_texture(self):
    pass
