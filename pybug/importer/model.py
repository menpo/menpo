import sys
import os.path
import Image
from ..mesh.face import Face
import numpy as np
import sys
import os
import commands
import tempfile
import pickle
import re


def import_face(path_to_file, **kwargs):
  """ Smart model importer. Chooses an appropriate importer based on the
  file extension of the model past in. pass keep_importer=True as a kwarg
  if you want the actual importer object attached to the returned face object
  at face.importer.
  """
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

def process_with_meshlabserver(file_path, output_dir=None, script_path=None, 
    output_filetype=None, export_flags=None):
  """ Interface to meshlabserver to perform preprocessing on meshes before 
  import. Returns a path to the result of the meshlabserver call, ready for 
  import as usual.
  Kwargs:
   * script_path: if specified this script will be run on the input mesh.
   * output_dir: if None provided, set to the users tmp directory.
   * output_filetype: the output desired from meshlabserver. If not provided
       the output type will be the same as the input.
   * export_flags: flags passed to the -om parameter. Allows for choosing what
       aspects of the model will be exported (normals, texture coords etc)
  """
  if output_dir == None:
    output_dir = tempfile.gettempdir()
  filename = os.path.split(file_path)[-1]
  if output_filetype != None:
    file_root = os.path.splitext(filename)[0]
    output_filename = file_root + '.' + output_filetype
  else:
    output_filename = filename
  output_path = os.path.join(output_dir, output_filename)
  command = 'meshlabserver -i ' + file_path + ' -o ' + \
            output_path 
  if script_path != None:
    command += ' -s ' + script_path 
  if export_flags != None:
    command +=  ' -om ' + export_flags
  commands.getoutput(command)
  return output_path

class ModelImporter(object):
  def __init__(self,path_to_file):
    self.path_to_file = os.path.abspath(
                      os.path.expanduser(path_to_file))
    self.path_and_filename = os.path.splitext(self.path_to_file)[0]
    # depreciate this once the other parsers are regexp
    self._file_handle = open(self.path_to_file)
    self.lines = self._file_handle.readlines()
    self._file_handle.close()
    with open(self.path_to_file) as f:
      self.text = f.read()
    self.parse_geometry()
    self.import_texture()
    self.import_landmarks()

  def parse_geometry(self):
    raise NotImplimentedException()

  def import_texture(self):
    raise NotImplimentedException()

  def generate_face(self, **kwargs):
      kwargs['texture'] = self.texture
      if self.texture_coords != None or self.texture_coords.size != 0:
        kwargs['texture_coords'] = self.texture_coords
      if self.texture_tri_index != None or self.texture_tri_index.size != 0:
        kwargs['texture_tri_index'] = self.texture_tri_index
      kwargs['landmarks'] = self.landmarks
      kwargs['file_path_no_ext'] = self.path_and_filename
      return Face(self.coords, self.tri_index, **kwargs)

  def import_landmarks(self):
    path_to_lm = self.path_and_filename + '.landmarks'
    try:
      f = open(path_to_lm, 'r')
      print 'found landmarks! Importing them'
      self.landmarks = pickle.load(f)
    except IOError:
      print 'no landmarks found'
      self.landmarks = {}

class OBJImporter(ModelImporter):
  def __init__(self, path_to_file, **kwargs):
    if kwargs.get('clean_up', False):
      print 'clean up of mesh requested'
      path_to_file = self.clean_up_mesh_on_path(path_to_file)
    print 'importing without cleanup'
    ModelImporter.__init__(self, path_to_file)

  def parse_geometry(self):
    re_v = re.compile(u'v ([^\s]+) ([^\s]+) ([^\s]+)')
    re_vn = re.compile(u'vn ([^\s]+) ([^\s]+) ([^\s]+)')
    re_tc = re.compile(u'vt ([^\s]+) ([^\s]+)')
    re_ti = re.compile(u'f (\d+)\/*\d*\/*\d* (\d+)\/*\d*\/*\d* (\d+)\/*\d*\/*\d*')
    re_tcti = re.compile(u'f \d+\/(\d+)\/*\d* \d+\/(\d+)\/*\d* \d+\/(\d+)\/*\d*')
    re_vnti = re.compile(u'f \d+\/\d*\/(\d+) \d+\/\d*\/(\d+) \d+\/\d*\/(\d+)')
    self.coords = np.array(re_v.findall(self.text), dtype=np.float)
    self.normals = np.array(re_vn.findall(self.text), dtype=np.float)
    self.texture_coords = np.array(re_tc.findall(self.text), dtype=np.float)
    self.tri_index = np.array(re_ti.findall(self.text), dtype=np.uint32) - 1
    self.texture_tri_index = np.array(re_tcti.findall(self.text), dtype=np.uint32) - 1
    self.normals_tri_index = np.array(re_vnti.findall(self.text), dtype=np.uint32) - 1

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
