import numpy as np
import os
from PIL import Image
import commands
import tempfile
import re
from . import metadata
from pybug.spatialdata.mesh import TriMesh


def process_with_meshlabserver(file_path, output_dir=None, script_path=None,
                               output_filetype=None, export_flags=None):
    """ Interface to `meshlabserver` to perform prepossessing on meshes before
    import. Returns a path to the result of the meshlabserver call, ready for
    import as usual.
    Kwargs:
     * script_path: if specified this script will be run on the input mesh.
     * output_dir: if None provided, set to the users tmp directory.
     * output_filetype: the output desired from meshlabserver. If not provided
             the output type will be the same as the input.
     * export_flags: flags passed to the -om parameter. Allows for choosing
             what aspects of the model will be exported (normals,
             texture coords etc)
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    filename = os.path.split(file_path)[-1]
    if output_filetype is not None:
        file_root = os.path.splitext(filename)[0]
        output_filename = file_root + '.' + output_filetype
    else:
        output_filename = filename
    output_path = os.path.join(output_dir, output_filename)
    command = ('meshlabserver -i ' + file_path + ' -o ' +
               output_path)
    if script_path is not None:
        command += ' -s ' + script_path
    if export_flags is not None:
        command += ' -om ' + export_flags
    commands.getoutput(command)
    return output_path


class MeshImporter(object):
    """Base class for importing 3D meshes
    """

    def __init__(self, filepath):
        self.filepath = os.path.abspath(os.path.expanduser(filepath))
        self.path_and_filename = os.path.splitext(self.filepath)[0]
        # depreciate this once the other parsers are regexp
        with open(self.filepath) as f:
            self.lines = f.readlines()
            # text is the entire file in one string (useful for regexp)
        with open(self.filepath) as f:
            self.text = f.read()
        self.parse_geometry()
        self.import_texture()
        self.import_landmarks()

    def parse_geometry(self):
        raise NotImplementedError

    def import_texture(self):
        raise NotImplementedError

    def import_landmarks(self):
        try:
            self.landmarks = metadata.json_pybug_landmarks(
                self.path_and_filename)
        except metadata.MissingLandmarksError:
            self.landmarks = None

    def build(self, **kwargs):
        mesh = TriMesh(self.points, self.trilist)
        if self.texture is not None:
            mesh.attach_texture(self.texture, self.tcoords,
                                tcoords_trilist=self.tcoords_trilist)
        if self.landmarks is not None:
            mesh.landmarks.add_reference_landmarks(self.landmarks)
        mesh.legacy = {'path_and_filename': self.path_and_filename}
        return mesh


class OBJImporter(MeshImporter):
    def __init__(self, filepath):
        MeshImporter.__init__(self, filepath)

    def parse_geometry(self):
        #v 1.345 2134.234 1e015
        re_v = re.compile(u'v ([^\s]+) ([^\s]+) ([^\s]+)')
        #vn 1.345 2134.234 1e015
        re_vn = re.compile(u'vn ([^\s]+) ([^\s]+) ([^\s]+)')
        #vt 0.0025 0.502
        re_tc = re.compile(u'vt ([^\s]+) ([^\s]+)')
        # now we just grab the three possible values that can be given
        # to each face grouping.
        re_ti = re.compile(
            u'f (\d+)/*\d*/*\d* (\d+)/*\d*/*\d* (\d+)/*\d*/*\d*')
        re_tcti = re.compile(
            u'f \d+/(\d+)/*\d* \d+/(\d+)/*\d* \d+/(\d+)/*\d*')
        re_vnti = re.compile(
            u'f \d+/\d*/(\d+) \d+/\d*/(\d+) \d+/\d*/(\d+)')
        self.points = np.array(re_v.findall(self.text), dtype=np.float)
        self.normals = np.array(re_vn.findall(self.text), dtype=np.float)
        self.tcoords = np.array(re_tc.findall(self.text), dtype=np.float)
        self.trilist = np.array(re_ti.findall(self.text), dtype=np.uint32) - 1
        self.tcoords_trilist = np.array(
            re_tcti.findall(self.text), dtype=np.uint32) - 1
        self.normals_trilist = np.array(
            re_vnti.findall(self.text), dtype=np.uint32) - 1

    def import_texture(self):
        # TODO: make this more intelligent in locating the texture
        # (i.e. from the materials file, this can be second guess)
        pathToJpg = os.path.splitext(self.filepath)[0] + '.jpg'
        print pathToJpg
        try:
            Image.open(pathToJpg)
            self.texture = Image.open(pathToJpg)
        except IOError:
            print 'Warning, no texture found'
            if self.tcoords:
                raise Exception(
                    'why do we have texture coords but no texture?')
            else:
                print '(there are no texture coordinates anyway so this is' \
                      ' expected)'
                self.texture = None


class WRLImporter(MeshImporter):
    """ WARNING - this class may need to be restructured to work correctly
    (see OBJImporter for an exemplary MeshImporter subclass)
    """

    def __init__(self, filepath):
        MeshImporter.__init__(self, filepath)

    def parse_geometry(self):
        self._sectionEnds = [i for i, line in enumerate(self.lines)
                             if ']' in line]
        self.points = self._getFloatDataForString(' Coordinate')
        self.tcoords = self._getFloatDataForString('TextureCoordinate')
        tcoords_trilist = self._getFloatDataForString('texCoordIndex',
                                                      separator=', ', cast=int)
        self.tcoords_trilist = [x[:-1] for x in tcoords_trilist]
        self.trilist = self.tcoords_trilist
        self.normalsIndex = None
        self.normals = None

    def _getFloatDataForString(self, string, **kwargs):
        sep = kwargs.get('separator', ' ')
        cast = kwargs.get('cast', float)
        start = self._findIndexOfFirstInstanceOfString(string)
        end = self._findNextSectionEnd(start)
        floatLines = self.lines[start + 1:end]
        return [[cast(x) for x in line[5:-3].split(sep)] for line in
                floatLines]

    def _findIndexOfFirstInstanceOfString(self, string):
        return [i for i, line in enumerate(self.lines) if string in line][0]

    def _findNextSectionEnd(self, beginningIndex):
        return [i for i in self._sectionEnds if i > beginningIndex][0]

    def import_texture(self):
        imageIndex = self._findIndexOfFirstInstanceOfString('ImageTexture') + 1
        self.imageName = self.lines[imageIndex].split('"')[1]
        pathToTexture = os.path.dirname(self.filepath) + '/' + self.imageName
        self.texture = Image.open(pathToTexture)


class OFFImporter(MeshImporter):
    """ WARNING - this class may need to be restructured to work correctly
    (see OBJImporter for an exemplary MeshImporter subclass)
    """

    def __init__(self, filepath):
        MeshImporter.__init__(self, filepath)
        #.off files only have geometry info - all other fields None
        self.tcoords = None
        self.normals = None
        self.normalsIndex = None
        self.tcoords_trilist = None
        self.texture = None

    def parse_geometry(self):
        lines = [l.rstrip() for l in self.lines]
        self.n_points = int(lines[1].split(' ')[0])
        offset = 2
        while lines[offset] == '':
            offset += 1
        x = self.n_points + offset
        coord_lines = lines[offset:x]
        coord_index_lines = lines[x:]
        self.points = [[float(x) for x in l.split(' ')] for l in coord_lines]
        self.trilist = [[int(x) for x in l.split(' ')[2:]] for l in
                        coord_index_lines if l != '']

    def import_texture(self):
        pass
