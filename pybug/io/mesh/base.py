import commands
import os.path as path
import tempfile
from pybug.io.base import Importer
from pybug.io.mesh.assimp import AIImporter
from pybug.io.exceptions import MeshImportError
from pybug.io.image import ImageImporter
from pybug.shape import TexturedTriMesh, TriMesh
from pyvrml import buildVRML97Parser
import pyvrml.vrml97.basenodes as basenodes
from scipy.spatial import Delaunay
import numpy as np


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
    filename = path.split(file_path)[-1]
    if output_filetype is not None:
        file_root = path.splitext(filename)[0]
        output_filename = file_root + '.' + output_filetype
    else:
        output_filename = filename
    output_path = path.join(output_dir, output_filename)
    command = ('meshlabserver -i ' + file_path + ' -o ' +
               output_path)
    if script_path is not None:
        command += ' -s ' + script_path
    if export_flags is not None:
        command += ' -om ' + export_flags
    commands.getoutput(command)
    return output_path


class MeshImporter(Importer):
    """Base class for importing 3D meshes
    """
    def __init__(self, filepath):
        super(MeshImporter, self).__init__(filepath)
        if self.texture_path is None or not path.exists(self.texture_path):
            self.texture_importer = None
        else:
            self.texture_importer = ImageImporter(self.texture_path)

    @property
    def texture_path(self):
        if self.relative_texture_path is None:
            return None
        else:
            return path.join(self.folder, self.relative_texture_path)

    def import_landmarks(self):
        import pybug.io.metadata as metadata
        try:
            self.landmarks = metadata.json_pybug_landmarks(self.filepath)
        except metadata.MissingLandmarksError:
            self.landmarks = None

    def build(self):
        meshes = []
        for mesh in self.meshes:
            if self.texture_importer is not None:
                meshes.append(TexturedTriMesh(mesh.points, mesh.trilist,
                                              mesh.tcoords,
                                              self.texture_importer.image))
            else:
                meshes.append(TriMesh(mesh.points, mesh.trilist))
        # if self.landmarks is not None:
        #     mesh.landmarks.add_reference_landmarks(self.landmarks)
        # mesh.legacy = {'path_and_filename': self.path_and_filename}
        return meshes

    # def __str__(self):
    #     msg = 'n_meshes: %d' % self.n_meshes
    #     if self.texture is not None:
    #         msg += 'texture'


class AssimpImporter(AIImporter, MeshImporter):
    """Base class for importing 3D meshes
    """
    def __init__(self, filepath):
        super(AssimpImporter, self).__init__(filepath)


class WRLImporter(MeshImporter):
    """
    Allows importing VRML meshes.
    Uses a fork of PyVRML97 to do (hopefully) more robust parsing of VRML
    files. It should be noted that, unfortunately, this is a lot slower than
    the C++-based assimp importer.
    """

    def __init__(self, filepath):
        with open(filepath) as f:
            self.text = f.read()
        # Assumes a single mesh per file
        mesh, texture_path = self.parse_vrml97(self.text)
        self.relative_texture_path = texture_path
        self.meshes = []
        self.meshes.append(mesh)
        # Setup class before super class call
        super(WRLImporter, self).__init__(filepath)

    def parse_vrml97(self, file):
        parser = buildVRML97Parser()
        vrml_tuple = parser.parse(file)

        # Build expando object (dynamic object hack)
        mesh = lambda: 0

        # I assume these tuples are always built in this order
        scenegraph = vrml_tuple[1][1]
        transform = None
        for child in scenegraph.children:
            if type(child) is basenodes.Transform:
                transform = child

        if transform is None:
            raise MeshImportError('Unable to find transform in scenegraph')

        shape = None
        for child in transform.children:
            if type(child) is basenodes.Shape:
                shape = child

        if shape is None:
            raise MeshImportError('Unable to find shape in transform')

        mesh.points = shape.geometry.coord.point
        mesh.tcoords = shape.geometry.texCoord.point
        # Drop the -1 delimiters
        mesh.trilist = shape.geometry.coordIndex.reshape([-1, 4])[:, :3]
        # See if we have a seperate texture index, if not just create an empty
        # array
        try:
            tex_trilist = shape.geometry.texCoordIndex.reshape([-1, 4])[:, :3]
        except AttributeError:
            tex_trilist = np.array([-1])

        # Fix texture coordinates - we can only have one index so we choose
        # to use the triangle index
        if np.max(tex_trilist) > np.max(mesh.trilist):
            new_tcoords = np.zeros([mesh.points.shape[0], 2])
            new_tcoords[mesh.trilist] = mesh.tcoords[tex_trilist]
            mesh.tcoords = new_tcoords

        # Get the texture path - it's fine not to have one defined
        try:
            texture_path = shape.appearance.texture.url[0]
        except (AttributeError, IndexError):
            texture_path = None

        return mesh, texture_path


class FIMImporter(MeshImporter):
    """
    Allows importing floating point images as meshes.
    This reads in the shape in to 3 channels and then triangulates the x and y
    coordinates to create a surface.
    """

    def __init__(self, filepath):
        # Impossible to know where the texture is in this format
        self.relative_texture_path = None
        self.meshes = []
        self.meshes.append(self.parse_fim(filepath))
        # Setup class before super class call
        super(FIMImporter, self).__init__(filepath)

    def parse_fim(self, filepath):
        with open(filepath, 'rb') as f:
            size = np.fromfile(f, dtype=np.uint32, count=3)
            data = np.fromfile(f, dtype=np.float32, count=np.product(size))
            data = data.reshape([size[0] * size[1], size[2]])

        # Build expando object (dynamic object hack)
        mesh = lambda: 0

        mesh.points = data
        # Triangulate just the 2D coordinates, as this is a surface
        mesh.trilist = Delaunay(data[:, :2]).simplices

        return mesh


class BNTImporter(MeshImporter):
    """
    Allows importing the BNT file format from the bosphorus dataset.
    This reads in the 5 channels (3D coordinates and texture coordinates),
    splits them appropriately and then triangulates the x and y
    coordinates to create a surface. The texture path is also given in the file
    format.
    """

    def __init__(self, filepath):
        mesh, texture_path = self.parse_bnt(filepath)
        self.relative_texture_path = texture_path
        self.meshes = []
        self.meshes.append(mesh)
        # Setup class before super class call
        super(BNTImporter, self).__init__(filepath)

    def parse_bnt(self, filepath):
        with open(filepath, 'rb') as f:
            # Currently these are unused, but they are in the format
            # Could possibly store as metadata?
            n_rows = np.fromfile(f, dtype=np.uint16, count=1)
            n_cols = np.fromfile(f, dtype=np.uint16, count=1)
            z_min = np.fromfile(f, dtype=np.float64, count=1)

            # Get integers and convert to valid string
            image_path_len = np.fromfile(f, dtype=np.uint16, count=1)
            texture_path = np.fromfile(f, dtype=np.uint8, count=image_path_len)
            texture_path = ''.join(map(chr, texture_path))

            # Get data and reshape (reshape in an odd order due to Matlab's
            # Fortran ordering). First three columns are 3D coordinates
            # and last two are 2D texture coordinates
            coords_len = np.fromfile(f, dtype=np.uint32, count=1)
            data = np.fromfile(f, dtype=np.float64, count=coords_len * 5)
            data = data.reshape([5, coords_len/5]).T

        # Build expando object (dynamic object hack)
        mesh = lambda: 0

        # Get the 3D coordinates
        points = data[:, :3]
        # We want to remove the z-min plane because otherwise the mesh is not
        # renderable. We assume that if any point has a z-coordinate of the
        # z-min value then the whole point is worthless, so we drop it.
        valid_indices = points[:, 2] != z_min
        points = points[valid_indices, :]
        mesh.points = points

        # Apparently the texture coordinates are upside down?
        mesh.tcoords = np.flipud(data[:, -2:])
        # We also filter the texture coordinate by the valid points. Because
        # the mesh is actually laid out as an image, each point has one
        # texture coordinate, so this mapping is valid.
        mesh.tcoords = mesh.tcoords[valid_indices]

        # Triangulate just the 2D coordinates, as this is a surface
        mesh.trilist = Delaunay(points[:, :2]).simplices

        return mesh, texture_path