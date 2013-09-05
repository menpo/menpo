import abc
import commands
import os.path as path
import tempfile
from pybug.io.base import Importer, find_alternative_files, get_importer
from pybug.io.mesh.assimp import AIImporter
from pybug.io.exceptions import MeshImportError
from pybug.shape import TexturedTriMesh, TriMesh
from pyvrml import buildVRML97Parser
import pyvrml.vrml97.basenodes as basenodes
from scipy.spatial import Delaunay
import numpy as np
import re


def process_with_meshlabserver(file_path, output_dir=None, script_path=None,
                               output_filetype=None, export_flags=None):
    r"""
    Interface to `meshlabserver` to perform prepossessing on meshes before
    import. Returns a path to the result of the meshlabserver call, ready for
    import as usual. **Requires Meshlab to be installed**.

    Parameters
    ----------
    file_path : string
        Absolute filepath to the mesh
    script_path : atring, optional
        If specified this script will be run on the input mesh.

        Default: ``None``
    output_dir : string, optional
        The output directory for the processed mesh.

        Default: The users tmp directory.
    output_filetype : string, optional
        The output filetype desired from meshlabserver. Takes the form of an
        extension, eg ``obj``.

        Default: The same as the input mesh
    export_flags : string, optional
        Flags passed to the ``-om`` parameter. Allows for choosing
        what aspects of the model will be exported (normals,
        texture coords etc)

    Returns
    -------
    output_path : string
        The absolute filepath to the processed mesh.
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
    r"""
    Abstract base class for importing meshes. Searches in the directory
    specified by filepath for landmarks and textures with the same basename as
    the mesh. If found, they are automatically attached. If a texture is found
    then a :class:`pybug.shape.mesh.textured.TexturedTriMesh` is built, else a
    :class:`pybug.shape.mesh.base.Trimesh` is built.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath):
        super(MeshImporter, self).__init__(filepath)
        self.meshes = []
        self.attempted_texture_search = False
        self.attempted_landmark_search = False

    def _build_texture_and_landmark_importers(self):
        r"""
        Search for a texture and landmark file in the same directory as the
        mesh. If they exist, create importers for them.
        """
        if self.texture_path is None or not path.exists(self.texture_path):
            self.texture_importer = None
        else:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import image_types
            self.texture_importer = get_importer(self.texture_path,
                                                 image_types)

        if self.landmark_path is None or not path.exists(self.landmark_path):
            self.landmark_importer = None
        else:
            # This import is here to avoid circular dependencies
            from pybug.io.extensions import mesh_landmark_types
            self.landmark_importer = get_importer(self.landmark_path,
                                                  mesh_landmark_types)

    def _search_for_texture(self):
        r"""
        Tries to find a texture with the same name as the mesh.

        Returns
        --------
        relative_texture_path : string
            The relative path to the texture or ``None`` if one can't be found
        """
        # Stop searching every single time we access the property
        self.attempted_texture_search = True
        # This import is here to avoid circular dependencies
        from pybug.io.extensions import image_types
        try:
            return find_alternative_files('texture', self.filepath,
                                          image_types)
        except ImportError:
            return None

    def _search_for_landmarks(self):
        """
        Tries to find a landmark file with the same name as the mesh.

        Returns
        --------
        relative_landmark_path : string
            The relative path to the landmarks or ``None`` if none can be found
        """
        # Stop searching every single time we access the property
        self.attempted_landmark_search = True
        # This import is here to avoid circular dependencies
        from pybug.io.extensions import mesh_landmark_types
        try:
            return find_alternative_files('landmarks', self.filepath,
                                          mesh_landmark_types)
        except ImportError:
            return None

    @property
    def texture_path(self):
        """
        Get the absolute path to the texture. Returns None if one can't be
        found. Makes it's best effort to find an appropriate texture by
        searching for textures with the same name as the mesh. Will only
        search for the path the first time ``texture_path`` is invoked.

        Sets the ``self.relative_texture_path`` attribute.

        Returns
        -------
        texture_path : string
            Absolute filepath to the texture
        """
        # Avoid attribute not being set
        if not hasattr(self, 'relative_texture_path'):
            self.relative_texture_path = None

        # Try find a texture path if we can
        if self.relative_texture_path is None and \
                not self.attempted_texture_search:
            self.relative_texture_path = self._search_for_texture()

        try:
            return path.join(self.folder, self.relative_texture_path)
        except AttributeError:
            return None

    @property
    def landmark_path(self):
        """
        Get the absolute path to the landmarks. Returns None if none can be
        found. Makes it's best effort to find an appropriate landmark set by
        searching for landmarks with the same name as the mesh. Will only
        search for the path the first time ``landmark_path`` is invoked.

        Sets the ``self.relative_landmark_path`` attribute.

        Returns
        -------
        landmark_path : string
            Absolute filepath to the landmarks
        """
        # Avoid attribute not being set
        if not hasattr(self, 'relative_landmark_path'):
            self.relative_landmark_path = None

        # Try find a texture path if we can
        if self.relative_landmark_path is None and \
                not self.attempted_landmark_search:
            self.relative_landmark_path = self._search_for_landmarks()

        try:
            return path.join(self.folder, self.relative_landmark_path)
        except AttributeError:
            return None

    @abc.abstractmethod
    def _parse_format(self):
        r"""
        Abstract method that handles actually building a mesh. This involves
        reading the mesh from disk and doing any necessary parsing.

        Should set the ``self.meshes`` attribute. Each mesh in ``self.meshes``
        is expected to be an object with attributes:

        ======== ==========================
        name     type
        ======== ==========================
        points   double ndarray
        trilist  int ndarray
        tcoords  double ndarray (optional)
        ======== ==========================

        May also set the ``self.relative_texture_path`` if it is specified by
        the format.
        """
        pass

    def build(self):
        r"""
        Overrides the :meth:`build <pybug.io.base.Importer.build>` method.

        Parse the format as defined by :meth:`_parse_format` and then search
        for valid textures and landmarks that may have been defined by the
        format.

        Build the appropriate type of mesh defined by parsing the format. May
        or may not be textured.

        Returns
        -------
        meshes : list of :class:`pybug.shape.mesh.textured.TexturedTriMesh` or :class:`pybug.shape.mesh.base.Trimesh`
            List of meshes
        """
        #
        self._parse_format()
        self._build_texture_and_landmark_importers()

        meshes = []
        for mesh in self.meshes:
            if self.texture_importer is not None:
                new_mesh = TexturedTriMesh(mesh.points.astype(np.float64),
                                           mesh.trilist,
                                           mesh.tcoords,
                                           self.texture_importer.build())
            else:
                new_mesh = TriMesh(mesh.points, mesh.trilist)

            if self.landmark_importer is not None:
                label, l_dict = self.landmark_importer.build(
                    scale_factors=np.max(mesh.points))
                new_mesh.add_landmark_set(label, l_dict)

            meshes.append(new_mesh)

        return meshes


class AssimpImporter(AIImporter, MeshImporter):
    """
    Uses assimp to import meshes. The assimp importing is wrapped via cython,

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """
    def __init__(self, filepath):
        super(AssimpImporter, self).__init__(filepath)

    def _parse_format(self):
        r"""
        Use assimp to build the mesh. Also, get the relative texture path.
        """
        self.build_scene()
        # Properties should have different names because of multiple
        # inheritance
        self.relative_texture_path = self.assimp_texture_path


class WRLImporter(MeshImporter):
    """
    Allows importing VRML meshes.
    Uses a fork of PyVRML97 to do (hopefully) more robust parsing of VRML
    files. It should be noted that, unfortunately, this is a lot slower than
    the C++-based assimp importer.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath):
        # Setup class before super class call
        super(WRLImporter, self).__init__(filepath)

    def _parse_format(self):
        r"""
        Use pyVRML to parse the file and build a mesh object. A single mesh per
        file is assumed.

        Raises
        ------
        MeshImportError
            If no transform or shape is found in the scenegraph
        """
        with open(self.filepath) as f:
            self.text = f.read()

        parser = buildVRML97Parser()
        vrml_tuple = parser.parse(self.text)

        # Build expando object (dynamic object hack)
        self.mesh = lambda: 0

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

        self.mesh.points = shape.geometry.coord.point
        self.mesh.tcoords = shape.geometry.texCoord.point
        # Drop the -1 delimiters
        self.mesh.trilist = shape.geometry.coordIndex.reshape([-1, 4])[:, :3]
        # See if we have a seperate texture index, if not just create an empty
        # array
        try:
            tex_trilist = shape.geometry.texCoordIndex.reshape([-1, 4])[:, :3]
        except AttributeError:
            tex_trilist = np.array([-1])

        # Fix texture coordinates - we can only have one index so we choose
        # to use the triangle index
        if np.max(tex_trilist) > np.max(self.mesh.trilist):
            new_tcoords = np.zeros([self.mesh.points.shape[0], 2])
            new_tcoords[self.mesh.trilist] = self.mesh.tcoords[tex_trilist]
            self.mesh.tcoords = new_tcoords

        # Get the texture path - it's fine not to have one defined
        try:
            self.relative_texture_path = shape.appearance.texture.url[0]
        except (AttributeError, IndexError):
            self.relative_texture_path = None

        # Assumes a single mesh per file
        self.meshes = [self.mesh]


class FIMImporter(MeshImporter):
    r"""
    Allows importing floating point images as meshes.
    This reads in the shape in to 3 channels and then triangulates the
    ``x`` and ``y`` coordinates to create a surface.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath):
        # Setup class before super class call
        super(FIMImporter, self).__init__(filepath)

    def _parse_format(self):
        r"""
        Read the file and parse it as necessary. Since the data lies on a grid
        we can triangulate the 2D coordinates to get a valid triangulation. One
        mesh is assumed per file.
        """
        with open(self.filepath, 'rb') as f:
            size = np.fromfile(f, dtype=np.uint32, count=3)
            data = np.fromfile(f, dtype=np.float32, count=np.product(size))
            data = data.reshape([size[0] * size[1], size[2]])

        # Build expando object (dynamic object hack)
        self.mesh = lambda: 0

        self.mesh.points = data
        # Triangulate just the 2D coordinates, as this is a surface
        self.mesh.trilist = Delaunay(data[:, :2]).simplices

        # Impossible to know where the texture is in this format
        self.relative_texture_path = None
        # Assumes a single mesh per file
        self.meshes = [self.mesh]


class ABSImporter(MeshImporter):
    r"""
    Allows importing the ABS file format from the FRGC dataset.
    This file format also includes a mask that we don't currently expose.

    The z-min value is stripped from the mesh to make it renderable. We are
    currently unable to texture these meshes as we don't have texture
    coordinates.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath):
        # Setup class before super class call
        super(ABSImporter, self).__init__(filepath)

    def _parse_format(self):
        r"""
        Read in the file and remove the z-min. Triangulate the 2D gridded
        coordinates to create a valid triangulation.
        """
        with open(self.filepath, 'r') as f:
            # Currently these are unused, but they are in the format
            # Could possibly store as metadata?
            # Assume first result for regexes
            re_rows = re.compile(u'([0-9]+) rows')
            n_rows = int(re_rows.findall(f.readline())[0])
            re_cols = re.compile(u'([0-9]+) columns')
            n_cols = int(re_cols.findall(f.readline())[0])

        # Currently this also loads the mask, which we don't use
        # The mask is located at
        #   >>> image_data[:, 0]
        image_data = np.loadtxt(self.filepath, skiprows=3, unpack=True)

        # Build expando object (dynamic object hack)
        self.mesh = lambda: 0

        # Get the 3D coordinates
        points = np.hstack([image_data[:, 1][..., None],
                            image_data[:, 2][..., None],
                            image_data[:, 3][..., None]])
        # We want to remove the z-min plane because otherwise the mesh is not
        # renderable. We assume that if any point has a z-coordinate of the
        # z-min value then the whole point is worthless, so we drop it.
        valid_indices = points[:, 2] != np.min(points[:, 2])
        points = points[valid_indices, :]
        self.mesh.points = points

        # Triangulate just the 2D coordinates, as this is a surface
        self.mesh.trilist = Delaunay(points[:, :2]).simplices

        # TODO: Although texture exist for this dataset, there is no texture
        # coordinate set, so we can't currently use them
        self.mesh.tcoords = []

        self.relative_texture_path = None
        # Assumes a single mesh per file
        self.meshes = [self.mesh]
