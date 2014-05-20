import abc
from collections import namedtuple
import commands
import os.path as path
import tempfile
import numpy as np
from cyassimp import AIImporter
from vrml.vrml97.parser import buildParser as buildVRML97Parser
import vrml.vrml97.basenodes as basenodes
from vrml.node import NullNode
from menpo.io.base import (Importer, find_alternative_files,
                           map_filepath_to_importer)
from menpo.io.exceptions import MeshImportError
from menpo.shape.mesh import ColouredTriMesh, TexturedTriMesh, TriMesh

# This formalises the return type of a mesh importer (before building)
# However, at the moment there is a disconnect between this and the
# Assimp type, and at some point they should become the same object
MeshInfo = namedtuple('MeshInfo', ['points', 'trilist', 'tcoords',
                                   'colour_per_vertex'])


def process_with_meshlabserver(file_path, output_dir=None, script_path=None,
                               output_filetype=None, export_flags=None,
                               meshlab_command='meshlabserver'):
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

        Default: `None`
    output_dir : string, optional
        The output directory for the processed mesh.

        Default: The users tmp directory.
    output_filetype : string, optional
        The output filetype desired from meshlabserver. Takes the form of an
        extension, eg `obj`.

        Default: The same as the input mesh
    export_flags : string, optional
        Flags passed to the `-om` parameter. Allows for choosing
        what aspects of the model will be exported (normals,
        texture coords etc)
    meshlab_command : string, optional
        The meshlabserver executable to run.

        Default: 'meshlabserver'

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
    command = (meshlab_command + ' -i ' + file_path + ' -o ' +
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
    then a :class:`menpo.shape.mesh.textured.TexturedTriMesh` is built, else a
    :class:`menpo.shape.mesh.base.Trimesh` is built. Note that this behavior
    can be overridden if desired.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath, texture=True):
        super(MeshImporter, self).__init__(filepath)
        self.meshes = []
        self.import_textures = texture
        self.attempted_texture_search = False
        self.relative_texture_path = None
        self.texture_importer = None

    def _build_texture_importer(self):
        r"""
        Search for a texture in the same directory as the
        mesh. If it exists, create an importer for it.
        """
        if self.texture_path is None or not path.exists(self.texture_path):
            self.texture_importer = None
        else:
            # This import is here to avoid circular dependencies
            from menpo.io.extensions import image_types
            self.texture_importer = map_filepath_to_importer(self.texture_path,
                                                             image_types)

    def _search_for_texture(self):
        r"""
        Tries to find a texture with the same name as the mesh.

        Returns
        --------
        relative_texture_path : string
            The relative path to the texture or `None` if one can't be found
        """
        # Stop searching every single time we access the property
        self.attempted_texture_search = True
        # This import is here to avoid circular dependencies
        from menpo.io.extensions import image_types
        try:
            return find_alternative_files('texture', self.filepath,
                                          image_types)
        except ImportError:
            return None

    @property
    def texture_path(self):
        """
        Get the absolute path to the texture. Returns None if one can't be
        found. Makes it's best effort to find an appropriate texture by
        searching for textures with the same name as the mesh. Will only
        search for the path the first time `texture_path` is invoked.

        Sets the `self.relative_texture_path` attribute.

        Returns
        -------
        texture_path : string
            Absolute filepath to the texture
        """
        # Try find a texture path if we can
        if (self.relative_texture_path is None and not
                self.attempted_texture_search):
            self.relative_texture_path = self._search_for_texture()

        try:
            return path.join(self.folder, self.relative_texture_path)
        # AttributeError POSIX, TypeError Windows
        except (AttributeError, TypeError):
            return None

    @abc.abstractmethod
    def _parse_format(self):
        r"""
        Abstract method that handles actually building a mesh. This involves
        reading the mesh from disk and doing any necessary parsing.

        Should set the `self.meshes` attribute. Each mesh in `self.meshes`
        is expected to be an object with attributes:

        ======== ==========================
        name     type
        ======== ==========================
        points   double ndarray
        trilist  int ndarray
        tcoords  double ndarray (optional)
        ======== ==========================

        May also set the `self.relative_texture_path` if it is specified by
        the format.
        """
        pass

    def build(self):
        r"""
        Overrides the :meth:`build <menpo.io.base.Importer.build>` method.

        Parse the format as defined by :meth:`_parse_format` and then search
        for valid textures and landmarks that may have been defined by the
        format.

        Build the appropriate type of mesh defined by parsing the format. May
        or may not be textured.

        Returns
        -------
        meshes : list of :class:`menpo.shape.mesh.textured.TexturedTriMesh` or :class:`menpo.shape.mesh.base.Trimesh`
            If more than one mesh, returns a list of meshes. If only one
            mesh, returns the single mesh.
        """
        #
        self._parse_format()
        if self.import_textures:
            self._build_texture_importer()

        meshes = []
        for mesh in self.meshes:
            if self.texture_importer is not None:
                new_mesh = TexturedTriMesh(mesh.points.astype(np.float64),
                                           mesh.tcoords,
                                           self.texture_importer.build(),
                                           trilist=mesh.trilist)
            elif mesh.colour_per_vertex is not None:
                new_mesh = ColouredTriMesh(mesh.points,
                                           colours=mesh.colour_per_vertex,
                                           trilist=mesh.trilist)
            else:
                new_mesh = TriMesh(mesh.points, trilist=mesh.trilist)

            meshes.append(new_mesh)
        if len(meshes) == 1:
            return meshes[0]
        else:
            return meshes


class AssimpImporter(MeshImporter):
    """
    Uses assimp to import meshes. The assimp importing is wrapped via cython,

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """
    def __init__(self, filepath, texture=True):
        MeshImporter.__init__(self, filepath, texture=texture)
        self.ai_importer = AIImporter(filepath)

    def _parse_format(self):
        r"""
        Use assimp to build the mesh and get the relative texture path.
        """
        self.ai_importer.build_scene()
        self.meshes = self.ai_importer.meshes
        self.relative_texture_path = self.ai_importer.assimp_texture_path


class WRLImporter(MeshImporter):
    """
    Allows importing VRML meshes.
    Uses a fork of PyVRML97 to do (hopefully) more robust parsing of VRML
    files. It should be noted that, unfortunately, this is a lot slower than
    the C++-based assimp importer.

    VRML allows non-triangular polygons, whilst our importer pipeline doesn't.
    Therefore, any non-triangular polygons are dropped. VRML also allows
    separate texture coordinate indices, which we do not support. To have
    a better formed mesh, try exporting the WRL as OBJ from Meshlab.

    Parameters
    ----------
    filepath : string
        Absolute filepath of the mesh.
    """

    def __init__(self, filepath, texture=True):
        # Setup class before super class call
        super(WRLImporter, self).__init__(filepath, texture=texture)

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

        # I assume these tuples are always built in this order
        scenegraph = vrml_tuple[1][1]
        shape_container = None

        # Let's check if
        for child in scenegraph.children:
            if type(child) in [basenodes.Transform, basenodes.Group]:
                # Only fetch the first container (unknown what do do with more
                # than one container at this time)
                shape_container = child
                break

        if shape_container is None:
            raise MeshImportError('Unable to find shape container in '
                                  'scenegraph')

        shape = None
        for child in shape_container.children:
            if type(child) is basenodes.Shape:
                # Only fetch the first shape (unknown what do do with more
                # than one shape at this time)
                shape = child
                break

        if shape is None:
            raise MeshImportError('Unable to find shape in container')

        mesh_points = shape.geometry.coord.point
        mesh_trilist = self._filter_non_triangular_polygons(
                    shape.geometry.coordIndex)

        if type(shape.geometry.texCoord) is NullNode:  # Colour per-vertex
            mesh_colour_per_vertex = shape.geometry.color.color
            mesh_tcoords = None
        else:  # Texture coordinates
            mesh_colour_per_vertex = None
            mesh_tcoords = shape.geometry.texCoord.point

            # See if we have a separate texture index, if not just create an
            # empty array
            try:
                tex_trilist = self._filter_non_triangular_polygons(
                    shape.geometry.texCoordIndex)
            except AttributeError:
                tex_trilist = np.array([-1])

            # Fix texture coordinates - we can only have one index so we choose
            # to use the triangle index
            if np.max(tex_trilist) > np.max(mesh_trilist):
                new_tcoords = np.zeros([mesh_points.shape[0], 2])
                new_tcoords[mesh_trilist] = mesh_tcoords[tex_trilist]
                mesh_tcoords = new_tcoords

            # Get the texture path - it's fine not to have one defined
            try:
                self.relative_texture_path = shape.appearance.texture.url[0]
            except (AttributeError, IndexError):
                self.relative_texture_path = None

        # Assumes a single mesh per file
        self.mesh = MeshInfo(mesh_points, mesh_trilist, mesh_tcoords,
                             mesh_colour_per_vertex)
        self.meshes = [self.mesh]

    def _filter_non_triangular_polygons(self, coord_list):
        # VRML allows arbitrary polygon coordinates, whilst we only support
        # triangles. They are delimited by -1, so we split on them and filter
        # out non-triangle polygons
        index_list = coord_list
        index_list = np.split(index_list, np.where(index_list == -1)[0])
        # The first polygon is missing the -1 at the beginning
        # Have to cast to int32 because that's the default, but on 64bit
        # machines a single number defaults to int64
        np.insert(index_list[0], 0, np.array([-1], dtype=np.int32))
        # Filter out those coordinates that are not triangles
        index_list = [i for i in index_list if len(i[1:]) == 3]
        # Convert to 2D array
        index_list = np.array(index_list)
        # Slice of -1 delimiters
        return index_list[:, 1:]
