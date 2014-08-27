import abc
from collections import namedtuple
import os.path as path
import json

import numpy as np
from menpo.io.input.base import Importer, find_alternative_files, import_image
from menpo.io.exceptions import MeshImportError
from menpo.shape.mesh import ColouredTriMesh, TexturedTriMesh, TriMesh



# This formalises the return type of a mesh importer (before building)
# However, at the moment there is a disconnect between this and the
# Assimp type, and at some point they should become the same object
MeshInfo = namedtuple('MeshInfo', ['points', 'trilist', 'tcoords',
                                   'colour_per_vertex'])


class MeshImporter(Importer):
    r"""
    Abstract base class for importing meshes. Searches in the directory
    specified by filepath for landmarks and textures with the same basename as
    the mesh. If found, they are automatically attached. If a texture is found
    then a :map:`TexturedTriMesh` is built, if colour information is found a
    :map:`ColouredTriMesh` is built, and if neither is found a :map:`Trimesh`
    is built. Note that this behavior can be overridden if desired.

    Parameters
    ----------
    filepath : `str`
        Absolute filepath of the mesh.
    texture: 'bool', optional
        If ``False`` even if a texture exists a normal :map:`TriMesh` is
        produced.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, filepath, texture=True):
        super(MeshImporter, self).__init__(filepath)
        self.meshes = []
        self.import_textures = texture
        self.attempted_texture_search = False
        self.relative_texture_path = None

    def _search_for_texture(self):
        r"""
        Tries to find a texture with the same name as the mesh.

        Returns
        --------
        relative_texture_path : `str`
            The relative path to the texture or ``None`` if one can't be found
        """
        # Stop searching every single time we access the property
        self.attempted_texture_search = True
        # This import is here to avoid circular dependencies
        from menpo.io.input.extensions import image_types
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
        search for the path the first time ``texture_path`` is invoked.

        Sets the :attr:`relative_texture_path`.

        Returns
        -------
        texture_path : `str`
            Absolute filepath to the texture
        """
        # Try find a texture path if we can
        if (self.relative_texture_path is None and not
                self.attempted_texture_search):
            self.relative_texture_path = self._search_for_texture()
        try:
            texture_path = path.join(self.folder, self.relative_texture_path)
        # AttributeError POSIX, TypeError Windows
        except (AttributeError, TypeError):
            return None
        if not path.isfile(texture_path):
            texture_path = None
        return texture_path

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

        May also set the :attr:`relative_texture_path` if it is specified by
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

        # Only want to create textured meshes if there is a texture path
        # and import_textures is True
        textured = self.import_textures and self.texture_path is not None

        meshes = []
        for mesh in self.meshes:
            if textured:
                new_mesh = TexturedTriMesh(mesh.points,
                                           mesh.tcoords,
                                           import_image(self.texture_path),
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
        from cyassimp import AIImporter  # expensive import
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
        # inlined expensive imports
        from vrml.vrml97.parser import buildParser as buildVRML97Parser
        import vrml.vrml97.basenodes as basenodes
        from vrml.node import NullNode
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


class MJSONImporter(MeshImporter):
    """
    Import meshes that are in a simple JSON format.

    """

    def _parse_format(self):
        with open(self.filepath, 'rb') as f:
            mesh_json = json.load(f)
        mesh = MeshInfo(mesh_json['points'], mesh_json['trilist'],
                        mesh_json.get('tcoords'),
                        mesh_json.get('colour_per_vertex'))
        self.meshes = [mesh]
