import commands
import os.path as path
import tempfile
from pybug.io import metadata
from pybug.shape import TexturedTriMesh, TriMesh
from pybug.io.base import Importer
from pybug.io.mesh.assimp import AIImporter
from pybug.io.image import ImageImporter


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


class MeshImporter(AIImporter, Importer):
    """Base class for importing 3D meshes
    """
    def __init__(self, filepath):
        super(MeshImporter, self).__init__(filepath)
        if self.texture_path is None:
            self.texture_importer = None
        else:
            self.texture_importer = ImageImporter(path.join(self.folder,
                                                  self.texture_path))

    def import_landmarks(self):
        try:
            self.landmarks = metadata.json_pybug_landmarks(
                self.path_and_filename)
        except metadata.MissingLandmarksError:
            self.landmarks = None

    def build(self, **kwargs):
        if self.texture_importer is not None:
            mesh = TexturedTriMesh(self.points, self.trilist,
                                   self.tcoords, self.texture_importer.image)
        else:
            mesh = TriMesh(self.points, self.trilist)
        if self.landmarks is not None:
            mesh.landmarks.add_reference_landmarks(self.landmarks)
        mesh.legacy = {'path_and_filename': self.path_and_filename}
        return mesh
