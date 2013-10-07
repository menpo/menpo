# A list of extensions that different importers support.
from pybug.io.spatial_image import BNTImporter, FIMImporter, ABSImporter
from pybug.io.landmark import LM3Importer, LANImporter, LM2Importer, \
    BNDImporter
from pybug.io.landmark_mesh import MeshPTSImporter
from pybug.io.mesh import AssimpImporter, WRLImporter
from pybug.io.image import PILImporter
from pybug.io.landmark_image import ImageASFImporter, ImagePTSImporter


mesh_types = {'.dae': AssimpImporter,
              '.3ds': AssimpImporter,
              '.ase': AssimpImporter,
              '.obj': AssimpImporter,
              '.ifc': AssimpImporter,
              '.xgl': AssimpImporter,
              '.zgl': AssimpImporter,
              '.ply': AssimpImporter,
              '.dxf': AssimpImporter,
              '.lwo': AssimpImporter,
              '.lws': AssimpImporter,
              '.lxo': AssimpImporter,
              '.stl': AssimpImporter,
              '.x': AssimpImporter,
              '.ac': AssimpImporter,
              '.md5': AssimpImporter,
              '.smd': AssimpImporter,
              '.vta': AssimpImporter,
              '.m3': AssimpImporter,
              '.3d': AssimpImporter,
              '.b3d': AssimpImporter,
              '.q3d': AssimpImporter,
              '.q3s': AssimpImporter,
              '.nff': AssimpImporter,
              '.off': AssimpImporter,
              '.raw': AssimpImporter,
              '.ter': AssimpImporter,
              '.mdl': AssimpImporter,
              '.hmp': AssimpImporter,
              '.ndo': AssimpImporter,
              '.ms3d': AssimpImporter,
              '.cob': AssimpImporter,
              '.scn': AssimpImporter,
              '.bvh': AssimpImporter,
              '.csm': AssimpImporter,
              '.xml': AssimpImporter,
              '.irrmesh': AssimpImporter,
              '.irr': AssimpImporter,
              '.md2': AssimpImporter,
              '.md3': AssimpImporter,
              '.pk3': AssimpImporter,
              '.mdc': AssimpImporter,
              # '.blend': AssimpImporter,
              '.wrl': WRLImporter}

spatial_image_types = {'.bnt': BNTImporter,
                       '.fim': FIMImporter,
                       '.abs': ABSImporter}

image_types = {'.bmp': PILImporter,
               '.dib': PILImporter,
               '.dcx': PILImporter,
               '.eps': PILImporter,
               '.ps': PILImporter,
               '.gif': PILImporter,
               '.im': PILImporter,
               '.jpg': PILImporter,
               '.jpe': PILImporter,
               '.jpeg': PILImporter,
               '.pcd': PILImporter,
               '.pcx': PILImporter,
               '.png': PILImporter,
               '.pbm': PILImporter,
               '.pgm': PILImporter,
               '.ppm': PILImporter,
               '.psd': PILImporter,
               '.tif': PILImporter,
               '.tiff': PILImporter,
               '.xbm': PILImporter,
               # '.pdf': PILImporter,
               '.xpm': PILImporter}

all_image_types = {}
all_image_types.update(spatial_image_types)
all_image_types.update(image_types)

image_landmark_types = {'.asf': ImageASFImporter,
                        '.lm2': LM2Importer,
                        '.pts': ImagePTSImporter}

mesh_landmark_types = {'.pts3': MeshPTSImporter,
                       '.lm3': LM3Importer,
                       '.lan': LANImporter,
                       '.bnd': BNDImporter}
