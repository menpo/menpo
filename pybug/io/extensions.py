# A list of extensions that different importers support.
from pybug.io.landmark import LM3Importer, LANImporter
from pybug.io.landmark_mesh import MeshPTSImporter
from pybug.io.mesh import AssimpImporter, WRLImporter, FIMImporter, \
    BNTImporter, ABSImporter
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
              '.wrl': WRLImporter,
              '.fim': FIMImporter,
              '.bnt': BNTImporter,
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

image_landmark_types = {'.asf': ImageASFImporter,
                        '.pts': ImagePTSImporter}

mesh_landmark_types = {'.pts': MeshPTSImporter,
                       '.lm3': LM3Importer,
                       '.lan': LANImporter}
