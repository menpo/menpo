# A list of extensions that different importers support.
from .landmark import LM2Importer, LJSONImporter
from .image import PILImporter, PILGIFImporter
from .landmark_image import ImageASFImporter, ImagePTSImporter
from .pickle import PickleImporter, GZipPickleImporter

image_types = {'.bmp': PILImporter,
               '.dib': PILImporter,
               '.dcx': PILImporter,
               '.eps': PILImporter,
               '.ps': PILImporter,
               '.gif': PILGIFImporter,
               '.im': PILImporter,
               '.jpg': PILImporter,
               '.jpg2': PILImporter,
               '.jpx': PILImporter,
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
                        '.lm2': LM2Importer,
                        '.pts': ImagePTSImporter,
                        '.ptsx': ImagePTSImporter,
                        '.ljson': LJSONImporter}

pickle_types = {'.pkl': PickleImporter,
                '.pkl.gz': GZipPickleImporter}
