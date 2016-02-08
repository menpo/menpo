from .landmark import LM2Importer, LJSONImporter
from .image import (PILImporter, PILGIFImporter, ImageioImporter, ABSImporter,
                    FLOImporter)
from .video import ImageioFFMPEGImporter
from .landmark_image import ImageASFImporter, ImagePTSImporter
from .pickle import PickleImporter, GZipPickleImporter


image_types = {'.bmp': ImageioImporter,
               '.dib': PILImporter,
               '.dcx': PILImporter,
               '.eps': PILImporter,
               '.ps': PILImporter,
               '.gif': PILGIFImporter,
               '.im': PILImporter,
               '.jpg': ImageioImporter,
               '.jpg2': ImageioImporter,
               '.jpx': PILImporter,
               '.jpe': ImageioImporter,
               '.jpeg': ImageioImporter,
               '.pcd': PILImporter,
               '.pcx': PILImporter,
               '.png': ImageioImporter,
               '.pbm': PILImporter,
               '.pgm': PILImporter,
               '.ppm': PILImporter,
               '.psd': PILImporter,
               '.tif': PILImporter,
               '.tiff': PILImporter,
               '.xbm': PILImporter,
               '.xpm': PILImporter,
               '.abs': ABSImporter,
               '.flo': FLOImporter}


ffmpeg_video_types = ImageioFFMPEGImporter.ffmpeg_types()

image_landmark_types = {'.asf': ImageASFImporter,
                        '.lm2': LM2Importer,
                        '.pts': ImagePTSImporter,
                        '.ptsx': ImagePTSImporter,
                        '.ljson': LJSONImporter}

pickle_types = {'.pkl': PickleImporter,
                '.pkl.gz': GZipPickleImporter}
