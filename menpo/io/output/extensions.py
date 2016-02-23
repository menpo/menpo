from .landmark import LJSONExporter, PTSExporter
from .image import PILExporter
from .video import ImageioVideoExporter, ImageioGifExporter
from .pickle import pickle_export


landmark_types = {
    '.ljson': LJSONExporter,
    '.pts': PTSExporter
}


image_types = {
    '.bmp': PILExporter,
    '.dib': PILExporter,
    '.dcx': PILExporter,
    '.eps': PILExporter,
    '.ps': PILExporter,
    '.gif': PILExporter,
    '.im': PILExporter,
    '.jpg': PILExporter,
    '.jpe': PILExporter,
    '.jpeg': PILExporter,
    '.pcd': PILExporter,
    '.pcx': PILExporter,
    '.png': PILExporter,
    '.pbm': PILExporter,
    '.pgm': PILExporter,
    '.ppm': PILExporter,
    '.psd': PILExporter,
    '.tif': PILExporter,
    '.tiff': PILExporter,
    '.xbm': PILExporter,
    '.xpm': PILExporter
}


pickle_types = {
    '.pkl': pickle_export,
    '.pkl.gz': pickle_export,
}


video_types = {
    '.mov': ImageioVideoExporter,
    '.avi': ImageioVideoExporter,
    '.mpg': ImageioVideoExporter,
    '.mpeg': ImageioVideoExporter,
    '.mp4': ImageioVideoExporter,
    '.mkv': ImageioVideoExporter,
    '.wmv': ImageioVideoExporter,
    '.gif': ImageioGifExporter
}
