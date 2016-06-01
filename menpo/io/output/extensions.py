from functools import partial

from .landmark import ljson_exporter, pts_exporter
from .image import pil_exporter
from .video import ffmpeg_video_exporter
from .pickle import pickle_exporter


landmark_types = {
    '.ljson': ljson_exporter,
    '.pts': pts_exporter
}


image_types = {
    '.bmp': pil_exporter,
    '.dib': pil_exporter,
    '.dcx': pil_exporter,
    '.eps': pil_exporter,
    '.ps': pil_exporter,
    '.gif': pil_exporter,
    '.im': pil_exporter,
    '.jpg': pil_exporter,
    '.jpe': pil_exporter,
    '.jpeg': pil_exporter,
    '.pcd': pil_exporter,
    '.pcx': pil_exporter,
    '.png': pil_exporter,
    '.pbm': pil_exporter,
    '.pgm': pil_exporter,
    '.ppm': pil_exporter,
    '.psd': pil_exporter,
    '.tif': pil_exporter,
    '.tiff': pil_exporter,
    '.xbm': pil_exporter,
    '.xpm': pil_exporter
}


pickle_types = {
    '.pkl': pickle_exporter,
    '.pkl.gz': pickle_exporter,
}


video_types = {
    '.mov': ffmpeg_video_exporter,
    '.avi': ffmpeg_video_exporter,
    '.mpg': ffmpeg_video_exporter,
    '.mpeg': ffmpeg_video_exporter,
    '.mp4': ffmpeg_video_exporter,
    '.mkv': ffmpeg_video_exporter,
    '.wmv': ffmpeg_video_exporter,
    '.gif': partial(ffmpeg_video_exporter, codec=None)
}
