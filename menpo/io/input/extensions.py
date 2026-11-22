from typing import Any, Callable

from .image import abs_importer, flo_importer, pillow_importer
from .landmark import ljson_importer, lm2_importer, tem_importer
from .landmark_image import asf_image_importer, pts_image_importer
from .pickle import pickle_gzip_importer, pickle_importer
from .video import ffmpeg_importer, ffmpeg_types

image_types: dict[str, Callable[..., Any]] = {
    ".bmp": pillow_importer,
    ".dib": pillow_importer,
    ".dcx": pillow_importer,
    ".eps": pillow_importer,
    ".ps": pillow_importer,
    ".gif": ffmpeg_importer,
    ".im": pillow_importer,
    ".jpg": pillow_importer,
    ".jpg2": pillow_importer,
    ".jpx": pillow_importer,
    ".jpe": pillow_importer,
    ".jpeg": pillow_importer,
    ".pcd": pillow_importer,
    ".pcx": pillow_importer,
    ".png": pillow_importer,
    ".pbm": pillow_importer,
    ".pgm": pillow_importer,
    ".ppm": pillow_importer,
    ".psd": pillow_importer,
    ".tif": pillow_importer,
    ".tiff": pillow_importer,
    ".xbm": pillow_importer,
    ".xpm": pillow_importer,
    ".abs": abs_importer,
    ".flo": flo_importer,
}


ffmpeg_video_types = ffmpeg_types()
ffmpeg_video_types[".gif"] = ffmpeg_importer

image_landmark_types = {
    ".asf": asf_image_importer,
    ".lm2": lm2_importer,
    ".pts": pts_image_importer,
    ".ptsx": pts_image_importer,
    ".ljson": ljson_importer,
    ".tem": tem_importer,
}

pickle_types = {".pkl": pickle_importer, ".pkl.gz": pickle_gzip_importer}
