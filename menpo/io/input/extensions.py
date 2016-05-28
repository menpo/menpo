from .landmark import lm2_importer, ljson_importer
from .image import (pillow_importer, imageio_importer, imageio_gif_importer,
                    abs_importer, flo_importer)
from .video import ffmpeg_types
from .landmark_image import asf_image_importer, pts_image_importer
from .pickle import pickle_importer, pickle_gzip_importer


image_types = {'.bmp': imageio_importer,
               '.dib': pillow_importer,
               '.dcx': pillow_importer,
               '.eps': pillow_importer,
               '.ps': pillow_importer,
               '.gif': imageio_gif_importer,
               '.im': pillow_importer,
               '.jpg': imageio_importer,
               '.jpg2': imageio_importer,
               '.jpx': pillow_importer,
               '.jpe': imageio_importer,
               '.jpeg': imageio_importer,
               '.pcd': pillow_importer,
               '.pcx': pillow_importer,
               '.png': pillow_importer,
               '.pbm': pillow_importer,
               '.pgm': pillow_importer,
               '.ppm': pillow_importer,
               '.psd': pillow_importer,
               '.tif': pillow_importer,
               '.tiff': pillow_importer,
               '.xbm': pillow_importer,
               '.xpm': pillow_importer,
               '.abs': abs_importer,
               '.flo': flo_importer}


ffmpeg_video_types = ffmpeg_types()

image_landmark_types = {'.asf': asf_image_importer,
                        '.lm2': lm2_importer,
                        '.pts': pts_image_importer,
                        '.ptsx': pts_image_importer,
                        '.ljson': ljson_importer}

pickle_types = {'.pkl': pickle_importer,
                '.pkl.gz': pickle_gzip_importer}
