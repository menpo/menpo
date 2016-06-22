import os
import subprocess as sp
import warnings

import numpy as np
from pathlib import Path

from menpo.visualize import print_progress
from ..utils import DEVNULL, _call_subprocess


_FFMPEG_CMD = lambda: str(Path(os.environ.get('MENPO_FFMPEG_CMD', 'ffmpeg')))


def ffmpeg_video_exporter(images, out_path, fps=30, codec='libx264',
                          preset='medium', bitrate=None, verbose=False,
                          **kwargs):
    r"""
    Uses subprocess PIPE to export the images using FFMPEG.

    There are is one important environment variable that can be set to alter
    the behaviour of this function:

        ================== ======================================
        ENV Variable       Definition
        ================== ======================================
        MENPO_FFMPEG_CMD   The path to the 'ffmpeg' executable.
        ================== ======================================

    Parameters
    ----------
    images : `list` of :map:`Image`
        List of Menpo images to export as a video.
    out_path : `Path`
        Path to save the video to.
    fps : `int`, optional
        The number of frames per second.
    codec : `str`, optional
        The video codec to use. Default 'libx264', which represents the
        widely available mpeg4. Except when saving .wmv files, then the
        defaults is 'msmpeg4' which is more commonly supported for windows.
    preset : `str`, optional
        The preset FFMPEG compression level.
    bitrate: `str`, optional
        The output video bitrate.
    verbose : `bool`, optional
        If ``True``, print a progress bar.
    """
    # Some of the below was inspired by moviepy:
    #   https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_writer.py
    # and is used under the terms of the MIT license which can be found at
    #   https://github.com/Zulko/moviepy/blob/master/LICENCE.txt
    first_image = images[0]
    frame_shape = first_image.shape
    # If the first image is gray then all the images will be assumed to be
    # gray
    colour = 'rgb24' if images[0].n_channels == 3 else 'gray8'
    cmd = [_FFMPEG_CMD(), '-y',
           '-s', '{}x{}'.format(frame_shape[1], frame_shape[0]),
           '-r', str(fps),
           '-an',
           '-pix_fmt', colour,
           '-c:v', 'rawvideo', '-f', 'rawvideo',
           '-i', '-']
    if codec:
        cmd.extend(['-vcodec', codec])
    if preset:
        cmd.extend(['-preset', preset])
    if bitrate:
        cmd.extend(['-b', str(bitrate)])
    cmd.append(str(out_path))

    images = (print_progress(images, prefix='Exporting frames') if verbose
              else images)

    # Pipe stdout to DEVNULL to ignore it
    with _call_subprocess(sp.Popen(cmd, stdin=sp.PIPE, stderr=sp.PIPE,
                                   stdout=DEVNULL)) as pipe:
        for k, image in enumerate(images):
            try:
                if image.n_channels != 1 and colour == 'gray8':
                    warnings.warn('Frame {} is non-greyscale and the initial '
                                  'frame was greyscale. This frame will be '
                                  'corrupted.'.format(k))
                if image.shape != frame_shape:  # Valid due to tuple/int
                    warnings.warn('Frame {} is not the same shape as the '
                                  'initial frame and therefore the output '
                                  'may be corrupted.'.format(k))

                i = image.pixels_with_channels_at_back(out_dtype=np.uint8)
                # Handle the case of a greyscale image amidst colour images
                if image.n_channels == 1 and colour == 'rgb24':
                    # Repeat the channels axis 3 times
                    i = i.reshape(i.shape + (1,)).repeat(3, axis=2)
                pipe.stdin.write(i.tostring())
            except IOError:
                error = ('FFMPEG encountered the following error while '
                         'writing the video:\n\n{}'.format(
                    pipe.stderr.read().decode()))
                # Re-raise the error for a useful error message
                raise IOError(error)


def imageio_video_exporter(images, out_path, fps=30, codec='libx264',
                           quality=None, bitrate=None, pixelformat='yuv420p',
                           **kwargs):
    r"""
    Uses imageio to export the images using FFMPEG. Please see the imageio
    documentation for more information.

    Parameters
    ----------
    images : `list` of :map:`Image`
        List of Menpo images to export as a video.
    out_path : `Path`
        Path to save the video to.
    fps : `int`, optional
        The number of frames per second.
    codec : `str`, optional
        The video codec to use. Default 'libx264', which represents the
        widely available mpeg4. Except when saving .wmv files, then the
        defaults is 'msmpeg4' which is more commonly supported for windows
    quality : `float` or `None`
        Video output quality. Uses variable bit rate. Highest
        quality is 10, lowest is 0. Specifying a fixed bitrate using ``bitrate``
        disables this parameter.
    bitrate : `int` or `None`, optional
        Set a constant bitrate for the video encoding. Default is ``None``
        causing ``quality` parameter to be used instead.  Better quality videos
        with smaller file sizes will result from using the ``quality`` variable
        bitrate parameter rather than specifying a fixed bitrate with this
        parameter.
    pixelformat: `str`, optional
        The output video pixel format.
    """
    import imageio

    writer = imageio.get_writer(str(out_path), mode='I', fps=fps,
                                codec=codec, quality=quality, bitrate=bitrate,
                                pixelformat=pixelformat)

    for v in images:
        v = v.pixels_with_channels_at_back(out_dtype=np.uint8)
        writer.append_data(v)
    writer.close()


def image_gif_exporter(images, out_path, fps=30, loop=0, duration=None,
                       **kwargs):
    r"""
    Uses imageio to export the images to a GIF. Please see the imageio
    documentation for more information.

    Parameters
    ----------
    images : `list` of :map:`Image`
        List of Menpo images to export as a video.
    out_path : `Path`
        Path to save the video to.
    fps : `float`, optional
        The number of frames per second. If ``duration`` is not given, the
        duration for each frame is set to 1/fps.
    loop : `int`, optional
        The number of iterations. 0 means loop indefinitely
    duration : `float` or list of `float`, optional
        The duration (in seconds) of each frame. Either specify one value
        that is used for all frames, or one value for each frame.
    """
    import imageio

    writer = imageio.get_writer(str(out_path), mode='I', fps=fps,
                                loop=loop, duration=duration)

    for v in images:
        v = v.pixels_with_channels_at_back(out_dtype=np.uint8)
        writer.append_data(v)
    writer.close()
