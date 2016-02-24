

def ImageioVideoExporter(images, out_path, fps=30, codec='libx264',
                         quality=None, bitrate=None, pixelformat='yuv420p',
                         **kwargs):
    r"""
    Uses imageio to export the images using FFMPEG. Please see the imageio
    documentation for more information.

    Parameters
    ----------
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
        v = v.as_imageio()
        writer.append_data(v)
    writer.close()


def ImageioGifExporter(images, out_path, fps=30, loop=0, duration=None,
                       **kwargs):
    r"""
    Uses imageio to export the images to a GIF. Please see the imageio
    documentation for more information.

    Parameters
    ----------
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
        v = v.as_imageio()
        writer.append_data(v)
    writer.close()
