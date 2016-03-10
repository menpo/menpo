def PILExporter(image, file_handle, extension='', **kwargs):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the image data. No value is returned.

    Uses PIL to save the image and so supports most commonly used image
    formats.

    Parameters
    ----------
    image : map:`Image` or subclass
        The image data to write out.
    file_handle : `file`-like object
        The file to write in to
    """
    from PIL.Image import EXTENSION
    # The extensions are only filled out when save or open are called - which
    # may not have been called before we reach here. So let's make sure that
    # pillow is properly initialised.
    if not EXTENSION:
        from PIL.Image import init, preinit
        preinit()
        init()

    pil_image = image.as_PILImage()
    # Also, the format kwarg of PIL/Pillow is a bit confusing and actually
    # refers to the underlying algorithm and not the extension. Therefore,
    # we need to reach into PIL/Pillow and grab the correct format for our
    # given extension.
    try:
        pil_extension = EXTENSION[extension]
    except KeyError:
        raise ValueError('PIL/Pillow does not support the provided '
                         'extension: ({})'.format(extension))
    pil_image.save(file_handle, format=pil_extension)
