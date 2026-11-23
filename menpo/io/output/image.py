def pil_exporter(image, file_handle, extension="", **kwargs):
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
    extension : `str`, optional
        The extension to use for writing if necessary
    """
    from PIL.Image import registered_extensions

    pil_image = image.as_PILImage()
    # The format kwarg of PIL/Pillow is a bit confusing and actually refers to
    # the underlying algorithm and not the extension. Therefore, we need to
    # reach into PIL/Pillow and grab the correct format for our given
    # extension. Pillow only registers the plugins of the formats that it has
    # used so far (a PNG that was imported registers PNG alone), and
    # registered_extensions() loads all of them before it answers.
    try:
        pil_extension = registered_extensions()[extension]
    except KeyError:
        raise ValueError(
            "PIL/Pillow does not support the provided extension: ({})".format(extension)
        )
    pil_image.save(file_handle, format=pil_extension)
