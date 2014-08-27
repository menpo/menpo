def PILExporter(file_handle, image):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the image data. No value is returned.

    Uses PIL to save the image and so supports most commonly used image
    formats.

    Parameters
    ----------
    file_handle : `file`-like object
        The file to write in to
    image : map:`Image` or subclass
        The image data to write out.
    """
    pil_image = image.as_PILImage()
    pil_image.save(file_handle)
