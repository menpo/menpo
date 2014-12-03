def PILExporter(image, file_handle):
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
    pil_image = image.as_PILImage()
    pil_image.save(file_handle)
