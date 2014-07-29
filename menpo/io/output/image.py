def PILExporter(file_handle, image):
    pil_image = image.as_PILImage()
    pil_image.save(file_handle)
