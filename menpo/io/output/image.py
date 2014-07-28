def PILExporter(image, file_handle):
    pil_image = image.as_PILImage()
    pil_image.save(file_handle)
