def setup(app):
    app.add_css_file("rtd_hack.css")

    return {"parallel_read_safe": True, "parallel_write_safe": True}
