def setup(app):
    app.add_stylesheet('rtd_hack.css')

    return {'parallel_read_safe': True, 'parallel_write_safe': True}
