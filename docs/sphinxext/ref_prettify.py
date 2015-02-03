from sphinx.domains.python import PyXRefRole


def setup(app):
    """
    Any time a python class is referenced, make it a pretty link that doesn't
    include the full package path. This makes the base classes much prettier.
    """
    app.add_role_to_domain('py', 'class', truncate_class_role)
    return {'parallel_read_safe': True}


def truncate_class_role(name, rawtext, text, lineno, inliner,
                        options={}, content=[]):
    if '<' not in rawtext:
        text = '{} <{}>'.format(text.split('.')[-1], text)
        rawtext = ':{}:`{}`'.format(name, text)

    # Return a python x-reference
    py_xref = PyXRefRole()
    return py_xref('py:class', rawtext, text, lineno,
                   inliner, options=options, content=content)
