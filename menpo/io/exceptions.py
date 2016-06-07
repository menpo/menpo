class MeshImportError(Exception):
    r"""
    Raised when a mesh importer finds an unexpected data format.
    """
    pass


class OverwriteError(ValueError):
    r"""
    Raised when an IO action would lead to the overwriting of an existing file
    without explit intention.
    """
    def __init__(self, message, path, *args):
        self.message = message  # without this you may get DeprecationWarning
        self.path = path
        # allow users initialize misc. arguments as any other builtin Error
        super(OverwriteError, self).__init__(message, *args)
