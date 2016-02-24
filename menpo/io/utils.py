import os
from pathlib import Path


def _norm_path(filepath):
    r"""
    Uses all the tricks in the book to expand a path out to an absolute one.
    """
    return Path(os.path.abspath(os.path.normpath(
        os.path.expandvars(os.path.expanduser(str(filepath))))))
