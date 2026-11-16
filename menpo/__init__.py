from . import base

from . import feature
from . import image
from . import io
from . import landmark
from . import math
from . import model
from . import shape
from . import transform
from . import visualize

try:
    from ._version import __version__
except ImportError:  # running from a source tree that has not been built or installed
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        __version__ = _version("menpo")
    except PackageNotFoundError:
        __version__ = "unknown"
