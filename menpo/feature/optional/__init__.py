from menpo.base import MenpoMissingDependencyError


try:
    from .vlfeat import dsift
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
