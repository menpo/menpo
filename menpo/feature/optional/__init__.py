from menpo.base import MenpoMissingDependencyError


try:
    from .vlfeat import (dsift, fast_dsift, vector_128_dsift,
                         hellinger_vector_128_dsift)
except MenpoMissingDependencyError:
    pass

# Remove from scope
del MenpoMissingDependencyError
