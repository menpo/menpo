from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("./pybug/spatialdata/mesh/cpptrianglemesh.pyx",))
setup(ext_modules = cythonize("./pybug/transform/geodesics/kirsanov.pyx",))
