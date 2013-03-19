from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("./pybug/geodesics/exact.pyx",))
setup(ext_modules = cythonize("./pybug/mesh/cpptrianglemesh.pyx",))
