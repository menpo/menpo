from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("./pybug/shape/representation/mesh/cpptrianglemesh.pyx",))
setup(ext_modules = cythonize("./pybug/geodesics/exact.pyx",))
