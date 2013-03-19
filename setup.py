from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("./ibugMM/geodesics/exact.pyx",))
setup(ext_modules = cythonize("./ibugMM/mesh/cpptrianglemesh.pyx",))
