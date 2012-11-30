from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("./ibugMM/mesh/cppmesh.pyx",
                              include_dirs=["./ibugMM/mesh"]))
