from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np
import dependencies
from pybug.rasterize.c.shaders import build_c_shaders

build_c_shaders()

cpp_cython_modules = ["pybug/geodesics/kirsanov.pyx",
                  "pybug/shape/mesh/cpptrimesh.pyx",
                  "pybug/io/mesh/assimp.pyx"]
c_cython_modules = ["pybug/rasterize/opengl.pyx"]

cpp_exts = cythonize(cpp_cython_modules, nthreads=2, quiet=True,
                     language='c++')
c_exts = cythonize(c_cython_modules, nthreads=2, quiet=True, language='c')
for c_ext in c_exts:
    c_ext.extra_compile_args += ['-std=c99']

setup(name='pybug',
      version='0.2',
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cpp_exts + c_exts,
      packages=find_packages(),
      install_requires=dependencies.requirements,
      dependency_links=dependencies.repositories
      )
