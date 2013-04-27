from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np
import dependencies

cython_modules = ["pybug/geodesics/kirsanov.pyx",
                  "pybug/shape/mesh/cpptrimesh.pyx"]

setup(name='pybug',
      version='0.1',
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cythonize(cython_modules, nthreads=2,
                            quiet=True, language='c++'),
      packages=find_packages(),
      install_requires=dependencies.requirements
      )
