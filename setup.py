#from distutils.core import setup
from setuptools import setup, find_packages
from Cython.Build import cythonize
import subprocess
import dependencies
import numpy as np

# grab the submodule and update if required
#subprocess.call("git submodule update --init -q", shell=True)

cython_modules = ["pybug/transform/geodesics/kirsanov.pyx",
                  "pybug/spatialdata/mesh/cpptrianglemesh.pyx"]

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

