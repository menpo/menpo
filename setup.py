#from distutils.core import setup
from setuptools import setup, find_packages
from setuptools.extension import Extension
import subprocess

# grab the submodule and update if required
subprocess.call("git submodule update --init -q", shell=True)

cython_modules = ["pybug/transform/geodesics/kirsanov.pyx",
                  "pybug/spatialdata/mesh/cpptrianglemesh.pyx"]

kirsanov_ext = Extension("kirsanov", ["pybug/transform/geodesics/kirsanov.cpp"])
cpptrianglemesh_ext = Extension("cpptrianglemesh", 
        ["pybug/spatialdata/mesh/cpptrianglemesh.cpp"])

setup(name='pybug',
      version = '0.1',
      description = 'iBUG Facial Modelling Toolkit',
      author = 'James Booth',
      author_email = 'james.booth08@imperial.ac.uk',
      ext_modules = [kirsanov_ext, cpptrianglemesh_ext],
      packages = find_packages(),
      install_requires = ['Cython>=0.18', 'numpy>=1.7.1',
          'ipython>=0.13.1', 'Spinx>=1.1.3']
      )

