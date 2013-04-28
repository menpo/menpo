from setuptools import setup, find_packages
from setuptools.extension import Extension
import dependencies

kirsanov_ext = Extension("kirsanov",
                         ["pybug/geodesics/kirsanov.cpp"])
cpptrimesh_ext = Extension("cpptrimesh",
                           ["pybug/shape/mesh/cpptrimesh.cpp"])

setup(name='pybug',
      version='0.1',
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      ext_modules=[kirsanov_ext, cpptrimesh_ext],
      packages=find_packages(),
      install_requires=dependencies.requirements
      )
