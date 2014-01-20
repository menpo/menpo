from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np
from pybug.rasterize.c.shaders import build_c_shaders

build_c_shaders()

cpp_cython_modules = ["pybug/geodesics/kirsanov.pyx",
                  "pybug/shape/mesh/cpptrimesh.pyx",
                  "pybug/shape/mesh/normals.pyx",
                  "pybug/io/mesh/assimp.pyx",
                  "pybug/interpolation/cinterp.pyx",
                  "pybug/transform/fastpwa.pyx"]
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
      install_requires=['Cython>=0.18',
                        'decorator>=3.4.0',
                        'ipython>=0.13.2',
                        'matplotlib>=1.2.1',
                        'nose>=1.3.0',
                        'numpy>=1.7.1',
                        'Pillow>=2.0.0',
                        'pyvrml>=2.4',
                        'pyzmq>=13.0.2',
                        'scikit-learn>=0.13.1',
                        'scikit-image>=0.8.2',
                        'scipy>=0.12.0',
                        'Sphinx>=1.2b1',
                        'numpydoc>=0.4',
                        'tornado>=3.0.1'],
      extras_require={'3d': 'mayavi>=4.3.0'},
      dependency_links=[
        'https://github.com/patricksnape/pyvrml/tarball/master#egg=pyvrml-2.4']
      )

# NOTE: Have to include the egg name in the dependency_links as well
