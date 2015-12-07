import os
import sys
from setuptools import setup, find_packages
import versioneer
import glob


on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    install_requires = []
    ext_modules = []
    include_dirs = []
    cython_exts = []
else:
    from Cython.Build import cythonize
    import numpy as np

    # ---- C/C++ EXTENSIONS ---- #
    cython_modules = ['menpo/shape/mesh/normals.pyx',
                      'menpo/transform/piecewiseaffine/fastpwa.pyx',
                      'menpo/feature/windowiterator.pyx',
                      'menpo/feature/gradient.pyx',
                      'menpo/external/skimage/_warps_cy.pyx',
                      'menpo/image/patches.pyx']

    cython_exts = cythonize(cython_modules, quiet=True)
    include_dirs = [np.get_include()]
    install_requires = ['numpy>=1.10,<1.11',
                        'scipy>=0.16,<0.17',
                        'matplotlib>=1.4,<1.6',
                        'pillow>=2.9,<2.10',
                        'Cython>=0.23,<0.24']

    if sys.version_info.major == 2:
        install_requires.append('pathlib==1.0')

setup(name='menpo',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A Python toolkit for handling annotated data',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=include_dirs,
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=install_requires,
      package_data={'menpo': ['data/*',
                              'feature/cpp/*.cpp',
                              'feature/cpp/*.h',
                              'transform/piecewiseaffine/fastpwa/*.c',
                              'transform/piecewiseaffine/fastpwa/*.h'],
                    '': ['*.pxd', '*.pyx']},
      tests_require=['nose', 'mock==1.0.1']
)
