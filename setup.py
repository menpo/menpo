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
                      'menpo/image/extract_patches.pyx']

    cython_exts = cythonize(cython_modules, quiet=True)
    include_dirs = [np.get_include()]
    install_requires = ['numpy>=1.9.1,<1.10',
                        'scipy>=0.15,<0.16',
                        'matplotlib>=1.4,<1.5',
                        'pillow>=2.8,<2.9',
                        'Cython>=0.22,<0.23']

    if sys.version_info.major == 2:
        install_requires.append('pathlib==1.0')

# Explicitly specify the image/landmark data in the data folder
builtin_data = filter(lambda x: os.path.isfile(x), glob.glob('menpo/data/*'))
builtin_data = [os.path.relpath(x, start='menpo') for x in builtin_data]

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
      package_data={'menpo': builtin_data + [
                             'data/logos/*',
                             'feature/cpp/*.cpp',
                             'feature/cpp/*.h',
                             'transform/piecewiseaffine/fastpwa/*.c',
                             'transform/piecewiseaffine/fastpwa/*.h'],
                    '': ['*.pxd', '*.pyx']},
      tests_require=['nose', 'mock']
)
