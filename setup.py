import os
import sys
from setuptools import setup, find_packages
import versioneer


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
                      'menpo/external/skimage/_warps_cy.pyx',
                      'menpo/image/extract_patches.pyx']

    cython_exts = cythonize(cython_modules, quiet=True)
    include_dirs = [np.get_include()]
    install_requires = ['numpy>=1.9.1,<1.10',
                        'scipy>=0.15,<0.16',
                        'matplotlib>=1.4,<1.5',
                        'pillow==2.7.0',
                        'Cython>=0.21,<0.22']

    if sys.version_info.major == 2:
        install_requires.append('pathlib==1.0')

# Versioneer allows us to automatically generate versioning from
# our git tagging system which makes releases simpler.
versioneer.VCS = 'git'
versioneer.versionfile_source = 'menpo/_version.py'
versioneer.versionfile_build = 'menpo/_version.py'
versioneer.tag_prefix = 'v'  # tags are like v1.2.0
versioneer.parentdir_prefix = 'menpo-'  # dirname like 'menpo-v1.2.0'

setup(name='menpo',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='iBUG Facial Modelling Toolkit',
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
      tests_require=['nose', 'mock']
)
