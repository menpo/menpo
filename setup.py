import os
import platform
import site
import sys

from setuptools import setup, find_packages, Extension

import versioneer

SYS_PLATFORM = platform.system().lower()
IS_LINUX = 'linux' in SYS_PLATFORM
IS_OSX = 'darwin' == SYS_PLATFORM
IS_WIN = 'windows' == SYS_PLATFORM

# Get Numpy include path without importing it
NUMPY_INC_PATHS = [os.path.join(r, 'numpy', 'core', 'include')
                   for r in site.getsitepackages() if
                   os.path.isdir(os.path.join(r, 'numpy', 'core', 'include'))]
if len(NUMPY_INC_PATHS) == 0:
    try:
        import numpy as np
    except ImportError:
        raise ValueError("Could not find numpy include dir and numpy not installed before build - "
                         "cannot proceed with compilation of cython modules.")
    else:
        # just ask numpy for it's include dir
        NUMPY_INC_PATHS = [np.get_include()]

elif len(NUMPY_INC_PATHS) > 1:
    print("Found {} numpy include dirs: "
          "{}".format(len(NUMPY_INC_PATHS), ', '.join(NUMPY_INC_PATHS)))
    print("Taking first (highest precedence on path): {}".format(
        NUMPY_INC_PATHS[0]))
NUMPY_INC_PATH = NUMPY_INC_PATHS[0]


# ---- C/C++ EXTENSIONS ---- #
# Stolen (and modified) from the Cython documentation:
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html
def no_cythonize(extensions, **_ignore):
    import os.path as op
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in ('.pyx', '.py'):
                if extension.language == 'c++':
                    ext = '.cpp'
                else:
                    ext = '.c'
                sfile = path + ext
                if not op.exists(sfile):
                    raise ValueError('Cannot find pre-compiled source file '
                                     '({}) - please install Cython'.format(sfile))
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions


def build_extension_from_pyx(pyx_path, extra_sources_paths=None):
    if extra_sources_paths is None:
        extra_sources_paths = []
    extra_sources_paths.insert(0, pyx_path)
    ext = Extension(name=pyx_path[:-4].replace('/', '.'),
                    sources=extra_sources_paths,
                    include_dirs=[NUMPY_INC_PATH],
                    language='c++')
    if IS_LINUX or IS_OSX:
        ext.extra_compile_args.append('-Wno-unused-function')
    if IS_OSX:
        ext.extra_link_args.append('-headerpad_max_install_names')
    return ext


try:
    from Cython.Build import cythonize
except ImportError:
    import warnings

    cythonize = no_cythonize
    warnings.warn('Unable to import Cython - attempting to build using the '
                  'pre-compiled C++ files.')

cython_modules = [
    build_extension_from_pyx('menpo/external/skimage/_warps_cy.pyx'),
    build_extension_from_pyx(
        'menpo/feature/windowiterator.pyx',
        extra_sources_paths=['menpo/feature/cpp/ImageWindowIterator.cpp',
                             'menpo/feature/cpp/WindowFeature.cpp',
                             'menpo/feature/cpp/HOG.cpp',
                             'menpo/feature/cpp/LBP.cpp']),
    build_extension_from_pyx('menpo/feature/_gradient.pyx'),
    build_extension_from_pyx('menpo/image/patches.pyx')
]
cython_exts = cythonize(cython_modules, quiet=True,
                        language_level=sys.version_info[0])

# Please see conda/meta.yaml for other binary dependencies
install_requires = ['numpy>=1.14',
                    'scipy>=1.0',
                    'matplotlib>=3.0',
                    'pillow>=4.0']

if sys.version_info.major == 2:
    install_requires.append('pathlib==1.0')

setup(name='menpo',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A Python toolkit for handling annotated data',
      author='The Menpo Team',
      author_email='hello@menpo.org',
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=install_requires,
      package_data={'menpo': ['data/*']},
      tests_require=['pytest>=5.0', 'mock>=3.0']
      )
