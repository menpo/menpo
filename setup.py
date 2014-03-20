import sys
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

from buildhelpers.shaders import build_c_shaders


# ---- C/C++ EXTENSIONS ---- #
cython_modules = ["menpo/geodesics/kirsanov.pyx",
                  "menpo/shape/mesh/cpptrimesh.pyx",
                  "menpo/shape/mesh/normals.pyx",
                  "menpo/io/mesh/assimp.pyx",
                  "menpo/interpolation/cinterp.pyx",
                  "menpo/transform/fastpwa.pyx",
		  "menpo/features/cppimagewindowiterator.pyx"]

cython_exts = cythonize(cython_modules, nthreads=2, quiet=True)


# ---- OPENGL C EXTENSIONS ---- #
# first, convert the plain text shaders into C string literals
build_c_shaders()

opengl_c_cython_modules = ["menpo/rasterize/copengl.pyx"]
opengl_c_exts = cythonize(opengl_c_cython_modules, nthreads=2, quiet=True)

# unfortunately, OpenGL is just different on OS X/Linux
if sys.platform.startswith('linux'):
    for c_ext in opengl_c_exts:
        c_ext.libraries += ['GL', 'GLU', 'glfw']
elif sys.platform == 'darwin':
    for c_ext in opengl_c_exts:
        c_ext.libraries += ['glfw3']
        # TODO why does it compile without these on OS X?!
        #c_ext.extra_compile_args += ['-framework OpenGL',
        #                             '-framework Cocoa', '-framework IOKit',
        #                             '-framework CoreVideo']

setup(name='menpo',
      version='0.2',
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cython_exts + opengl_c_exts,
      packages=find_packages(),
      install_requires=[# Core
                        'numpy>=1.8.0',
                        'scipy>=0.12.0',
                        'Cython>=0.18',

                        # Image
                        'Pillow>=2.0.0',
                        'scikit-image>=0.8.2',

                        # 3D import
                        'menpo-pyvrml97',

                        # Visualization
                        'matplotlib>=1.2.1',

                        # Need to decide if this is really needed
                        'decorator>=3.4.0',

                        # Docs and testing
                        'Sphinx>=1.2b1',
                        'numpydoc>=0.4',
                        'nose>=1.3.0'],
      extras_require={'3d': 'mayavi>=4.3.0'}
      )


