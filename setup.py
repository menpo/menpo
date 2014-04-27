from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np


# ---- C/C++ EXTENSIONS ---- #
cython_modules = ["menpo/geodesics/kirsanov.pyx",
                  "menpo/shape/mesh/cpptrimesh.pyx",
                  "menpo/shape/mesh/normals.pyx",
                  "menpo/transform/piecewiseaffine/fastpwa.pyx",
                  "menpo/features/cppimagewindowiterator.pyx"]

cython_exts = cythonize(cython_modules, quiet=True)

setup(name='menpo',
      version='0.2.1',
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=[# Core
                        'numpy>=1.8.0',
                        'scipy>=0.12.0',
                        'Cython>=0.20.1',  # req on OS X Mavericks

                        # Image
                        'Pillow>=2.0.0',
                        'scikit-image>=0.8.2',

                        # ML
                        'scikit-learn>=0.14.1',

                        # 3D import
                        'menpo-pyvrml97==2.3.0a4',
                        'cyassimp>=0.1.3',

                        # Rasterization
                        'cyrasterize>=0.1.5',

                        # Visualization
                        'matplotlib>=1.2.1'],
      package_data={'menpo': ['data/*']},
      test_requires=['nose>=1.3.0'],
      extras_require={'3d': 'mayavi>=4.3.0'}
      )
