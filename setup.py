from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np
import versioneer


# Versioneer allows us to automatically generate versioning from
# our git tagging system which makes releases simpler.  
versioneer.VCS = 'git'
versioneer.versionfile_source = 'menpo/_version.py'
versioneer.versionfile_build = 'menpo/_version.py'
versioneer.tag_prefix = 'v'             # tags are like v1.2.0
versioneer.parentdir_prefix = 'menpo-'  # dirname like 'menpo-v1.2.0'

# ---- C/C++ EXTENSIONS ---- #
cython_modules = ["menpo/shape/mesh/normals.pyx",
                  "menpo/transform/piecewiseaffine/fastpwa.pyx",
                  "menpo/image/feature/cppimagewindowiterator.pyx"]

cython_exts = cythonize(cython_modules, quiet=True)

setup(name='menpo',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='iBUG Facial Modelling Toolkit',
      author='James Booth',
      author_email='james.booth08@imperial.ac.uk',
      include_dirs=[np.get_include()],
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=[# Core
                        'numpy>=1.8.0',
                        'scipy>=0.14.0',
                        'Cython>=0.20.1',

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
                        'matplotlib>=1.2.1',
                        'mayavi>=4.3.0'],
      package_data={'menpo': ['data/*']},
      tests_require=['nose>=1.3.0'],
      )

