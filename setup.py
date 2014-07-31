import os
import sys
from setuptools import setup, find_packages
import versioneer

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
CUSTOM_FLAGS = []
for arg in sys.argv:
    if arg.startswith("-D"):
        CUSTOM_FLAGS.append(arg)
        sys.argv.remove(arg)

if on_rtd:
    install_requires = []
    ext_modules = []
    include_dirs = []
    cython_exts = []
else:
    import numpy as np
    from Cython.Build import cythonize
    from distutils.extension import Extension
    from cuda_build_ext import locate_cuda, convert_to_cuda_pyx

    CUDA = locate_cuda()

    # Obtain the numpy include directory.
    # This logic works across numpy versions.
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()

    # ---- C/C++ EXTENSIONS ---- #
    cython_modules = [
        "menpo/shape/mesh/normals.pyx",
        "menpo/transform/piecewiseaffine/fastpwa.pyx",
        "menpo/image/feature/cppimagewindowiterator.pyx",
    ]
    # ---- CUDA  EXTENSIONS ---- #
    if CUDA:
        cython_cumodules = [
            "menpo/cuda/cutools.pyx",
            convert_to_cuda_pyx("menpo/image/feature/cppimagewindowiterator.pyx"),
        ]
    else:
        cython_cumodules = []
    
    # Build extensions
    cython_exts = []
    for module in cython_modules:
        cython_exts.append(
                Extension(name=module[:-4],
                          sources=[module],
                          extra_compile_args=CUSTOM_FLAGS))
    for module in cython_cumodules:
        # Every library compiled that way will require the user
        # to have CUDA libraries on its system
        module_ext = Extension(name=module[:-4],
                               sources=[module],
                               library_dirs=[CUDA['lib64']],
                               libraries=['cudart'],
                               language='c++',
                               runtime_library_dirs=[CUDA['lib64']],
                               extra_compile_args={
                                   'gcc': CUSTOM_FLAGS,
                                   'nvcc': [
                                       '-arch=sm_20', '--ptxas-options=-v',
                                       '-c', '--compiler-options', "'-fPIC'"
                                   ] + CUSTOM_FLAGS},
                               include_dirs=[
                                   numpy_include,
                                   CUDA['include'], 'src', 'menpo/cuda/cu',
                               ])
        cython_exts.append(module_ext)
    
    cython_exts = cythonize(cython_exts, quiet=True)
    include_dirs = [numpy_include]
    install_requires = [# Core
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
                        'mayavi>=4.3.0']

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
      package_data={'menpo': ['data/*']},
      tests_require=['nose>=1.3.0', 'mock>=1.0.1']
)
