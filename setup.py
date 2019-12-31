import sys

from setuptools import setup, find_packages

import versioneer

# Please see conda/meta.yaml for other binary dependencies
install_requires = ['numpy>=1.14',
                    'scipy>=1.0',
                    'matplotlib>=3.0',
                    'pillow>=4.0']

if sys.version_info.major == 2:
    install_requires.append('pathlib==1.0')

setup(
    name='menpo',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='A Python toolkit for handling annotated data',
    author='The Menpo Team',
    author_email='hello@menpo.org',
    packages=find_packages(),
    install_requires=install_requires,
    package_data={'menpo': ['data/*']},
    tests_require=['pytest>=5.0', 'mock>=3.0']
)
