import sys

from setuptools import find_packages, setup


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("menpo")

# Please see conda/meta.yaml for other binary dependencies
install_requires = ["numpy>=1.14", "scipy>=1.0", "matplotlib>=3.0", "pillow>=4.0"]


setup(
    name="menpo",
    version=version,
    cmdclass=cmdclass,
    description="A Python toolkit for handling annotated data",
    author="The Menpo Team",
    author_email="hello@menpo.org",
    packages=find_packages(),
    install_requires=install_requires,
    package_data={"menpo": ["data/*"]},
    tests_require=["pytest>=6.0", "pytest-mock>=3.0", "black>=20.0"],
)
