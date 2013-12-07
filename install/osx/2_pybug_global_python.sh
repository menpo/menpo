#! /bin/sh

# OS X PYTHON SETUP SCRIPT
#
# Before running this, `1_pybug_brew.sh` must have been installed
# without issue. In particular, check `brew doctor` passes before
# proceeding!
# If you are at all unsure what this means please see README.md!

pip install --upgrade pip setuptools virtualenvwrapper
pip install --upgrade 'ipython[notebook]' cython numpy scipy matplotlib pillow sphinx numpydoc

