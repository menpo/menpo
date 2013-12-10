#! /bin/sh

# before this, brew doctor must pass without warnings,
# and XQuartz needs to be installed. If you are unsure
# about what this script does please see README.md!

brew install git boost gfortran python cmake pkg-config readline libpng libjpeg freetype
brew install assimp pyqt

brew tap homebrew/versions
brew tap homebrew/science
brew install vtk5

