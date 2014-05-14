[![Build Status](http://img.shields.io/travis/menpo/menpo.svg?style=flat)](https://travis-ci.org/menpo/menpo)
[![Coverage Status](http://img.shields.io/coveralls/menpo/menpo.svg?style=flat)](https://coveralls.io/r/menpo/menpo)
[![PyPI Release](http://img.shields.io/pypi/v/menpo.svg?style=flat)](https://pypi.python.org/pypi/menpo)

Menpo is a statistical modelling toolkit, providing all the tools 
required to build, fit, visualize, and test generative deformable models like Active Appearance Models, Constrained Local Models, Supervised Descent Method and Morphable Models.

IMPORTANT
=========

Menpo is currently under **agressive development** and is very much a
**work in progress**. Documentation and examples are outdated, docstrings
may be incorrect, tests are missing, and even core ideas are not yet fully
formed. The project is at a pre-alpha stage, and this should be kept in mind
at all times.

Installation
------------
Here in the Menpo team, we are firm believers in making installation as simple as possible. Unfortunately, we are a complex project that relies on satisfying a number of complex 3rd party library dependencies. The default Python packing environment does not make this an easy task. Therefore, we evangelise the use of the conda ecosystem, provided by Anaconda. In order to make things as simple as possible, we suggest that you use conda too! To try and persuade you, go to the [Menpo Wiki](https://github.com/menpo/menpo/wiki) to find installation instructions for all major platforms.

Usage
-----
Menpo makes extensive use of IPython Notebooks to explain functionality of the package. These Notebooks are hosted in the [menpo/menpo-notebooks](https://github.com/menpo/menpo-notebooks) repository. We strongly suggest that after installation you:

1. Download the [latest version of the notebooks](https://github.com/menpo/menpo-notebooks/releases) 
2. Run `ipython notebook`
3. Play around with the notebooks.

Want to get a feel for Menpo without installing anything? You can browse the notebooks straight from the [menpo website](http://www.menpo.io/notebooks.html).


Documentation
-------------
See our documentation on [ReadTheDocs](http://menpo.readthedocs.org)

Testing
-------
We use [nosetests](https://nose.readthedocs.org/en/latest/) for unit tests. You can check our current coverage on [coveralls](https://coveralls.io/r/menpo/menpo).

    >> nosetests

from the top of the repository is all this is required.

