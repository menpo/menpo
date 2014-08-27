[![Stories in Ready](https://badge.waffle.io/menpo/menpo.png?label=ready&title=Ready)](https://waffle.io/menpo/menpo)
[![Build Status](http://img.shields.io/travis/menpo/menpo.svg?style=flat)](https://travis-ci.org/menpo/menpo)
[![Coverage Status](http://img.shields.io/coveralls/menpo/menpo.svg?style=flat)](https://coveralls.io/r/menpo/menpo)
[![PyPI Release](http://img.shields.io/pypi/v/menpo.svg?style=flat)](https://pypi.python.org/pypi/menpo)

IMPORTANT
=========

Menpo is very much a **work in progress**. The project is at a 
beta stage, and this should be kept in mind at all times.

Menpo. A deformable modelling toolkit.
======================================
What is Menpo?
--------------
Menpo is a statistical modelling toolkit, providing all the tools 
required to build, fit, visualize, and test deformable models like Active Appearance Models, Constrained Local Models and Supervised Descent Method.

> Menpo were facial armours which covered all or part of the face and provided a way to secure the top-heavy kabuto (helmet). The Shinobi-no-o (chin cord) of the kabuto would be tied under the chin of the menpo. There were small hooks called ori-kugi or posts called odome located on various places to help secure the kabuto's chin cord.
>
> --- Wikipedia, Menpo

Installation
------------
Here in the Menpo team, we are firm believers in making installation as simple as possible. Unfortunately, we are a complex project that relies on satisfying a number of complex 3rd party library dependencies. The default Python packing environment does not make this an easy task. Therefore, we evangelise the use of the conda ecosystem, provided by [Anaconda](https://store.continuum.io/cshop/anaconda/). In order to make things as simple as possible, we suggest that you use conda too! To try and persuade you, go to the [Menpo website](http://www.menpo.io/installation/) to find installation instructions for all major platforms.

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

