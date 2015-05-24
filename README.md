[![Stories in Ready][waffle_shield]][waffle]
[![Build Status][travis_shield]][travis]
[![Coverage Status][coveralls_shield]][coveralls]
[![PyPI Release][pypi_shield]][pypi]
[![BSD License][bsd_shield]][bsd]

[waffle]: https://waffle.io/menpo/menpo
[waffle_shield]: https://badge.waffle.io/menpo/menpo.png?label=ready&title=Ready
[travis]: https://travis-ci.org/menpo/menpo
[travis_shield]: http://img.shields.io/travis/menpo/menpo.svg?style=flat
[coveralls]: https://coveralls.io/r/menpo/menpo
[coveralls_shield]: http://img.shields.io/coveralls/menpo/menpo.svg?style=flat
[pypi]: https://pypi.python.org/pypi/menpo
[pypi_shield]: http://img.shields.io/pypi/v/menpo.svg?style=flat
[bsd]: https://github.com/menpo/menpo/blob/master/LICENSE.txt
[bsd_shield]: http://img.shields.io/badge/License-BSD-green.svg

IMPORTANT
=========
Menpo has been designed for academic use. The project changes quickly as
determined by our research, and this should be kept in mind at all times.

Menpo. A Python toolkit for handling annotated data.
====================================================
What is Menpo?
--------------
Menpo was designed from the ground up to make importing, manipulating and
visualizing image and mesh data as simple as possible. In particular,
we focus on **annotated** data which is common within the fields of Machine
Learning and Computer Vision. All core types are `Landmarkable` and 
visualizing these landmarks is very simple. Since landmarks are first class
citizens within Menpo, it makes tasks like masking images, cropping images
inside landmarks and aligning images very simple.

> Menpo were facial armours which covered all or part of the face and provided 
> a way to secure the top-heavy kabuto (helmet). The Shinobi-no-o (chin cord) 
> of the kabuto would be tied under the chin of the menpo. There were small 
> hooks called ori-kugi or posts called odome located on various places to 
> help secure the kabuto's chin cord.
>
> --- Wikipedia, Menpo

Installation
------------
Here in the Menpo team, we are firm believers in making installation as simple 
as possible. Unfortunately, we are a complex project that relies on satisfying 
a number of complex 3rd party library dependencies. The default Python packing 
environment does not make this an easy task. Therefore, we evangelise the use 
of the conda ecosystem, provided by 
[Anaconda](https://store.continuum.io/cshop/anaconda/). In order to make things 
as simple as possible, we suggest that you use conda too! To try and persuade 
you, go to the [Menpo website](http://www.menpo.io/installation/) to find 
installation instructions for all major platforms.

Usage
-----
Menpo makes extensive use of IPython Notebooks to explain functionality of the 
package. These Notebooks are hosted in the 
[menpo/menpo-notebooks](https://github.com/menpo/menpo-notebooks) repository. 
We strongly suggest that after installation you:

  1. Download the [latest version of the notebooks][notebooks_gh]
  2. Run `ipython notebook`
  3. Play around with the notebooks.

  
[notebooks_gh]: https://github.com/menpo/menpo-notebooks/releases

Want to get a feel for Menpo without installing anything? You can browse the 
notebooks straight from the [menpo website](http://www.menpo.io/notebooks.html).

Other Menpo projects
--------------------
Menpo is designed to be a core library for implementing algorithms within
the Machine Learning and Computer Vision fields. For example, we have developed
a number of more specific libraries that rely on the core components of Menpo:

  - [menpofit][mf_gh]: Implementations of state-of-the-art deformable modelling
    algorithms including Active Appearance Models, Constrained Local Models
    and the Supervised Descent Method.
  - [menpo3d][m3d_gh]: Useful tools for handling 3D mesh data including
    visualization and an OpenGL rasterizer. The requirements of this package
    are complex and really benefit from the use of conda!
  - [menpodetect][md_gh]: A package that wraps existing sources of object 
    detection. The core project is under a BSD license, but since other projects 
    are wrapped, they may not be compatible with this BSD license. Therefore, 
    we urge caution be taken when interacting with this library for 
    non-academic purposes.
  
[mf_gh]: https://github.com/menpo/menpofit
[m3d_gh]: https://github.com/menpo/menpo3d
[md_gh]: https://github.com/menpo/menpodetect

Documentation
-------------
See our documentation on [ReadTheDocs](http://menpo.readthedocs.org)

Testing
-------
We use [nose](https://nose.readthedocs.org/en/latest/) for unit tests. 
You can check our current coverage on 
[coveralls](https://coveralls.io/r/menpo/menpo).

After installing `nose`, running

    >> nosetests .

from the top of the repository will run all of the unit tests. Some tests
check the behavior of viewing functions which should only be importable if the
user also has `menpo3d` installed - if you are running the test suite in an
environment with all of the menpo libraries installed, you will want to exclude
these tests by running

    >> nosetests -a '!viewing' .

instead.
