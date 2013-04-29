DEPENDENCIES
============

see `dependences.py` for a full list. All of these bar those noted in the
installation section below will be installed for you by `pip install`.
Note however that these packages will have many of their own non-python
dependencies (such as BLAS, or a Fortran compiler) and you will be expected
to have these installed. If you are using Ubuntu, the quickest way of ensuring
you have these base dependencies is to use the `build-dep` feature of
`apt-get`, which grabs all build dependencies for a given Debian package.

For example, running

    sudo apt-get build-dep python-numpy

will install all libraries that numpy requires to build. The libraries will
definitely be sufficient to build the version of numpy that ships with your
version of Ubuntu - and hopefully will be sufficient for the (newer) version
of numpy pip will download for you when installing pybug.


INSTALLATION
============

pybug requires numpy, cython, and the python VTK bindings are present prior to
installation. After that, just run

    pip install -e git+https://github.com/YOURGITHUBACCOUNT/pybug.git#egg=pybug

to install. We highly recommend you do this inside a virtualenv. Take a look
at virtualenvwrapper to make your life easier. Note that we recommend that you
grab a copy of the code from a personal fork of the project onto
YOURGITHUBACCOUNT - this will make issuing changes back to the main repository
much easier.

