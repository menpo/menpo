
requirements = ['Cython>=0.18',
                'decorator>=3.4.0',
                'ipython>=0.13.2',
                'matplotlib>=1.2.1',
                'mayavi>=4.3.0',
                'nose>=1.3.0',
                'numpy>=1.7.1',
                'Pillow>=2.0.0',
                'pyvrml>=2.4',
                'pyzmq>=13.0.2',
                'scikit-learn>=0.13.1',
                'scipy>=0.12.0',
                'Sphinx>=1.2b1',
                'tornado>=3.0.1']

optionals = {'mlabwrap': 'mlabwrap>=1.2'}

# NOTE: Have to include the egg name in the requirements list as well
repositories = ['https://github.com/patricksnape/pyvrml/tarball/master#egg=pyvrml-2.4',
                'https://github.com/patricksnape/mlabwrap/tarball/master#egg=mlabwrap-1.2']
