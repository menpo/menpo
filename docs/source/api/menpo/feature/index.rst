.. _api-feature-index:

:mod:`menpo.feature`
====================

.. toctree::
  :maxdepth: 1


Features
--------

.. toctree::
  :maxdepth: 1

  no_op
  gradient
  gaussian_filter
  igo
  es
  lbp
  hog
  daisy


Optional
--------
The following features are optional and may or may not be available depending
on whether the required packages that implement them are available. If
conda was used to install menpo then it is highly likely that all the optional
packages will be available.

Vlfeat
``````
Features that have been wrapped from the Vlfeat [1]_ project. Currently,
the wrapped features are all variants on the SIFT [2]_ algorithm.

.. toctree::
  :maxdepth: 1

  dsift
  fast_dsift
  vector_128_dsift
  hellinger_vector_128_dsift


Predefined (Partial Features)
-----------------------------
The following features are are built from the features listed above, but are
partial functions. This implies that some sensible parameter choices have
already been made that provides a unique set of properties.

.. toctree::
  :maxdepth: 1

  double_igo
  sparse_hog


Visualization
-------------

.. toctree::
  :maxdepth: 1

  glyph
  sum_channels


Widget
------

.. toctree::
  :maxdepth: 1

  features_selection_widget


References
----------
.. [1] Vedaldi, Andrea, and Brian Fulkerson. "VLFeat: An open and portable
       library of computer vision algorithms." Proceedings of the international
       conference on Multimedia. ACM, 2010.
.. [2] Lowe, David G. "Distinctive image features from scale-invariant
       keypoints." International journal of computer vision 60.2 (2004): 91-110.
