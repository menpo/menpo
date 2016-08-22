.. _api-transform-index:

:mod:`menpo.transform`
======================


Composite Transforms
--------------------

.. toctree::
  :maxdepth: 2

  rotate_ccw_about_centre
  scale_about_centre


Homogeneous Transforms
----------------------

.. toctree::
  :maxdepth: 2

  Homogeneous
  Affine
  Similarity
  Rotation
  Translation
  Scale
  UniformScale
  NonUniformScale


Alignments
----------

.. toctree::
  :maxdepth: 2

  ThinPlateSplines
  PiecewiseAffine
  AlignmentAffine
  AlignmentSimilarity
  AlignmentRotation
  AlignmentTranslation
  AlignmentUniformScale


Group Alignments
----------------

.. toctree::
  :maxdepth: 2

  GeneralizedProcrustesAnalysis


Composite Transforms
--------------------

.. toctree::
  :maxdepth: 2

  TransformChain


Radial Basis Functions
----------------------

.. toctree::
  :maxdepth: 2

  R2LogR2RBF
  R2LogRRBF


Abstract Bases
--------------

.. toctree::
  :maxdepth: 2

  Transform
  Transformable
  ComposableTransform
  Invertible
  Alignment
  MultipleAlignment
  DiscreteAffine

Performance Specializations
---------------------------

Mix-ins that provide fast vectorized variants of methods.

.. toctree::
  :maxdepth: 2

  VComposable
  VInvertible
