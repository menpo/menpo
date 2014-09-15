.. _api-fit-index:

:mod:`menpo.fit`
================

Abstract Classes
----------------

.. toctree::
   :maxdepth: 1

   Fitter

Gradient Descent
----------------

.. toctree::
   :maxdepth: 1

   gradientdescent/GradientDescent
   gradientdescent/RegularizedLandmarkMeanShift

Residuals
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   gradientdescent/residual/Residual
   gradientdescent/residual/Robust
   gradientdescent/residual/Cauchy
   gradientdescent/residual/Fair
   gradientdescent/residual/GemanMcClure
   gradientdescent/residual/Huber
   gradientdescent/residual/L1L2
   gradientdescent/residual/SSD
   gradientdescent/residual/Turkey
   gradientdescent/residual/Welsch

Lucas Kanade
------------

.. toctree::
   :maxdepth: 1

   lucaskanade/LucasKanade

Appearance-Based Fitters
~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   lucaskanade/appearance/AppearanceLucasKanade
   lucaskanade/appearance/AdaptiveForwardAdditive
   lucaskanade/appearance/AdaptiveForwardCompositional
   lucaskanade/appearance/AdaptiveInverseCompositional
   lucaskanade/appearance/AlternatingForwardAdditive
   lucaskanade/appearance/AlternatingForwardCompositional
   lucaskanade/appearance/AlternatingInverseCompositional
   lucaskanade/appearance/ProbabilisticForwardAdditive
   lucaskanade/appearance/ProbabilisticForwardCompositional
   lucaskanade/appearance/ProbabilisticInverseCompositional
   lucaskanade/appearance/ProjectOutForwardAdditive
   lucaskanade/appearance/ProjectOutForwardCompositional
   lucaskanade/appearance/ProjectOutInverseCompositional
   lucaskanade/appearance/SimultaneousForwardAdditive
   lucaskanade/appearance/SimultaneousForwardCompositional
   lucaskanade/appearance/SimultaneousInverseCompositional

Image-Based Fitters
~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   lucaskanade/image/ImageLucasKanade
   lucaskanade/image/ImageForwardAdditive
   lucaskanade/image/ImageForwardCompositional
   lucaskanade/image/ImageInverseCompositional

Residuals
~~~~~~~~~

.. toctree::
   :maxdepth: 1

   lucaskanade/residual/Residual
   lucaskanade/residual/ECC
   lucaskanade/residual/GaborFourier
   lucaskanade/residual/GradientCorrelation
   lucaskanade/residual/GradientImages
   lucaskanade/residual/LSIntensity

Regression
----------

Regressors
~~~~~~~~~~

Classes
^^^^^^^

.. toctree::
   :maxdepth: 1

   regression/Regressor
   regression/NonParametricRegressor
   regression/SemiParametricRegressor
   regression/ParametricRegressor

.. _regression_functions:

Functions
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   regression/regressionfunctions/regression
   regression/regressionfunctions/mlr
   regression/regressionfunctions/mlr_pca
   regression/regressionfunctions/mlr_pca_weights
   regression/regressionfunctions/mlr_svd


Trainers
~~~~~~~~

.. toctree::
   :maxdepth: 1

   regression/NonParametricRegressorTrainer
   regression/RegressorTrainer
   regression/ParametricRegressorTrainer
   regression/SemiParametricRegressorTrainer
   regression/SemiParametricClassifierBasedRegressorTrainer

.. _parametric_features:

Parametric Features
~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   regression/parametricfeatures/appearance
   regression/parametricfeatures/difference
   regression/parametricfeatures/probabilistic
   regression/parametricfeatures/project_out
   regression/parametricfeatures/quadratic_weights
   regression/parametricfeatures/weights
   regression/parametricfeatures/whitened_weights


Fitting Result
--------------

.. toctree::
   :maxdepth: 1

   FittingResult
   ParametricFittingResult
   SemiParametricFittingResult
   NonParametricFittingResult
   TrackingResultList
