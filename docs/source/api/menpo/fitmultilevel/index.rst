.. _api-fitmultilevel-index:

:mod:`menpo.fitmultilevel`
==========================

Base Classes
------------

.. toctree::
   :maxdepth: 1

   MultilevelFitter
   DeformableModelBuilder

Active Appearance Models
------------------------

Builders
^^^^^^^^

.. toctree::
   :maxdepth: 1

   aam/AAM
   aam/AAMBuilder
   aam/build_reference_frame
   aam/PatchBasedAAM
   aam/PatchBasedAAMBuilder
   aam/build_patch_reference_frame

Fitters
^^^^^^^

.. toctree::
   :maxdepth: 1

   aam/AAMFitter
   aam/LucasKanadeAAMFitter

Constrained Local Models
------------------------

.. _clm_builders:

Builders
^^^^^^^^

.. toctree::
   :maxdepth: 1

   clm/CLM
   clm/CLMBuilder
   clm/check_classifier_type
   clm/check_patch_shape
   clm/get_pos_neg_grid_positions

.. _classifier_functions:

Classifier Functions
^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   clm/classifier
   clm/linear_svm_lr

Fitters
^^^^^^^

.. toctree::
   :maxdepth: 1

   clm/CLMFitter
   clm/GradientDescentCLMFitter

Supervised Descent Method
-------------------------

Trainers
^^^^^^^^

.. toctree::
   :maxdepth: 1

   sdm/SDTrainer
   sdm/SDAAMTrainer
   sdm/SDCLMTrainer
   sdm/SDMTrainer

Fitters
^^^^^^^

.. toctree::
   :maxdepth: 1

   sdm/SDFitter
   sdm/SDAAMFitter
   sdm/SDCLMFitter
   sdm/SDMFitter

Fitting Results
---------------

.. toctree::
   :maxdepth: 1

   fittingresult/MultilevelFittingResult
   fittingresult/AAMMultilevelFittingResult

.. _feature_functions:

Features Functions
------------------

.. toctree::
   :maxdepth: 1

   functions/compute_features
   functions/sparse_hog

Utility Functions
-----------------

.. toctree::
   :maxdepth: 1

   functions/align_shape_with_bb
   functions/build_sampling_grid
   functions/compute_error
   functions/extract_local_patches
   functions/mean_pointcloud
   functions/noisy_align
