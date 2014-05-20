from mock import patch
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from StringIO import StringIO

import menpo.io as mio
from menpo.landmark import labeller, ibug_68_trimesh
from menpo.fitmultilevel.sdm import SDMTrainer, SDAAMTrainer, SDCLMTrainer
from menpo.fitmultilevel.featurefunctions import sparse_hog
from menpo.fit.regression.regressionfunctions import mlr, mlr_svd
from menpo.fit.regression.parametricfeatures import weights
from menpo.transform import PiecewiseAffine
from menpo.fitmultilevel.aam import AAMBuilder

# load images
filenames = ['breakingbad.jpg', 'takeo.ppm', 'lenna.png', 'einstein.jpg']
training_images = []
for i in range(4):
    im = mio.import_builtin_asset(filenames[i])
    im.crop_to_landmarks_proportion_inplace(0.1)
    labeller(im, 'PTS', ibug_68_trimesh)
    if im.n_channels == 3:
        im = im.as_greyscale(mode='luminosity')
    training_images.append(im)

# build sdms
sdm1 = SDMTrainer(regression_type=mlr_svd,
                  regression_features=sparse_hog,
                  patch_shape=(16, 16),
                  feature_type=None,
                  normalization_diagonal=150,
                  n_levels=2,
                  downscale=1.3,
                  pyramid_on_features=True,
                  noise_std=0.04,
                  rotation=False,
                  n_perturbations=2,
                  interpolator='scipy').train(training_images, group='PTS')

aam = AAMBuilder(feature_type=sparse_hog,
                 transform=PiecewiseAffine,
                 trilist=training_images[0].landmarks['ibug_68_trimesh'].
                 lms.trilist,
                 normalization_diagonal=150,
                 n_levels=3,
                 downscale=1.2,
                 scaled_shape_models=False,
                 pyramid_on_features=True,
                 max_shape_components=None,
                 max_appearance_components=3,
                 boundary=3,
                 interpolator='scipy').build(training_images, group='PTS')


@raises(ValueError)
def test_feature_type_exception():
    sdm = SDMTrainer(regression_type=mlr_svd,
                     regression_features=sparse_hog,
                     patch_shape=(16, 16),
                     feature_type=['igo', sparse_hog],
                     normalization_diagonal=150,
                     n_levels=3,
                     downscale=1.3,
                     pyramid_on_features=False,
                     noise_std=0.04,
                     rotation=False,
                     n_perturbations=2,
                     interpolator='scipy').train(training_images, group='PTS')


@raises(ValueError)
def test_feature_type_with_pyramid_on_features_exception():
    sdm = SDMTrainer(regression_type=mlr_svd,
                     regression_features=sparse_hog,
                     patch_shape=(16, 16),
                     feature_type=['igo', sparse_hog],
                     normalization_diagonal=150,
                     n_levels=3,
                     downscale=1.3,
                     pyramid_on_features=True,
                     noise_std=0.04,
                     rotation=False,
                     n_perturbations=2,
                     interpolator='scipy').train(training_images, group='PTS')


@raises(ValueError)
def test_regression_features_exception():
    sdm = SDAAMTrainer(aam, regression_features=[None, None]).train(
        training_images, group='PTS')


@raises(ValueError)
def test_n_levels_exception():
    sdm = SDMTrainer(n_levels=0).train(training_images, group='PTS')


@raises(ValueError)
def test_downscale_exception():
    sdm = SDMTrainer(downscale=1).train(training_images,
                                        group='PTS')
    assert (aam.downscale == 1)
    sdm = SDMTrainer(downscale=0).train(training_images,
                                        group='PTS')


@raises(ValueError)
def test_n_perturbations_exception():
    sdm = SDAAMTrainer(aam, n_perturbations=-10).train(training_images,
                                                       group='PTS')


@patch('sys.stdout', new_callable=StringIO)
def test_verbose_mock(mock_stdout):
    sdm = SDMTrainer(regression_type=mlr_svd,
                     regression_features=sparse_hog,
                     patch_shape=(16, 16),
                     feature_type=None,
                     normalization_diagonal=150,
                     n_levels=1,
                     downscale=1.3,
                     pyramid_on_features=True,
                     noise_std=0.04,
                     rotation=False,
                     n_perturbations=2,
                     interpolator='scipy').train(training_images, group='PTS',
                                                 verbose=True)

@patch('sys.stdout', new_callable=StringIO)
def test_str_mock(mock_stdout):
    print sdm1
