from mock import patch
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from StringIO import StringIO

import menpo.io as mio
from menpo.landmark import labeller, ibug_68_trimesh
from menpo.fitmultilevel.sdm import SDMTrainer, SDAAMTrainer, SDCLMTrainer
from menpo.transform.modeldriven import OrthoMDTransform
from menpo.transform.homogeneous import AlignmentSimilarity
from menpo.fitmultilevel.featurefunctions import sparse_hog
from menpo.fit.regression.regressionfunctions import mlr, mlr_svd
from menpo.fit.regression.parametricfeatures import weights, warped_image
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

sdm2 = SDAAMTrainer(aam,
                    regression_type=mlr,
                    regression_features=weights,
                    noise_std=0.04,
                    rotation=False,
                    n_perturbations=1,
                    update='compositional',
                    md_transform=OrthoMDTransform,
                    global_transform=AlignmentSimilarity,
                    n_shape=25,
                    n_appearance=None).train(training_images, group='PTS')


@raises(ValueError)
def test_feature_type_exception():
    sdm = SDMTrainer(feature_type=['igo', sparse_hog],
                     n_levels=3).train(training_images, group='PTS')


@raises(ValueError)
def test_feature_type_with_pyramid_on_features_exception():
    sdm = SDMTrainer(feature_type=['igo', sparse_hog, 'hog'],
                     n_levels=3,
                     pyramid_on_features=True).train(training_images,
                                                     group='PTS')


@raises(ValueError)
def test_regression_features_sdmtrainer_exception_1():
    sdm = SDMTrainer(n_levels=2, regression_features=[None, None, None]).\
        train(training_images, group='PTS')


@raises(ValueError)
def test_regression_features_sdmtrainer_exception_2():
    sdm = SDMTrainer(n_levels=3, regression_features=[None, sparse_hog, 1]).\
        train(training_images, group='PTS')


@raises(ValueError)
def test_regression_features_sdaamtrainer_exception_1():
    sdm = SDAAMTrainer(aam, regression_features=[None, sparse_hog]).\
        train(training_images, group='PTS')


@raises(ValueError)
def test_regression_features_sdaamtrainer_exception_2():
    sdm = SDAAMTrainer(aam, regression_features=7).\
        train(training_images, group='PTS')


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
    print sdm2
    
    
def test_sdm_1():
    assert (sdm1._n_training_images == 8)
    assert (sdm1.n_levels == 2)
    assert (sdm1.downscale == 1.3)
    assert (sdm1.feature_type[0] is None)
    assert (sdm1.interpolator == 'scipy')
    assert (sdm1.algorithm == 'SDM-Non-Parametric')
    assert sdm1.pyramid_on_features
    assert (sdm1._fitters[0].algorithm == sdm1._fitters[1].algorithm ==
            'Non-Parametric')
    assert (sdm1._fitters[0].regressor.__name__ ==
            sdm1._fitters[1].regressor.__name__ == 'mlr_svd_fitting')


def test_sdm_2():
    assert (sdm2._n_training_images == 8)
    assert (sdm2.n_levels == 3)
    assert (sdm2.downscale == 1.2)
    assert (sdm2.interpolator == 'scipy')
    assert (sdm2.algorithm == 'SD-AAM-Parametric')
    assert sdm2.pyramid_on_features
    assert (sdm2._fitters[0].algorithm == sdm2._fitters[1].algorithm ==
            sdm2._fitters[2].algorithm == 'Parametric')
    assert (sdm2._fitters[0].regressor.__name__ ==
            sdm2._fitters[1].regressor.__name__ ==
            sdm2._fitters[2].regressor.__name__ == 'mlr_fitting')
