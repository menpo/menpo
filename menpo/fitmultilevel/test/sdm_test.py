from mock import patch
from nose.tools import raises
from StringIO import StringIO

import numpy as np
import menpo.io as mio
from menpo.landmark import labeller, ibug_face_68_trimesh
from menpo.fitmultilevel.sdm import SDMTrainer, SDAAMTrainer
from menpo.feature import sparse_hog, igo, hog, no_op
from menpo.fitmultilevel.clm.classifier import linear_svm_lr
from menpo.fit.regression.regressionfunctions import mlr_svd
from menpo.transform import PiecewiseAffine
from menpo.fitmultilevel.aam import AAMBuilder
from menpo.fitmultilevel.clm import CLMBuilder

# load images
filenames = ['breakingbad.jpg', 'takeo.ppm', 'lenna.png', 'einstein.jpg']
training_images = []
for i in range(4):
    im = mio.import_builtin_asset(filenames[i])
    im.crop_to_landmarks_proportion_inplace(0.1)
    labeller(im, 'PTS', ibug_face_68_trimesh)
    if im.n_channels == 3:
        im = im.as_greyscale(mode='luminosity')
    training_images.append(im)

# Seed the random number generator
np.random.seed(seed=1000)

aam = AAMBuilder(features=sparse_hog,
                 transform=PiecewiseAffine,
                 trilist=training_images[0].landmarks['ibug_face_68_trimesh'].
                 lms.trilist,
                 normalization_diagonal=150,
                 n_levels=3,
                 downscale=1.2,
                 scaled_shape_models=False,
                 max_shape_components=None,
                 max_appearance_components=3,
                 boundary=3).build(training_images, group='PTS')

clm = CLMBuilder(classifier_trainers=linear_svm_lr,
                 features=sparse_hog,
                 normalization_diagonal=100,
                 patch_shape=(5, 5),
                 n_levels=1,
                 downscale=1.1,
                 scaled_shape_models=True,
                 max_shape_components=25,
                 boundary=3).build(training_images, group='PTS')

@raises(ValueError)
def test_features_exception():
    sdm = SDMTrainer(features=[igo, sparse_hog],
                     n_levels=3).train(training_images, group='PTS')


@raises(ValueError)
def test_regression_features_sdmtrainer_exception_1():
    sdm = SDMTrainer(n_levels=2, regression_features=[no_op, no_op, no_op]).\
        train(training_images, group='PTS')


@raises(ValueError)
def test_regression_features_sdmtrainer_exception_2():
    sdm = SDMTrainer(n_levels=3, regression_features=[no_op, sparse_hog, 1]).\
        train(training_images, group='PTS')


@raises(ValueError)
def test_regression_features_sdaamtrainer_exception_1():
    sdm = SDAAMTrainer(aam, regression_features=[no_op, sparse_hog]).\
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
                     features=no_op,
                     normalization_diagonal=150,
                     n_levels=1,
                     downscale=1.3,
                     noise_std=0.04,
                     rotation=False,
                     n_perturbations=2).train(training_images, group='PTS',
                                              verbose=True)
