from StringIO import StringIO
from mock import patch
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

import menpo.io as mio
from menpo.landmark import labeller, ibug_face_68_trimesh
from menpo.fitmultilevel.clm import CLMBuilder
from menpo.feature import sparse_hog, igo, no_op
from menpo.fitmultilevel.clm.classifier import linear_svm_lr
from menpo.fitmultilevel.base import name_of_callable

from sklearn import qda


def random_forest(X, t):
    clf = qda.QDA()
    clf.fit(X, t)

    def random_forest_predict(x):
        return clf.predict_proba(x)[:, 1]

    return random_forest_predict

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

# build clms
clm1 = CLMBuilder(classifier_trainers=[linear_svm_lr],
                  patch_shape=(5, 5),
                  features=[igo, sparse_hog, no_op],
                  normalization_diagonal=150,
                  n_levels=3,
                  downscale=2,
                  scaled_shape_models=False,
                  max_shape_components=[1, 2, 3],
                  boundary=3).build(training_images, group='PTS')

clm2 = CLMBuilder(classifier_trainers=[random_forest, linear_svm_lr],
                  patch_shape=(3, 10),
                  features=[no_op, no_op],
                  normalization_diagonal=None,
                  n_levels=2,
                  downscale=1.2,
                  scaled_shape_models=True,
                  max_shape_components=None,
                  boundary=0).build(training_images, group='PTS')

clm3 = CLMBuilder(classifier_trainers=[linear_svm_lr],
                  patch_shape=(2, 3),
                  features=igo,
                  normalization_diagonal=None,
                  n_levels=1,
                  downscale=3,
                  scaled_shape_models=True,
                  max_shape_components=[1],
                  boundary=2).build(training_images, group='PTS')


@raises(ValueError)
def test_classifier_type_1_exception():
    CLMBuilder(classifier_trainers=[linear_svm_lr, linear_svm_lr]).build(
        training_images, group='PTS')

@raises(ValueError)
def test_classifier_type_2_exception():
    CLMBuilder(classifier_trainers=['linear_svm_lr']).build(training_images,
                                                        group='PTS')

@raises(ValueError)
def test_patch_shape_1_exception():
    CLMBuilder(patch_shape=(5, 1)).build(training_images, group='PTS')

@raises(ValueError)
def test_patch_shape_2_exception():
    CLMBuilder(patch_shape=(5, 6, 7)).build(training_images, group='PTS')

@raises(ValueError)
def test_features_exception():
    CLMBuilder(features=[igo, sparse_hog]).build(training_images, group='PTS')

@raises(ValueError)
def test_n_levels_exception():
    clm = CLMBuilder(n_levels=0).build(training_images, group='PTS')


@raises(ValueError)
def test_downscale_exception():
    clm = CLMBuilder(downscale=1).build(training_images, group='PTS')
    assert (clm.downscale == 1)
    CLMBuilder(downscale=0).build(training_images, group='PTS')


@raises(ValueError)
def test_normalization_diagonal_exception():
    CLMBuilder(normalization_diagonal=10).build(training_images,
                                                group='PTS')


@raises(ValueError)
def test_max_shape_components_1_exception():
    CLMBuilder(max_shape_components=[1, 0.2, 'a']).build(training_images,
                                                         group='PTS')


@raises(ValueError)
def test_max_shape_components_2_exception():
    CLMBuilder(max_shape_components=[1, 2]).build(training_images,
                                                  group='PTS')


@raises(ValueError)
def test_boundary_exception():
    CLMBuilder(boundary=-1).build(training_images, group='PTS')


@patch('sys.stdout', new_callable=StringIO)
def test_verbose_mock(mock_stdout):
    CLMBuilder().build(training_images, group='PTS', verbose=True)


@patch('sys.stdout', new_callable=StringIO)
def test_str_mock(mock_stdout):
    print(clm1)
    print(clm2)
    print(clm3)


def test_clm_1():
    assert (clm1.n_training_images == 4)
    assert (clm1.n_levels == 3)
    assert (clm1.downscale == 2)
    #assert (clm1.features[0] == igo and clm1.features[2] is no_op)
    assert_allclose(np.around(clm1.reference_shape.range()), (109., 103.))
    assert (not clm1.scaled_shape_models)
    assert (not clm1.pyramid_on_features)
    assert_allclose(clm1.patch_shape, (5, 5))
    assert_allclose([clm1.shape_models[j].n_components
                     for j in range(clm1.n_levels)], (1, 2, 3))
    assert_allclose(clm1.n_classifiers_per_level, [68, 68, 68])

    ran_0 = np.random.randint(0, clm1.n_classifiers_per_level[0])
    ran_1 = np.random.randint(0, clm1.n_classifiers_per_level[1])
    ran_2 = np.random.randint(0, clm1.n_classifiers_per_level[2])

    assert (name_of_callable(clm1.classifiers[0][ran_0])
            == 'linear_svm_lr')
    assert (name_of_callable(clm1.classifiers[1][ran_1])
            == 'linear_svm_lr')
    assert (name_of_callable(clm1.classifiers[2][ran_2])
            == 'linear_svm_lr')


def test_clm_2():
    assert (clm2.n_training_images == 4)
    assert (clm2.n_levels == 2)
    assert (clm2.downscale == 1.2)
    #assert (clm2.features[0] is no_op and clm2.features[1] is no_op)
    assert_allclose(np.around(clm2.reference_shape.range()), (169., 161.))
    assert clm2.scaled_shape_models
    assert (not clm2.pyramid_on_features)
    assert_allclose(clm2.patch_shape, (3, 10))
    assert (np.all([clm2.shape_models[j].n_components == 3
                    for j in range(clm2.n_levels)]))
    assert_allclose(clm2.n_classifiers_per_level, [68, 68])

    ran_0 = np.random.randint(0, clm2.n_classifiers_per_level[0])
    ran_1 = np.random.randint(0, clm2.n_classifiers_per_level[1])

    assert (name_of_callable(clm2.classifiers[0][ran_0])
            == 'random_forest_predict')
    assert (name_of_callable(clm2.classifiers[1][ran_1])
            == 'linear_svm_lr')


def test_clm_3():
    assert (clm3.n_training_images == 4)
    assert (clm3.n_levels == 1)
    assert (clm3.downscale == 3)
    #assert (clm3.features[0] == igo and len(clm3.features) == 1)
    assert_allclose(np.around(clm3.reference_shape.range()), (169., 161.))
    assert clm3.scaled_shape_models
    assert clm3.pyramid_on_features
    assert_allclose(clm3.patch_shape, (2, 3))
    assert (np.all([clm3.shape_models[j].n_components == 1
                    for j in range(clm3.n_levels)]))
    assert_allclose(clm3.n_classifiers_per_level, [68])
    ran_0 = np.random.randint(0, clm3.n_classifiers_per_level[0])

    assert (name_of_callable(clm3.classifiers[0][ran_0])
            == 'linear_svm_lr')
