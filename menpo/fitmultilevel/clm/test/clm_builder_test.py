from StringIO import StringIO
from mock import patch
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

import menpo.io as mio
from menpo.landmark import labeller, ibug_68_trimesh
from menpo.fitmultilevel.clm import CLMBuilder
from menpo.fitmultilevel.featurefunctions import sparse_hog
from menpo.fitmultilevel.clm.classifierfunctions import linear_svm_lr

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
    labeller(im, 'PTS', ibug_68_trimesh)
    if im.n_channels == 3:
        im = im.as_greyscale(mode='luminosity')
    training_images.append(im)

# build clms
clm1 = CLMBuilder(classifier_type=[linear_svm_lr],
                  patch_shape=(5, 5),
                  feature_type=['igo', sparse_hog, None],
                  normalization_diagonal=150,
                  n_levels=3,
                  downscale=2,
                  scaled_shape_models=False,
                  pyramid_on_features=False,
                  max_shape_components=[1, 2, 3],
                  boundary=3,
                  interpolator='scipy').build(training_images, group='PTS')

clm2 = CLMBuilder(classifier_type=[random_forest, linear_svm_lr],
                  patch_shape=(3, 10),
                  feature_type=None,
                  normalization_diagonal=None,
                  n_levels=2,
                  downscale=1.2,
                  scaled_shape_models=True,
                  pyramid_on_features=False,
                  max_shape_components=None,
                  boundary=0,
                  interpolator='scipy').build(training_images, group='PTS')

clm3 = CLMBuilder(classifier_type=[linear_svm_lr],
                  patch_shape=(2, 3),
                  feature_type='igo',
                  normalization_diagonal=None,
                  n_levels=1,
                  downscale=3,
                  scaled_shape_models=True,
                  pyramid_on_features=True,
                  max_shape_components=[1],
                  boundary=2,
                  interpolator='scipy').build(training_images, group='PTS')


@raises(ValueError)
def test_classifier_type_1_exception():
    clm = CLMBuilder(classifier_type=[linear_svm_lr, linear_svm_lr]).build(
        training_images, group='PTS')

@raises(ValueError)
def test_classifier_type_2_exception():
    clm = CLMBuilder(classifier_type=['linear_svm_lr']).build(
        training_images, group='PTS')

@raises(ValueError)
def test_patch_shape_1_exception():
    clm = CLMBuilder(patch_shape=(5, 1)).build(training_images, group='PTS')

@raises(ValueError)
def test_patch_shape_2_exception():
    clm = CLMBuilder(patch_shape=(5, 6, 7)).build(training_images, group='PTS')

@raises(ValueError)
def test_feature_type_exception():
    clm = CLMBuilder(feature_type=['igo', sparse_hog],
                     pyramid_on_features=False).build(training_images,
                                                      group='PTS')


@raises(ValueError)
def test_feature_type_with_pyramid_on_features_exception():
    clm = CLMBuilder(feature_type=['igo', sparse_hog]).build(training_images,
                                                             group='PTS')


@raises(ValueError)
def test_n_levels_exception():
    clm = CLMBuilder(n_levels=0).build(training_images,
                                       group='PTS')


@raises(ValueError)
def test_downscale_exception():
    clm = CLMBuilder(downscale=1).build(training_images,
                                        group='PTS')
    assert (clm.downscale == 1)
    clm = CLMBuilder(downscale=0).build(training_images,
                                        group='PTS')


@raises(ValueError)
def test_normalization_diagonal_exception():
    clm = CLMBuilder(normalization_diagonal=10).build(training_images,
                                                      group='PTS')


@raises(ValueError)
def test_max_shape_components_1_exception():
    clm = CLMBuilder(max_shape_components=[1, 0.2, 'a']).build(training_images,
                                                               group='PTS')


@raises(ValueError)
def test_max_shape_components_2_exception():
    clm = CLMBuilder(max_shape_components=[1, 2]).build(training_images,
                                                        group='PTS')


@raises(ValueError)
def test_boundary_exception():
    clm = CLMBuilder(boundary=-1).build(training_images, group='PTS')


@patch('sys.stdout', new_callable=StringIO)
def test_verbose_mock(mock_stdout):
    clm = CLMBuilder().build(training_images, group='PTS', verbose=True)


@patch('sys.stdout', new_callable=StringIO)
def test_str_mock(mock_stdout):
    print(clm1)
    print(clm2)
    print(clm3)


def test_clm_1():
    assert (clm1.n_training_images == 4)
    assert (clm1.n_levels == 3)
    assert (clm1.downscale == 2)
    assert (clm1.feature_type[0] == 'igo' and clm1.feature_type[2] is None)
    assert (clm1.interpolator == 'scipy')
    assert_allclose(np.around(clm1.reference_shape.range()), (109., 103.))
    assert (not clm1.scaled_shape_models)
    assert (not clm1.pyramid_on_features)
    assert_allclose(clm1.patch_shape, (5, 5))
    assert_allclose([clm1.shape_models[j].n_components
                     for j in range(clm1.n_levels)], (1, 2, 3))
    assert_allclose(clm1.n_classifiers_per_level, [68, 68, 68])
    assert (clm1.
            classifiers[0][np.random.
            randint(0, clm1.n_classifiers_per_level[0])].__name__
            == 'linear_svm_predict')
    assert (clm1.
            classifiers[1][np.random.
            randint(0, clm1.n_classifiers_per_level[1])].__name__
            == 'linear_svm_predict')
    assert (clm1.
            classifiers[2][np.random.
            randint(0, clm1.n_classifiers_per_level[2])].__name__
            == 'linear_svm_predict')


def test_clm_2():
    assert (clm2.n_training_images == 4)
    assert (clm2.n_levels == 2)
    assert (clm2.downscale == 1.2)
    assert (clm2.feature_type[0] is None and clm2.feature_type[1] is None)
    assert (clm2.interpolator == 'scipy')
    assert_allclose(np.around(clm2.reference_shape.range()), (169., 161.))
    assert clm2.scaled_shape_models
    assert (not clm2.pyramid_on_features)
    assert_allclose(clm2.patch_shape, (3, 10))
    assert (np.all([clm2.shape_models[j].n_components == 3
                    for j in range(clm2.n_levels)]))
    assert_allclose(clm2.n_classifiers_per_level, [68, 68])
    assert (clm2.
            classifiers[0][np.random.
            randint(0, clm2.n_classifiers_per_level[0])].__name__
            == 'random_forest_predict')
    assert (clm2.
            classifiers[1][np.random.
            randint(0, clm2.n_classifiers_per_level[1])].__name__
            == 'linear_svm_predict')


def test_clm_3():
    assert (clm3.n_training_images == 4)
    assert (clm3.n_levels == 1)
    assert (clm3.downscale == 3)
    assert (clm3.feature_type[0] == 'igo' and len(clm3.feature_type) == 1)
    assert (clm3.interpolator == 'scipy')
    assert_allclose(np.around(clm3.reference_shape.range()), (169., 161.))
    assert clm3.scaled_shape_models
    assert clm3.pyramid_on_features
    assert_allclose(clm3.patch_shape, (2, 3))
    assert (np.all([clm3.shape_models[j].n_components == 1
                    for j in range(clm3.n_levels)]))
    assert_allclose(clm3.n_classifiers_per_level, [68])
    assert (clm3.
            classifiers[0][np.random.
            randint(0, clm3.n_classifiers_per_level[0])].__name__
            == 'linear_svm_predict')
