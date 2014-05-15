import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

import menpo.io as mio
from menpo.landmark import labeller, ibug_68_trimesh
from menpo.transform import PiecewiseAffine, ThinPlateSplines
from menpo.fitmultilevel.aam import AAMBuilder
from menpo.fitmultilevel.featurefunctions import sparse_hog

# load images
filenames = ['breakingbad.jpg', 'einstein.jpg']
training_images = []
for i in range(2):
    im = mio.import_builtin_asset(filenames[i])
    im.crop_to_landmarks_proportion(0.1)
    labeller(im, 'PTS', ibug_68_trimesh)
    if im.n_channels == 3:
        im = im.as_greyscale(mode='luminosity')
    training_images.append(im)

# build aams
aam1 = AAMBuilder(feature_type=['igo', sparse_hog, None],
                  transform=PiecewiseAffine,
                  trilist=training_images[0].landmarks['ibug_68_trimesh'].
                  lms.trilist,
                  normalization_diagonal=150,
                  n_levels=3,
                  downscale=2,
                  scaled_shape_models=False,
                  pyramid_on_features=False,
                  max_shape_components=[1, 2, 3],
                  max_appearance_components=[3, 3, 3],
                  boundary=3,
                  interpolator='scipy').build(training_images, group='PTS')

aam2 = AAMBuilder(feature_type=None,
                  transform=ThinPlateSplines,
                  trilist=None,
                  normalization_diagonal=None,
                  n_levels=2,
                  downscale=1.2,
                  scaled_shape_models=True,
                  pyramid_on_features=False,
                  max_shape_components=None,
                  max_appearance_components=10,
                  boundary=0,
                  interpolator='scipy').build(training_images, group='PTS')

aam3 = AAMBuilder(feature_type='igo',
                  transform=ThinPlateSplines,
                  trilist=None,
                  normalization_diagonal=None,
                  n_levels=1,
                  downscale=3,
                  scaled_shape_models=True,
                  pyramid_on_features=True,
                  max_shape_components=[1],
                  max_appearance_components=10,
                  boundary=2,
                  interpolator='scipy').build(training_images, group='PTS')


@raises(ValueError)
def test_feature_type_exception():
    aam = AAMBuilder(feature_type=['igo', sparse_hog],
                     pyramid_on_features=False).build(training_images,
                                                      group='PTS')


@raises(ValueError)
def test_feature_type_with_pyramid_on_features_exception():
    aam = AAMBuilder(feature_type=['igo', sparse_hog]).build(training_images,
                                                             group='PTS')


@raises(ValueError)
def test_n_levels_exception():
    aam = AAMBuilder(n_levels=0).build(training_images,
                                       group='PTS')


@raises(ValueError)
def test_downscale_exception():
    aam = AAMBuilder(downscale=1).build(training_images,
                                        group='PTS')
    assert (aam.downscale == 1)
    aam = AAMBuilder(downscale=0).build(training_images,
                                        group='PTS')


@raises(ValueError)
def test_normalization_diagonal_exception():
    aam = AAMBuilder(normalization_diagonal=100).build(training_images,
                                                       group='PTS')
    assert (aam.appearance_models[0].n_features == 378)
    aam = AAMBuilder(normalization_diagonal=10).build(training_images,
                                                      group='PTS')


@raises(ValueError)
def test_max_shape_components_exception():
    aam = AAMBuilder(max_shape_components=[1, 0.2, 'a']).build(training_images,
                                                               group='PTS')


@raises(ValueError)
def test_max_appearance_components_exception():
    aam = AAMBuilder(max_appearance_components=[1, 2]).build(training_images,
                                                             group='PTS')


@raises(ValueError)
def test_boundary_exception():
    aam = AAMBuilder(boundary=-1).build(training_images, group='PTS')


def test_aam_1():
    assert (aam1.n_training_images == 2)
    assert (aam1.n_levels == 3)
    assert (aam1.downscale == 2)
    assert (aam1.feature_type[0] == 'igo' and aam1.feature_type[2] is None)
    assert (aam1.interpolator == 'scipy')
    assert_allclose(np.around(aam1.reference_shape.range()), (110., 102.))
    assert (not aam1.scaled_shape_models)
    assert (not aam1.pyramid_on_features)
    assert (np.all([aam1.shape_models[j].n_components == 1
                    for j in range(aam1.n_levels)]))
    assert (np.all([aam1.appearance_models[j].n_components == 1
                    for j in range(aam1.n_levels)]))
    assert_allclose([aam1.appearance_models[j].template_instance.n_channels
                     for j in range(aam1.n_levels)], (2, 36, 1))
    assert_allclose([aam1.appearance_models[j].components.shape[1]
                     for j in range(aam1.n_levels)], (13892, 250056, 6946))


def test_aam_2():
    assert (aam2.n_training_images == 2)
    assert (aam2.n_levels == 2)
    assert (aam2.downscale == 1.2)
    assert (aam2.feature_type[0] is None and aam2.feature_type[1] is None)
    assert (aam2.interpolator == 'scipy')
    assert_allclose(np.around(aam2.reference_shape.range()), (224., 207.))
    assert aam2.scaled_shape_models
    assert (not aam2.pyramid_on_features)
    assert (np.all([aam2.shape_models[j].n_components == 1
                    for j in range(aam2.n_levels)]))
    assert (np.all([aam2.appearance_models[j].n_components == 1
                    for j in range(aam2.n_levels)]))
    assert (np.all([aam2.appearance_models[j].template_instance.n_channels == 1
                    for j in range(aam2.n_levels)]))
    assert_allclose([aam2.appearance_models[j].components.shape[1]
                     for j in range(aam2.n_levels)], (20447, 29443))


def test_aam_3():
    assert (aam3.n_training_images == 2)
    assert (aam3.n_levels == 1)
    assert (aam3.downscale == 3)
    assert (aam3.feature_type[0] is 'igo' and len(aam3.feature_type) == 1)
    assert (aam3.interpolator == 'scipy')
    assert_allclose(np.around(aam3.reference_shape.range()), (224., 207.))
    assert aam3.scaled_shape_models
    assert aam3.pyramid_on_features
    assert (np.all([aam3.shape_models[j].n_components == 1
                    for j in range(aam3.n_levels)]))
    assert (np.all([aam3.appearance_models[j].n_components == 1
                    for j in range(aam3.n_levels)]))
    assert (np.all([aam3.appearance_models[j].template_instance.n_channels == 2
                    for j in range(aam3.n_levels)]))
    assert_allclose([aam3.appearance_models[j].components.shape[1]
                     for j in range(aam3.n_levels)], (58886))
