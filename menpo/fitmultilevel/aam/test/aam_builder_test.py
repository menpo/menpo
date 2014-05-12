import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

import menpo.io as mio
from menpo.landmark import labeller, ibug_68_trimesh
from menpo.fitmultilevel.aam import AAMBuilder
from menpo.fitmultilevel.featurefunctions import sparse_hog
from menpo.transform import PiecewiseAffine, ThinPlateSplines

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


@raises(ValueError)
def test_feature_type_exception():
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
    # build aam
    aam = AAMBuilder(feature_type=['igo', sparse_hog, None],
                     transform=PiecewiseAffine,
                     trilist=training_images[0].landmarks['ibug_68_trimesh'].
                     lms.trilist,
                     normalization_diagonal=150,
                     n_levels=3,
                     downscale=2,
                     scaled_shape_models=False,
                     max_shape_components=[1, 2, 3],
                     max_appearance_components=[3, 3, 3],
                     boundary=3,
                     interpolator='scipy').build(training_images, group='PTS')
    # test builder
    assert (aam.n_training_images == 2)
    assert (aam.n_levels == 3)
    assert (aam.downscale == 2)
    assert (aam.feature_type[0] == 'igo' and aam.feature_type[2] is None)
    assert (aam.interpolator == 'scipy')
    assert_allclose(np.around(aam.reference_shape.range()), (110., 102.))
    assert (not aam.scaled_shape_models)
    assert (np.all([aam.shape_models[j].n_components == 1
                    for j in range(aam.n_levels)]))
    assert (np.all([aam.appearance_models[j].n_components == 1
                    for j in range(aam.n_levels)]))
    assert_allclose([aam.appearance_models[j].template_instance.n_channels
                     for j in range(aam.n_levels)], (2, 36, 1))
    assert_allclose([aam.appearance_models[j].components.shape[1]
                     for j in range(aam.n_levels)], (13892, 250056, 6946))


def test_aam_2():
    # build aam
    aam = AAMBuilder(feature_type=None,
                     transform=ThinPlateSplines,
                     trilist=None,
                     normalization_diagonal=None,
                     n_levels=2,
                     downscale=1.2,
                     scaled_shape_models=True,
                     max_shape_components=None,
                     max_appearance_components=10,
                     boundary=0,
                     interpolator='scipy').build(training_images, group='PTS')
    # test builder
    assert (aam.n_training_images == 2)
    assert (aam.n_levels == 2)
    assert (aam.downscale == 1.2)
    assert (aam.feature_type[0] is None and aam.feature_type[1] is None)
    assert (aam.interpolator == 'scipy')
    assert_allclose(np.around(aam.reference_shape.range()), (224., 207.))
    assert aam.scaled_shape_models
    assert (np.all([aam.shape_models[j].n_components == 1
                    for j in range(aam.n_levels)]))
    assert (np.all([aam.appearance_models[j].n_components == 1
                    for j in range(aam.n_levels)]))
    assert (np.all([aam.appearance_models[j].template_instance.n_channels == 1
                    for j in range(aam.n_levels)]))
    assert_allclose([aam.appearance_models[j].components.shape[1]
                     for j in range(aam.n_levels)], (20447, 29443))
