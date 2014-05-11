import numpy as np
from numpy.testing import assert_allclose

import menpo.io as mio
from menpo.landmark import labeller, ibug_68_trimesh
from menpo.fitmultilevel.aam import AAMBuilder
from menpo.fitmultilevel.featurefunctions import sparse_hog
from menpo.transform import PiecewiseAffine, ThinPlateSplines


def aam_builder_1_test():
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
    assert (np.all([aam.shape_models[i].n_components == 1
                    for i in range(aam.n_levels)]))
    assert (np.all([aam.appearance_models[i].n_components == 1
                    for i in range(aam.n_levels)]))
    assert_allclose([aam.appearance_models[i].template_instance.n_channels
                     for i in range(aam.n_levels)], (2, 36, 1))
    assert_allclose([aam.appearance_models[i].components.shape[1]
                     for i in range(aam.n_levels)], (13892, 250056, 6946))

def aam_builder_2_test():
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
    assert (np.all([aam.shape_models[i].n_components == 1
                    for i in range(aam.n_levels)]))
    assert (np.all([aam.appearance_models[i].n_components == 1
                    for i in range(aam.n_levels)]))
    assert (np.all([aam.appearance_models[i].template_instance.n_channels == 1
                    for i in range(aam.n_levels)]))
    assert_allclose([aam.appearance_models[i].components.shape[1]
                     for i in range(aam.n_levels)], (20447, 29443))
