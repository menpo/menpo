# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Fit an AAM to an Input Images
# ##### Version 0.1
#
# ### 1.1 Load the Data

# <codecell>

from pybug.io import auto_import
from pybug.landmark.labels import ibug_68_points, ibug_68_contour, ibug_68_trimesh, labeller

# load the training images of the LFPW database as landmarked images using the autoimporter
images = auto_import('/vol/atlas/databases/lfpw/train/' + '*.png')


# label the landmarks using the ibug's "standard" 68 points mark-up
labeller(images, 'PTS', ibug_68_points)
labeller(images, 'PTS', ibug_68_contour)
labeller(images, 'PTS', ibug_68_trimesh)

from pybug.shape import PointCloud

points = [img.landmarks['PTS'].all_landmarks.points - 1  for img in images]
shapes = [PointCloud(p) for p in points]

# <codecell>

import pickle

# load a previously buid AAM
aam = pickle.load(open('/vol/atlas/aams/aam_lfpw', "rb"))

template = aam["template"]
template_landmarks = aam["template_landmarks"]
tps_appearance_model = aam["tps_appearance_model"]
shape_model = aam["shape_model"]

# <markdowncell>

# ### 2.1 Build a Statistically Driven + Similarity Transform

# <codecell>

from pybug.transform.affine import SimilarityTransform
import numpy as np

test_shape = shapes[0]
test_image = images[0].as_greyscale()

similarity_transform = SimilarityTransform.estimate(shape_model.mean.points, test_shape.points)

global_weights = similarity_transform.as_vector()
global_weights[1] = 0
global_transform = SimilarityTransform.from_vector(global_weights)

aligned_test_points = global_transform.inverse.apply(test_shape.points)
aligned_test_shape = PointCloud(aligned_test_points)
shape_weights = shape_model.project(aligned_test_shape)

initial_weights = np.concatenate([global_weights, np.zeros_like(shape_weights)])

# <codecell>

from pybug.align.nonrigid.tps import TPS
from pybug.transform.statisticallydriven import StatisticallyDrivenTransform

# function for directly constructing tps transform objects from tps objects
def tps_constructor(src_landmarks, tgt_landmarks):
    tps = TPS(src_landmarks, tgt_landmarks)
    return tps.transform

initial_transform = StatisticallyDrivenTransform(shape_model,
                                                 tps_constructor,
                                                 source=template_landmarks,
                                                 parameters=initial_weights,
                                                 global_transform=global_transform)

# <codecell>

from pybug.warp import scipy_warp

initial_warp = scipy_warp(test_image, template, initial_transform)

temp_tps = TPS(template_landmarks.points, test_shape.points)
temp = scipy_warp(test_image, template, temp_tps.transform)

test_image.add_landmark_set("initial", {"initial": initial_transform.target})
test_image.get_landmark_set("initial").view()

initial_warp.view()
temp.view()

# <codecell>

from pybug.align.lucaskanade.residual import LSIntensity
from pybug.align.lucaskanade.base import ImageForwardAdditive, ImageForwardCompositional

residual = LSIntensity()
lk_algorithm = ImageForwardCompositional(test_image, temp, residual, initial_transform)
optimal_transform = lk_algorithm.align()
