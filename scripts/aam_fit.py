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
images = auto_import('/vol/atlas/databases/lfpw/train/image_0001.png')

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

from pybug.model.linear import SimilarityModel

test_shape = shapes[0]
test_image = images[0].as_greyscale()

similarity_model = SimilarityModel(shape_model.mean)

similarity_weights = similarity_model.project(test_shape)
# remove rotation
similarity_weights[1] = 0

similarity_transform = similarity_model.equivalent_similarity_transform(similarity_weights)
aligned_test_points = similarity_transform.inverse.apply(test_shape.points)
aligned_test_shape = PointCloud(aligned_test_points)
shape_weights = shape_model.project(aligned_test_shape)

import numpy as np

initial_weights = np.concatenate([similarity_weights, np.zeros_like(shape_weights)])

# <codecell>

from pybug.align.nonrigid.tps import TPS
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.model.linear import SimilarityModel
from pybug.transform.statisticallydriven import StatisticallyDrivenPlusSimilarityTransform

# function for directly constructing tps transform objects from tps objects
def tps_constructor(src_landmarks, tgt_landmarks):
    tps = TPS(src_landmarks, tgt_landmarks)
    return tps.transform

initial_transform = StatisticallyDrivenPlusSimilarityTransform(shape_model,
                                                       similarity_model,
                                                       tps_constructor,
                                                       source=template_landmarks.points,
                                                       weights=initial_weights)

# <codecell>

from pybug.warp import scipy_warp

initial_warp = scipy_warp(test_image, template, initial_transform)

temp_tps = TPS(template_landmarks.points, test_shape.points)
temp = scipy_warp(test_image, template, temp_tps.transform)

test_image.add_landmark_set("initial", {"initial": PointCloud(initial_transform.target)})
test_image.get_landmark_set("initial").view()

initial_warp.view()
temp.view() 

# <codecell>

from pybug.align.lucaskanade.residual import LSIntensity
from pybug.align.lucaskanade.base import ImageForwardAdditive

residual = LSIntensity()
project_out = ImageForwardAdditive(test_image, temp, residual, initial_transform)
optimal_transform = project_out.align(max_iters=50)

# <codecell>

fitted_shape = PointCloud(optimal_transform.target)
fitted_appearance = scipy_warp(test_image, template, optimal_transform)
test_image.add_landmark_set("fitting_result", {"fitting_result": fitted_shape})
test_image.get_landmark_set("fitting_result").view()
fitted_appearance.view()

# <codecell>

