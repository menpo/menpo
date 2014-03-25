import numpy as np
from menpo.groupalign import GeneralizedProcrustesAnalysis
from menpo.shape import PointCloud
from menpo.transform.tps import TPS
from menpo.transform import Translation, Scale
from menpo.rasterize import GLRasterizer
from menpo.rasterize.transform import (ExtractNDims, AddNDims,
                                       CylindricalUnwrapTransform)


def cylindrical_unwrap_and_translation(points):
    from menpo.misctools.circlefit import circle_fit
    from menpo.transform import Translation
    # find the optimum centre to unwrap
    xy = points.points[:, [0, 2]]  # just in the x-z plane
    centre, radius = circle_fit(xy)
    # convert the 2D circle data into the 3D space
    translation = np.array([centre[0], 0, centre[1]])
    centring_transform = Translation(-translation)
    unwrap = CylindricalUnwrapTransform(radius)
    
    def translate_and_unwrap(pc):
        return unwrap.apply(centring_transform.apply(pc))
    
    return translate_and_unwrap


def clip_space_transform(points, boundary_proportion=0.1):
    r"""
    Produces a transform which fits 2D points into the OpenGL
    clipping space ([-1, 1], [-1, 1])
    """
    centering = Translation(points.centre_of_bounds).pseudoinverse
    scale = Scale(points.range() / 2)
    b_scale = Scale(1 - boundary_proportion, ndims=2)
    return centering.compose_before(scale.pseudoinverse).compose_before(b_scale)


def mean_pointcloud(pointclouds):
    return PointCloud(sum(pointclouds) / len(pointclouds))


class MMBuilder(object):

    def __init__(self, models, group=None, label='all'):
        self.models = models
        self.group = group
        self.label = label
        # ra = Rigid Aligned
        self.ra_models, self.ra_mean_lms, self.unwrapper = None, None, None
        print 'Rigidly aligning...'
        self.rigidly_align()
        # u = Unwrapped (infers ra) 
        self.u_models, self.u_mean_lms = None, None
        self.u_2d, self.u_mean_lms_2d = None, None
        print 'Unwrapping and flattening...'
        self.unwrap_and_flatten()
        # w = Warped (infers ra + u)
        self.w_models_2d = None
        print 'Warping...'
        self.warp()
        self.r = GLRasterizer()
        self.shape_images = []
        print 'Rasterizing...'
        self.rasterize()

    def lms_for(self, x):
        return x.landmarks[self.group][self.label].lms

    def rigidly_align(self):
        gpa = GeneralizedProcrustesAnalysis([self.lms_for(m) 
                                             for m in self.models])
        self.ra_models = [t.apply(m) for t, m in zip(gpa.transforms,
                                                     self.models)]
        self.ra_mean_lms = mean_pointcloud([self.lms_for(m).points
                                            for m in self.ra_models])
        self.unwrapper = cylindrical_unwrap_and_translation(self.ra_mean_lms)

    def unwrap_and_flatten(self):
        self.u_models = [self.unwrapper(m) for m in self.ra_models]
        self.u_mean_lms = self.unwrapper(self.ra_mean_lms)
        extract_2d = ExtractNDims(2)
        self.u_2d = [extract_2d.apply(u) for u in self.u_models]
        self.u_mean_lms_2d = extract_2d.apply(self.u_mean_lms)

    def warp(self):
        tps_transforms = [TPS(self.lms_for(u), self.u_mean_lms_2d)
                          for u in self.u_2d]
        self.w_models_2d = [t.apply(u) for t, u in zip(tps_transforms,
                                                       self.u_2d)]

    def rasterize(self):
        trans_to_clip_space = clip_space_transform(self.u_mean_lms_2d)
        cs_w_models_2d = [trans_to_clip_space.apply(m)
                          for m in self.w_models_2d]
        add_nill_z = AddNDims(1)  # adds an all-zero z axis
        cs_w_models = [add_nill_z.apply(m) for m in cs_w_models_2d]

        self.shape_images = []
        for cs_w_model, orig in zip(cs_w_models, self.ra_models):
            blank, shape_image = self.r.rasterize_mesh_with_f3v_interpolant(
                cs_w_model, per_vertex_f3v=orig.points)
            self.shape_images.append(shape_image)
