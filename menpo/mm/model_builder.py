import numpy as np
from menpo.groupalign import GeneralizedProcrustesAnalysis
from menpo.shape import PointCloud
from menpo.transform.base import Transform
from menpo.transform.tps import TPS
from menpo.transform import Translation, Scale
from menpo.rasterize import GLRasterizer


class CylindricalUnwrapTransform(Transform):
    r"""
    Unwraps 3D points into cylindrical coordinates:
    x -> radius * theta
    y -> z
    z -> depth

    The cylinder is oriented st. it's axial vector is [0, 1, 0]
    and it's centre is at the origin. discontinuity in theta values
    occurs at y-z plane for NEGATIVE z values (i.e. the interesting
    information you are wanting to unwrap better have positive z values).

    radius - the distance of the unwrapping from the axis.
    z -  the distance along the axis of the cylinder (maps onto the y
         coordinate exactly)
    theta - the angular distance around the cylinder, in radians. Note
         that theta itself is not outputted but theta * radius, preserving
         distances.

    depth - is the displacement away from the radius along the radial vector.
    """
    def __init__(self, radius):
        self.radius = radius

    def _apply(self, x, **kwargs):
        cy_coords = np.zeros_like(x)
        depth = np.sqrt(x[:, 0]**2 + x[:, 2]**2) - self.radius
        theta = np.arctan2(x[:, 0], x[:, 2])
        z = x[:, 1]
        cy_coords[:, 0] = theta * self.radius
        cy_coords[:, 1] = z
        cy_coords[:, 2] = depth
        return cy_coords
    
    
class Extract2D(Transform):
    r"""
    Extracts out the x-y dim
    """
    def __init__(self):
        pass
    
    def _apply(self, x, **kwargs):
        return x[:, :2].copy()
    
    
class AddNillZ(Transform):
    r"""
    Adds a z axis of all zeros
    """
    
    def _apply(self, x, **kwargs):
        return np.hstack([x, np.zeros([x.shape[0], 1])]).copy()
        
    
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


def clip_space_transform(points):
    r"""
    Produces a transform which fits 2D points into the OpenGL
    clipping space ([-1, 1], [-1, 1])
    """
    centering = Translation(-points.centre)
    centred = centering.apply(points)
    scale = Scale(centred.range() / 2)
    return centering.compose_before(scale.pseudoinverse)


def mean_pointcloud(pointclouds):
    PointCloud(sum(pointclouds) / len(pointclouds))


class MMBuilder(object):

    def __init__(self, models, group=None, label='all'):
        self.models = models
        self.group = group
        self.label = label
        self.ra_models, self.ra_mean_lms, self.unwrapper = None
        self.rigidly_align()
        self.u_models, self.u_mean_lms, self.u_2d, self.u_mean_lms_2d = None
        self.unwrap_and_flatten()
        self.wu_models_2d = None
        self.warp()
        self.r = GLRasterizer()
        self.shape_images = []
        self.rasterize()

    def lms_for(self, x):
        return x[self.group][self.label]

    def rigidly_align(self):
        gpa = GeneralizedProcrustesAnalysis(
            [m.landmarks[self.group][self.label].lms
             for m in self.models])
        self.ra_models = [t.apply(m) for t, m in zip(gpa.transforms,
                                                     self.models)]
        self.ra_mean_lms = mean_pointcloud(self.lms_for(m).points
                                           for m in self.ra_models)
        self.unwrapper = cylindrical_unwrap_and_translation(self.ra_mean_lms)

    def unwrap_and_flatten(self):
        self.u_models = [self.unwrapper(m) for m in self.ra_models]
        self.u_mean_lms = self.unwrapper(self.ra_mean_lms)
        extract_2d = Extract2D()
        self.u_2d = [extract_2d.apply(u) for u in self.u_models]
        self.u_mean_lms_2d = extract_2d.apply(self.u_mean_lms)

    def warp(self):
        tps_transforms = [TPS(self.lms_for(u), self.u_mean_lms_2d)
                          for u in self.u_2d]
        self.wu_models_2d = [t.apply(u) for t, u in zip(tps_transforms,
                                                        self.wu_models_2d)]

    def rasterize(self):
        trans_to_clip_space = clip_space_transform(self.u_mean_lms_2d)
        cs_wu_models_2d = [trans_to_clip_space.apply(m)
                           for m in self.wu_models_2d]
        add_nill_z = AddNillZ()  # adds an all-zero z axis
        cs_wu_models = [add_nill_z.apply(m) for m in cs_wu_models_2d]

        self.shape_images = []
        for cs_wu_model, orig in zip(cs_wu_models, self.ra_models):
            blank, shape_image = self.r.rasterize_mesh_with_f3v_interpolant(
                cs_wu_model, per_vertex_f3v=orig.points)
            self.shape_images.append(shape_image)
