import numpy as np
from scipy.spatial import distance
from scipy.stats import norm
from pybug.shape.collection import TriMeshShapeClass
from geodesics import TriMeshGeodesics


class GeodesicMasker(TriMeshShapeClass):
    """A trimesh class that can produce masked versions of its members
    as judged by a geodesic distance from a set of landmark points.
    """
    def __init__(self, trimeshiter):
        TriMeshShapeClass.__init__(self, trimeshiter)

    def geodesics_from(self, label, distance):
        """Returns
        """
        result = []
        for x in self.data:
            x.geodesics = TriMeshGeodesics(x.points, x.trilist)
            indexes = [lm.index for lm in x.landmarks.with_label(label)]
            print indexes
            phi = x.geodesics.geodesics(indexes)['phi']
            result.append(x.new_trimesh(pointmask=(phi < distance)))
        return result


class GeodesicCorrespondence(TriMeshShapeClass):

    def __init__(self, trimeshiter):
        TriMeshShapeClass.__init__(self, trimeshiter)
        for x in self.data:
            x.geodesics = TriMeshGeodesics(x.points, x.trilist)
        self.gsig_all_points()
        self.f0 = self.data[0]
        self.f1 = self.data[1]
        self.calculate_mapping()
        self.generate_mapped_faces()
        #self.calculate_mapped_faces()

    def gsig_all_points(self):
        for x in self.data:
            gsig = np.empty([x.n_points, x.landmarks.n_points])
            for i, lm in enumerate(x.landmarks.reference_landmarks()):
                geodesic = x.geodesics.geodesics(lm.index)
                gsig[:,i] = geodesic['phi']
            x.add_pointfield('gsig', gsig)

    def calculate_mapping(self):
        min_1_to_0, min_0_to_1 = linear_geodesic_mapping(
                self.f0.pointfields['gsig'],
                self.f1.pointfields['gsig'])
        self.f0.add_pointfield(self.f1, min_1_to_0)
        self.f1.add_pointfield(self.f0, min_0_to_1)

    def generate_mapped_faces(self):
        self.face_1_on_0 = new_face_from_mapping(self.f0, self.f1)
        self.face_0_on_1 = new_face_from_mapping(self.f1, self.f0)

def linear_geodesic_mapping(phi_1, phi_2):
    distances = distance.cdist(phi_1, phi_2)
    min_2_to_1 = np.argmin(distances, axis=1)
    min_1_to_2 = np.argmin(distances, axis=0)
    return min_2_to_1, min_1_to_2

def weighted_geodesic_mapping(phi_1, phi_2, std_dev=40):
    # weighting_1 is how influential each value of each phi vector is considered by the first face
    weightings_1 = gaussian_weighting(phi_1, std_dev)
    weightings_2 = gaussian_weighting(phi_2, std_dev)
    differences = phi_1[:,np.newaxis,:] - phi_2
    # multiply by the first face weightings (store straight back in differences)
    np.multiply(weightings_1[:,np.newaxis,:], differences, out=differences)
    np.square(differences, out=differences)
    diffs = np.sum(differences, axis=2)
    min_2_to_1 = np.argmin(diffs,axis=1)
    # now repeat but for the second weighting
    differences = np.subtract(phi_1[:,np.newaxis,:], phi_2, out=differences)
    np.multiply(weightings_2[np.newaxis,:,:], differences, out=differences)
    diffs = np.sum(differences, axis=2)
    min_1_to_2 = np.argmin(diffs,axis=0)
    return min_2_to_1, min_1_to_2

def gaussian_weighting(x, std_dev):
    return std_dev*np.sqrt(2*np.pi) * norm.pdf(x, scale=std_dev)

def geodesic_signiture_per_lm_group(face, method='exact'):
    phi_vectors = np.zeros([face.n_vertices, face.n_landmarks])
    for i,k in enumerate(face.landmarks):
        print i
        print k
        phi_vectors[:,i] = face.geodesics_about_lm(k, method)['phi']
    # ph_coords s.t. geodesic_signiture[0] is the geodesic vector for the first ordinate
    return phi_vectors

def geodesic_signiture_per_lm_group_bar_mouth(face, method='exact'):
    phi_vectors = np.zeros([face.n_vertices, face.n_landmark_vertices - 7])
    i = 0
    for k in face.landmarks.iterkeys():
        if k != 'mouth':
            for v in face.landmarks[k]:
                phi_vectors[:,i] = face.geodesics_about_vertices([v], method)['phi']
                i += 1
        else:
            phi_vectors[:,i] = face.geodesics_about_lm(k, method)['phi']
            i += 1
    print i
    print face.n_landmark_vertices - 7
    return phi_vectors


def geodesic_signiture_for_all_landmarks_with_mask(face, mask, method='exact'):
    phi_vectors = np.zeros([face.n_vertices, len(face_custom_lm)])
    for i,v in enumerate(face_custom_lm):
            phi_vectors[:,i] = face.geodesics_about_vertices([v], method)['phi']
    # ph_coords s.t. geodesic_signiture[0] is the geodesic vector for the first ordinate
    return phi_vectors

def new_face_from_mapping(face_a, face_b):
    newface = face_a.new_trimesh()
    return newface

def unique_closest_n(phi, n):
    ranking = np.argsort(phi, axis=1)
    top_n = ranking[:,:n]
    sort_order = np.lexsort(top_n.T)
    top_n_ordered = top_n[sort_order]
    top_n_diff = np.diff(top_n_ordered, axis=0)
    ui = np.ones(len(top_n), 'bool')
    ui[1:] = (top_n_diff != 0).any(axis=1)
    classes = top_n_ordered[ui]
    ui[0] = False
    classes_index_sorted = np.cumsum(ui)
    class_index = np.zeros_like(sort_order)
    class_index[sort_order] = classes_index_sorted
    return classes, class_index

