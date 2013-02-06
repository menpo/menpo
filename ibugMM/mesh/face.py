import numpy as np
from mayavi import mlab
from cppmesh import CppMesh
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse import linalg 


class CorrespondingFaces(object):

  def __init__(self, template_face):
    self.template_face = template_face

class Face(CppMesh):

  def __init__(self, **kwargs):
    CppMesh.__init__(self)
    self.textureCoords      = kwargs.get('textureCoords')
    self.textureCoordsIndex = kwargs.get('textureCoordsIndex')
    self.texture            = kwargs.get('texture')
    self.calculated_geodesics = {}
    self.landmarks = {}
    self.last_key = None
    # some file types include normals - if they are not none, import them
    #if kwargs['normals']:
    #self.normals            = np.array(kwargs.get(['normals']))
    #self.normalsIndex       = np.array(kwargs.get)['normalsIndex']))
    #else:
    #  self.normals, self.normalsIndex = None, None

  def view(self):
    figure = mlab.gcf()
    mlab.clf()
    s = self._render_face()
    mlab.show()
    return s

  def _render_face(self):
    return mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.coordsIndex, 
                             color=(0.5,0.5,0.5)) 

  @property
  def number_of_landmarks(self):
    return len(self.landmarks)

  def view_with_landmarks(self):
    self.view()
    num_landmarks = self.number_of_landmarks
    for num, key in enumerate(self.landmarks):
      i = self.landmarks[key]
      colors = np.ones_like(i)*((num*1.0+0.1)/num_landmarks)
      print 'key: ' + `key` + ' color: ' + `colors`
      s = mlab.points3d(self.coords[i,0], self.coords[i,1],
                        self.coords[i,2], 
                        colors, scale_factor=5.0,
                        vmax=1.0, vmin=0.0) 
    mlab.show()

  def view_location_of_vertex(self, i):
    self.view()
    s = mlab.points3d(self.coords[i,0], self.coords[i,1],
                      self.coords[i,2], 
                      color=(1,1,1), scale_factor=5.0) 
    mlab.show()

  def view_geodesic_contours(self, phi):
    rings = np.mod(phi,20)
    s = mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.coordsIndex, 
                             scalars=rings) 
    mlab.show()

  def view_last_geodesic_contours(self):
    if self.last_key:
      self.view_geodesic_contours(self.calculated_geodesics[self.last_key]['phi'])
    else:
      print "No geodesics have been calculated for this face"

  def calculate_geodesics_for_all_landmarks(self):
    for key in self.landmarks:
      self.calculate_geodesics(self.landmarks[key])


  def calculate_geodesics(self, source_vertices):
    key = tuple(sorted(set(source_vertices)))
    geodesic = self.calculated_geodesics.get(key)
    if geodesic is not None:
      print 'already calculated this geodesic, returning it'
      return geodesic
    else:
      geodesic = self.heat_geodesics(source_vertices)
      self.calculated_geodesics[key] = geodesic
      self.last_key = key
      return geodesic

  def new_face_from_vertex_mask(self, vertex_mask):
    original_vertex_index = np.arange(self.n_coords)
    kept_vertices = original_vertex_index[vertex_mask]
    bool_coord_index_mask = \
      np.in1d(self.coordsIndex, kept_vertices).reshape(self.coordsIndex.shape)
    kept_triangles_orig_index = self.coordsIndex[np.all(bool_coord_index_mask, axis = 1)]
    # some additional vertices will have to be removed as they no longer 
    # form part of a triangle
    kept_vertices_orig_index = np.unique(kept_triangles_orig_index)
    number_conversion = np.zeros_like(original_vertex_index)
    new_vertex_numbering = np.arange(kept_vertices_orig_index.shape[0])
    number_conversion[kept_vertices_orig_index] = new_vertex_numbering
    new_coord_index = number_conversion[kept_triangles_orig_index]
    new_coords = self.coords[kept_vertices_orig_index]
    new_landmarks = self.landmarks.copy()
    for feature in new_landmarks:
      new_landmarks[feature] = number_conversion[new_landmarks[feature]]
    face = Face(coords=new_coords, coordsIndex=new_coord_index.astype(np.uint32))
    face.landmarks = new_landmarks
    return face

  def new_face_masked_from_nose_landmark(self, **kwargs):
    """Returns a face containing only vertices within distance of the nose lm
    """
    distance = kwargs.get('distance', 100)
    phi = self.calculate_geodesics(self.landmarks['nose'])['phi']
    mask = np.logical_and(phi < 100, phi >= 0)
    return self.new_face_from_vertex_mask(phi < 100)


