import numpy as np
from cppmesh import CppMesh
import scipy.sparse.linalg as sparse_linalg
from tvtk.api import tvtk
from tvtk.tools import ivtk
from tvtk.pyface import picker
from mayavi import mlab
import pickle

class Face(CppMesh):

  def __init__(self, coords, tri_index, file_path_no_ext=None, texture=None, texture_coords=None,
               texture_tri_index=None, landmarks = {}):
    CppMesh.__init__(self, coords, tri_index)
    self.landmarks = landmarks.copy()
    self.file_path_no_ext = file_path_no_ext
    t_in, tc_in, tti_in = texture, texture_coords, texture_tri_index
    message = 'Building new face'
    self.texture_coords_per_vertex = None
    if t_in is None:
      message += ' with no texture'
      if tc_in is not None or tti_in is not None:
        message += ', yet you supplied texture_coords or a texture_tri_index'
        message += ' (these will be ignored)'
        tc_in = None
        tti_in = None
    else:
      message += ' with a texture'
      if tc_in is None:
        message += ' but no texture_coords! (the texture will be ignored)'
        t_in = None
        tti_in = None
      else:
        message += ', texture_coords'
        if tti_in is not None:
          message += ', and texture_tri_index ->'
          message += ' assuming per-triangle texturing'
          message += ' (texture_coords per vertex will be generated)'
          self.texture_coords_per_vertex = _per_vertex_texture_coords(
              tri_index, tti_in, tc_in)
        else:
          message += ' but without a texture_tri_index ->'
          message += ' assuming per vertex texturing'
          if tc_in.shape[0] != coords.shape[0]:
            message += "..but I can't! there isn't a 1to1 mapping"
            message += " of texture_coords to coords. All texturing ignored"
            t_in = None
            tc_in = None
          else:
            self.texture_coords_per_vertex = tc_in.copy()
            tc_in = None
    print message
    self.texture = t_in
    self.texture_coords = tc_in
    self.texture_tri_index = tti_in
    self._cached_geodesics = {}
    self.last_key = None

  def _requires_texture(func):
    def check_texture(self, **kwargs):
      if self.texture is not None:
        return func(self, **kwargs)
      else:
        print 'This face has no texture associated with it'
    return check_texture


  #@_requires_texture
  def view_textured(self, mode='select'):
    pd = tvtk.PolyData()
    pd.points = self.coords
    pd.polys = self.tri_index
    pd.point_data.t_coords = self.texture_coords_per_vertex
    mapper = tvtk.PolyDataMapper(input=pd)
    actor = tvtk.Actor(mapper=mapper)
    #get out texture as a np arrage and arrange it for inclusion with a tvtk ImageData class
    np_texture = np.array(self.texture)
    image_data = np.flipud(np_texture).flatten().reshape([-1,3]).astype(np.uint8)
    image = tvtk.ImageData()
    image.point_data.scalars = image_data
    image.dimensions = np_texture.shape[1], np_texture.shape[0], 1
    texture = tvtk.Texture(input=image)
    actor.texture = texture
    v = ivtk.IVTK(size=(700,700))
    v.open()
    v.scene.add_actors(actor)
    if mode == 'select':
      pick_handler = FaceSelectPickHandler(v, self)
    elif mode == 'landmark':
      pick_handler = FaceLandmarkPickHandler(v, self)
    v.scene.picker.pick_handler = pick_handler
    v.scene.picker.show_gui = False
    return v

  def view(self):
    figure = mlab.gcf()
    mlab.clf()
    s = self._render_face()
    #s.actor.mapper.input.point_data.t_coords = self.texture_coords_per_vertex
    #np_texture = np.array(self.texture)
    #image_data = np.flipud(np_texture).flatten().reshape([-1,3]).astype(np.uint8)
    #image = tvtk.ImageData()
    #image.point_data.scalars = image_data
    #image.dimensions = np_texture.shape[1], np_texture.shape[0], 1
    #texture = tvtk.Texture(input=image)
    #s.actor.texture = texture
    ##s.parent.parent.outputs[0].point_data.t_coords = self.texture_coords
    #s.mlab_source.dataset.point_data.t_coords = self.texture_coords
    ##image = tvtk.JPEGReader()
    ##image.file_name = self.texture.filename
    #self.image = np.array(self.texture)
    #texture = tvtk.Texture(input=image, interpolate=1)
    #s.actor.texture = texture
    #s.actor.enable_texture = True
    ##engine = mlab.get_engine()
    #mlab.show()

  def _render_face(self):
    return mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.tri_index,
                             color=(0.5,0.5,0.5))

  @property
  def n_landmarks(self):
    return len(self.landmarks)

  @property
  def all_landmarks(self):
    all_landmarks = []
    for l in self.landmarks.itervalues():
      all_landmarks += l
    return all_landmarks

  @property
  def n_landmark_vertices(self):
    return len(self.all_landmarks)

  def view_with_landmarks(self):
    self.view()
    num_landmarks = self.n_landmarks
    for num, key in enumerate(self.landmarks):
      i = self.landmarks[key]
      colors = np.ones_like(i)*((num*1.0+0.1)/num_landmarks)
      print 'key: ' + `key` + ' color: ' + `colors`
      s = mlab.points3d(self.coords[i,0], self.coords[i,1],
                        self.coords[i,2],
                        colors, scale_factor=5.0,
                        vmax=1.0, vmin=0.0)
    mlab.show()

  def view_location_of_vertices(self, i):
    self.view()
    s = mlab.points3d(self.coords[i,0], self.coords[i,1],
                      self.coords[i,2],
                      color=(1,1,1), scale_factor=5.0)
    mlab.show()

  def view_location_of_triangles(self, i):
    self.view_location_of_vertices(np.unique(self.tri_index[i]))

  def view_phi(self, phi, periodicity=20):
    print 'viewing geodesics with periodicity ' + `periodicity`
    rings = np.mod(phi, periodicity)
    self.view_scalar_per_vertex(rings)


  def view_scalar_per_vertex(self, scalar):
    s = mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.tri_index,
                             scalars=scalar)
    mlab.show()

  def view_geodesic_contours_about_vertices(self, vertices,
      periodicity=20, method='heat'):
    phi = self.geodesics_about_vertices(vertices, method)['phi']
    self.view_phi(phi, periodicity)

  def view_geodesic_contours_about_lm(self, landmark_key,
      periodicity=20, method='heat'):
    self.view_geodesic_contours_about_vertices(
        self.landmarks[landmark_key], periodicity, method)

  def store_geodesics_for_all_landmarks(self, method='heat'):
    for key in self.landmarks:
      self.geodesics_about_vertices(self.landmarks[key], method)

  def geodesics_about_vertices(self, source_vertices, method='heat'):
    key = tuple([method] + sorted(set(source_vertices)))
    geodesic = self._cached_geodesics.get(key)
    if geodesic is not None:
      print 'already calculated this geodesic, returning it'
      return geodesic
    else:
      geodesic = self.geodesics(source_vertices, method)
      self._cached_geodesics[key] = geodesic
      return geodesic

  def geodesics_about_lm(self, landmark_key, method='heat'):
    return self.geodesics_about_vertices(self.landmarks[landmark_key], method)

  def lsqr_heat_phi_about_lm(self, landmark_key):
    geo = self.geodesics_about_lm(landmark_key, method='heat')
    div_X = geo['div_X']
    L_c = self._cache['laplacian_cotangent']
    lsqr_solution = sparse_linalg.lsqr(L_c, div_X, show=True)
    phi_lsqr = lsqr_solution[0]
    phi_lsqr = phi_lsqr - phi_lsqr[self.landmarks[landmark_key]]
    return phi_lsqr

  def new_face_from_vertex_mask(self, vertex_mask):
    original_vertex_index = np.arange(self.n_vertices)
    kept_vertices = original_vertex_index[vertex_mask]
    bool_coord_index_mask = \
      np.in1d(self.tri_index, kept_vertices).reshape(self.tri_index.shape)
    # remove any triangle missing any number of vertices
    kept_triangles_orig_index = self.tri_index[np.all(bool_coord_index_mask, axis = 1)]
    # some additional vertices will have to be removed as they no longer
    # form part of a triangle
    kept_vertices_orig_index = np.unique(kept_triangles_orig_index)
    ci_map = np.zeros_like(original_vertex_index)
    new_vertex_numbering = np.arange(kept_vertices_orig_index.shape[0])
    ci_map[kept_vertices_orig_index] = new_vertex_numbering
    new_coord_index = ci_map[kept_triangles_orig_index].astype(np.uint32)
    new_coords = self.coords[kept_vertices_orig_index]
    new_landmarks = self.landmarks.copy()
    for feature in new_landmarks:
      new_landmarks[feature] = list(ci_map[new_landmarks[feature]])
    # now map across texture coordinates
    new_tc, new_tci = None, None
    if self.texture_tri_index is not None:
      # have per-face texturing. Provide new_tc/new_tci -> new Face will
      # generate tc_per_verex automatically
      kept_tci_orig_index = self.texture_tri_index[
          np.all(bool_coord_index_mask, axis = 1)]
      kept_tc_orig_index = np.unique(kept_tci_orig_index)
      tci_map = np.zeros(self.texture_coords.shape[0])
      new_tc_numbering = np.arange(kept_tc_orig_index.shape[0])
      tci_map[kept_tc_orig_index] = new_tc_numbering
      new_tci = tci_map[kept_tci_orig_index].astype(np.uint32)
      new_tc  = self.texture_coords[kept_tc_orig_index]
    elif texture is not None:
      # have per-verex texturing only. just generate new_tc and submit
      new_tc = self.texture_coords_per_vertex[kept_vertices_orig_index]

    face = Face(new_coords, new_coord_index, texture=self.texture,
                texture_coords =new_tc, texture_tri_index=new_tci)
    face.landmarks = new_landmarks
    return face

  def new_face_masked_from_lm(self, lm_key, distance=100, method='heat'):
    """Returns a face containing only vertices within distance of the nose lm
    """
    return self.new_face_masked_from_lms([lm_key], [distance], method)

  def new_face_masked_from_lms(self, lm_keys, distances, method='heat'):
    # create a mask that is all false
    phi = np.zeros(self.n_vertices)
    mask = phi != 0
    for lm_key, distance in zip(lm_keys, distances):
      phi = self.geodesics_about_lm(lm_key, method)['phi']
      new_mask = np.logical_and(phi < distance, phi >= 0)
      # OR with the mask (e.g. for one lm_key, this is a no op.
      # for multiple lm keys this OR's all the masks together
      mask = np.logical_or(mask, new_mask)
    return self.new_face_from_vertex_mask(mask)

  def save_landmarks(self, captured=True, path=None):
    if self.file_path_no_ext == None and path == None:
      raise Exception("face has no knowledge of it's file path, and you didn't\
          provide one")
    elif path == None:
      path = self.file_path_no_ext + '.landmarks'
    f = open(path, 'w')
    if captured:
      print 'saving out the captured landmarks to ' + `path`
      pickle.dump(self.captured_landmarks, f)
    else:
      print 'saving out the landmarks to ' + `path`
      pickle.dump(self.landmarks, f)

  def load_landmarks(self, path=None):
    if self.file_path_no_ext == None and path == None:
      raise Exception("face has no knowledge of it's file path, and you didn't\
          provide one")
    elif path == None:
      path = self.file_path_no_ext + '.landmarks'
    f = open(path, 'r')
    self.landmarks = pickle.load(f)
    print 'loaded landmarks from ' + `path`

def _per_vertex_texture_coords(tri_index, texture_tri_index, texture_coords):
  # need to change the per-face tc to per-vertex. obviously
  # this means we loose data (and some faces will have fugly
  # textures) but on the whole it will work.
  u_ci, ind_of_u_ci = np.unique(tri_index, return_index=True)
  # grab these positions from the texture_coord_index to find an instance
  # of a tc at each vertex
  per_vertex_tci = texture_tri_index.flatten()[ind_of_u_ci]
  return texture_coords[per_vertex_tci]


#class MyInteractorStyle(tvtk.InteractorStyleTrackballCamera):
#  def __init__(self, parent=None):
#    self.add_observer("MiddleButtonPressEvent", self.middleB


class FaceSelectPickHandler(picker.PickHandler):

  def __init__(self, scene, face):
    picker.PickHandler.__init__(self)
    self.count = 0

  def handle_pick(self, data):
    print `self.count` + ': ' + `data.point_id`
    self.count += 1


class FaceLandmarkPickHandler(picker.PickHandler):

  def __init__(self, scene, face):
    picker.PickHandler.__init__(self)
    self.scene = scene
    self.face = face
    self.i = 0
    self.j = 0
    face.captured_landmarks = []
    self.landmark_titles = ['mouth', 'nose', 'l_brow', 'bridge', 'r_brow', 'l_eye', 'r_eye']
    self.landmark_counts = [8, 1, 3, 1, 3, 8, 8]
    self.landmarks = []
    self.landmarks.append(self._create_lm_dict('mouth' , 8))
    self.landmarks.append(self._create_lm_dict('nose'  , 1))
    self.landmarks.append(self._create_lm_dict('l_brow', 3))
    self.landmarks.append(self._create_lm_dict('bridge', 1))
    self.landmarks.append(self._create_lm_dict('r_brow', 3))
    self.landmarks.append(self._create_lm_dict('l_eye' , 8))
    self.landmarks.append(self._create_lm_dict('r_eye' , 8))

  def _create_lm_dict(self, title, count):
    landmark = {} 
    landmark['title'] = title
    landmark['count'] = count
    landmark['landmarks'] = []
    return landmark

  def handle_pick(self, data):
    print `self.landmarks[self.i]['title']` + '_' + `self.j` + ': ' + `data.point_id`
    self.landmarks[self.i]['landmarks'].append(data.point_id)
    self.j += 1
    if self.j == self.landmarks[self.i]['count']:
      self.i += 1
      self.j = 0
      if self.i == len(self.landmarks):
        print 'saving landmarks'
        face_landmarks = {}
        for lm in self.landmarks:
          face_landmarks[lm['title']] = lm['landmarks']
        self.face.captured_landmarks = face_landmarks
        self.scene.close()

