import numpy as np
from tvtk.api import tvtk
from tvtk.tools import ivtk
from tvtk.pyface import picker
from mayavi import mlab

class PointCloud(object):

  def __init__(self, coords):
    self.coords = coords

  def view(self, mode=None, picker='select'):
    """View the model. By default, model's with textures are viewed textured,
    if not just a monochrome model rendering is shown. This function serves 
    dual purpose as a basic landmarker
    """
    # by default view with texture if we have one, else just model
    if mode == None:
      if self.texture is not None:
        mode = 'texture'
      else:
        mode = 'model'

    if mode == 'model':
      scene = self._render_model()
    elif mode == 'texture':
      scene = self._render_textured()
    else:
      raise Exception('dont understand display mode')
    if picker == 'select':
      pick_handler = FaceSelectPickHandler(self)
    elif picker == 'landmark':
      pick_handler = FaceLandmarkPickHandler(self)
    elif type(picker) != str:
      pick_handler = picker
    scene.picker.pick_handler = pick_handler
    scene.picker.show_gui = False
    return scene

  def _render_model(self):
    figure = mlab.gcf()
    mlab.clf()
    s = mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.tri_index,
                             color=(0.5,0.5,0.5))
    return s.scene


  def view_location_of_vertex_indices(self, i):
    self.view_location_of_vertices(self.coords[i])

  def view_location_of_vertices(self, coords):
    self.view()
    s = mlab.points3d(coords[:,0], coords[:,1],
                      coords[:,2],
                      color=(1,1,1), mode='axes')
    mlab.show()

  def view_scalar_per_vertex(self, scalar, **kwargs):
    colormap = kwargs.get('colormap', 'jet')
    s = mlab.triangular_mesh(self.coords[:,0], self.coords[:,1],
                             self.coords[:,2], self.tri_index,
                             scalars=scalar, colormap=colormap)
    mlab.show()

 new_f_from_vertex_mask(self, vertex_mask):
   (self, vertex_mask):
    original_vertex_index = np.arange(self.n_vertices)
    kept_vertices = original_vertex_index[vertex_mask]
    bool_coord_index_mask = \
      np.in1d(self.tri_index, kept_vertices).reshape(self.tri_index.shape)
    # remove any triangle missing any number of vertices
    kept_triangles_orig_index = self.tri_index[np.all(bool_coord_index_mask, 
      axis = 1)]
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

