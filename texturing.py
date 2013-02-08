from tvtk.api import tvtk
from tvtk.tools import ivtk

self = o_ioannis_1
pd = tvtk.PolyData()
pd.points = self.coords
pd.polys = self.coordsIndex
pd.point_data.t_coords = self.textureCoords
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
