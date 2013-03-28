


class Landmarks(object):
  """Class for storing and manipulating Landmarks associated
  with a shape. Landmarks are named sets of annotations.
  """
  def __init__(self):
    pass

class Shape(object):
  """ Abstract representation of a n-dimentional shape. This
  Could be simply a set of vectors in an n-dimentional shape.
  Optionally, all shapes can have associated with them
  a Landmarks object containing annotations about the object
  in the space.
  """
  def __init__(self):
    self.landmarks = Landmarks()
