import numpy as np
from collections import OrderedDict
from pybug.visualization import PointCloudViewer3d, LabelViewer3d


class FieldError(Exception):
    pass

class PointFieldError(FieldError):
    pass

class SpatialDataConstructionError(Exception):
    pass


class SpatialData(object):
    """ Abstract representation of a n-dimentional piece of spatial data.
    This could be simply be a set of vectors in an n-dimentional space,
    or a structed surface or mesh. At this level of abstraction we only
    define basic metadata that can be attached to all kinds of spatial
    data
    """
    def __init__(self):
        pass


class PointCloud(SpatialData):
    """n-dimensional point cloud. Can be coerced to a PCL Point Cloud Object
    for using their library methods (TODO). Handles the addition of spatial
    metadata (most commonly landmarks) by storing all such 'metapoints'
    (points which aren't part of the shape) and normal points together into
    a joint field (points_and_metapoints). This is masked from the end user 
    by the use of properties.
    """
    def __init__(self, points, n_metapoints=0):
        SpatialData.__init__(self)
        self.n_points, n_dims  = points.shape
        self.n_metapoints = n_metapoints
        self.points_and_metapoints = np.empty(
                [self.n_points + self.n_metapoints, n_dims])
        self.points_and_metapoints[:self.n_points] = points
        self.pointfields = {}

    @property
    def points(self):
        return self.points_and_metapoints[:self.n_points]

    @property
    def metapoints(self):
        """Points which are solely for metadata. Are guaranteed to be
        transformed in exactly the same way that points are. Useful for
        storing explicit landmarks (landmarks that have coordinates and
        don't simply reference exisiting points).
        """
        return self.points_and_metapoints[self.n_points:]

    @property
    def n_dims(self):
        return self.points.shape[1]

    def __str__(self):
        message = str(type(self)) + ': n_points: ' + `self.n_points`  \
                + ', n_dims: ' + `self.n_dims`
        if len(self.pointfields) != 0:
            message += '\n  pointfields:'
            for k,v in self.pointfields.iteritems():
                try:
                    field_dim = v.shape[1]
                except IndexError:
                    field_dim = 1
                message += '\n    ' + str(k) + '(' + str(field_dim) + 'D)'
        return message


    def add_pointfield(self, name, field):
        """Add another set of field values (of arbitrary dimention) to each
        point.
        """
        if field.shape[0] != self.n_points:
            raise PointFieldError("Trying to add a field with " +
                    `field.shape[0]` + " values (need one field value per " +
                    "point => " + `self.n_points` + " values required")
        else:
            self.pointfields[name] = field

    def view(self):
            print 'arbitrary dimensional PointCloud rendering is not supported.'


class PointCloud3d(PointCloud):
    """A PointCloud constrained to 3 dimensions. Has additional visualization
    methods. Should always be used in lieu of PointCloud when using 3D data.
    """
    def __init__(self, points, n_metapoints=0):
        PointCloud.__init__(self, points, n_metapoints)
        if self.n_dims != 3:
            raise SpatialDataConstructionError(
                    'Trying to build a 3D Point Cloud with from ' +
                    str(self.n_dims) + ' data')

    def view(self, **kwargs):
        viewer = PointCloudViewer3d(self.points, **kwargs)
        return viewer.view()

    def attach_landmarks(self, landmarks_dict):
        self.landmarks = Landmarks(self, landmarks_dict)


class Landmarks(object):
    """Class for storing and manipulating Landmarks associated with a shape.
    Landmarks index into the points and metapoints of the associated 
    PointCloud. Landmarks which are expicitly given as coordinates would
    be entirely constructed from metapoints, whereas point indexed landmarks
    would be composed entirely of points. This class can handle any arbitrary
    mixture of the two.
    """
    def __init__(self, pointcloud, landmarks_dict):
        """ pointcloud - the shape whose these landmarks apply to
        landmark_dict - keys - landmark classes (e.g. 'mouth')
                        values - ordered list of landmark indices into 
                        pointcloud.points_and_metapoints
        """
        self.pc = pointcloud
        # indexes are the indexes into the points and metapoints of self.pc.
        # note that the labels are always sorted when stored.
        self.indexes = OrderedDict(sorted(landmarks_dict.iteritems()))

    def all(self, labels=False, indexes=False, numbered=False):
        """return all the landmark indexes. The order is always guaranteed to
        be the same for a given landmark configuration - specifically, the 
        points will be returned by sorted label, and always in the order that 
        each point the landmark was construted in.
        """
        all_lm = []
        lmlabels = []
        for k, v in self.indexes.iteritems():
            if indexes:
                all_lm += v
            else:
                all_lm += list(self.pc.points_and_metapoints[v])
            newlabels = [k] * len(v)
            if numbered:
                newlabels = [x + '_' + str(i) for i, x in enumerate(newlabels)]
            lmlabels += newlabels
        if labels:
           return np.array(all_lm), lmlabels
        return np.array(all_lm)

    def __getitem__(self, label):
        return self.pc.points_and_metapoints[self.indexes[label]]

    def view(self, **kwargs):
        """ View all landmarks on the current shape, using the default
        shape view method. Kwargs passed in here will be passed through
        to the shapes view method.
        """
        lms, labels = self.all(labels=True, numbered=True)
        pcviewer = self.pc.view(**kwargs)
        pointviewer = PointCloudViewer3d(lms)
        pointviewer.view(onviewer=pcviewer)
        lmviewer = LabelViewer3d(lms, labels, offset=np.array([0,16,0]))
        lmviewer.view(onviewer=pcviewer)
        return lmviewer

    @property
    def n_points(self):
        return self.all().shape[0]

    @property
    def n_groups(self):
        return len(self.indexes)

    @property
    def config(self):
        """A nested tuple specifying the precise nature of the landmarks
        (labels, and n_points per label). Allows for comparison of Landmarks
        to see if they are likely describing the same shape.
        """
        return tuple((k,len(v)) for k,v in self.indexes.iteritems())


class SpatialDataCollectionError(Exception):
    pass

class ShapeClassError(SpatialDataCollectionError):
    pass


class SpatialDataCollection(object):
    """ A bag of SpatialData. Provides funtionality for 
    - viewing all the data in the set
    - performing transformations on all pieces of data in the set

    will enforce that all the added elements are instances of SpecialData
    but that's it.
    """
    def __init__(self, spatialdataiter):
        if not all(isinstance(x, SpatialData) for x in spatialdataiter):
            notsd = [x for x in spatialdataiter 
                    if not isinstance(x, SpatialData)]
            raise SpatialDataCollectionError('Can only add SpatialData '\
                    + ' instances (' + `notsd` + ' are not)')
        self.data = set(spatialdataiter)

    def add_spatialdata(self, spatialdata):
        """ Adds an instance of spatialdata to the collection
        """
        if not isinstance(spatialdata, SpatialData):
            raise SpatialDataCollectionError('Can only add SpatialData '\
                    + ' instances')
        else:
            self.data.add(spatialdata)


class ShapeClass(SpatialDataCollection):
    """A collection of SpatialData that all have the same
    landmark configuration (and so can be considered to be of the same shape)
    """
    def __init__(self, spatialdataiter):
        SpatialDataCollection.__init__(self, spatialdataiter)
        try:
            unique_lm_configs = set(x.landmarks.config for x in self.data)
        except AttributeError:
            raise ShapeClassError("All elements of a shape class must have "\
                    + "landmarks attached")
        if len(unique_lm_configs) != 1:
            raise ShapeClassError("All elements in shape class must have "\
                    + "landmarks with the same config")

