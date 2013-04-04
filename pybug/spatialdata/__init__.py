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
    def __init__(self, points):
        SpatialData.__init__(self)
        self.n_points, n_dims  = points.shape
        self.n_metapoints = 0
        cachesize = 1000
        # prealocate allpoints to have enough room for cachesize metapoints
        self._allpoints = np.empty([self.n_points + cachesize, n_dims])
        self._allpoints[:self.n_points] = points
        self.pointfields = {}

    @property
    def points(self):
        return self._allpoints[:self.n_points]

    @property
    def metapoints(self):
        """Points which are solely for metadata. Are guaranteed to be
        transformed in exactly the same way that points are. Useful for
        storing explicit landmarks (landmarks that have coordinates and
        don't simply reference exisiting points).
        """
        return self._allpoints[self.n_points:]

    @property
    def points_and_metapoints(self):
        return self._allpoints[:self.n_points_and_metapoints]

    @property
    def n_points_and_metapoints(self):
        return self.n_points + self.n_metapoints

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

    def add_metapoint(self, metapoint):
        """Adds a new metapoint to the pointcloud. Returns the index
        position that this point is stored at in self.points_and_metapoints.
        """
        if metapoint.size != self.n_dims:
            raise Exception("metapoint must be of the same number of dims"\
                    + " as the parent pointcloud")
        next_index = self.n_points_and_metapoints
        self._allpoints[next_index] = metapoint.flatten()
        self.n_metapoints += 1
        return next_index


    def view(self):
            print 'arbitrary dimensional PointCloud rendering is not supported.'


class PointCloud3d(PointCloud):
    """A PointCloud constrained to 3 dimensions. Has additional visualization
    methods. Should always be used in lieu of PointCloud when using 3D data.
    """
    def __init__(self, points):
        PointCloud.__init__(self, points)
        if self.n_dims != 3:
            raise SpatialDataConstructionError(
                    'Trying to build a 3D Point Cloud with from ' +
                    str(self.n_dims) + ' data')
        self.landmarks = LandmarkManager(self)

    def view(self, **kwargs):
        viewer = PointCloudViewer3d(self.points, **kwargs)
        return viewer.view()


class Landmark(object):
    """ An object representing an annotated point in a pointcloud.
    Only makes sense in the context of a parent pointcloud, and so
    one is required at construction.
    """
    def __init__(self, pointcloud, pointcloud_index, label, label_index):
        self.pointcloud = pointcloud
        self.pointcloud_index = pointcloud_index
        self.label = label
        self.label_index = label_index

    def aspoint(self):
        return self.pointcloud.points_and_metapoints[self.pointcloud_index]

    def asindex(self):
        return self.pointcloud_index

    @property
    def numbered_label(self):
        return self.label + '_' + str(self.label_index)


class ReferenceLandmark(Landmark):
    """A Landmark that references a point that is a part of a point cloud
    """
    def __init__(self, pointcloud, pointcloud_index, label, label_index):
        Landmark.__init__(self, pointcloud, pointcloud_index,
                label, label_index)
        if pointcloud_index < 0 or pointcloud_index > self.pointcloud.n_points:
            raise Exception("Reference landmarks have to have an index "\
                    + "in the range 0 < i < n_points of the parent pointcloud")


class MetaLandmark(Landmark):
    """A landmark that is totally seperate to the parent point cloud."
    """
    def __init__(self, pointcloud, metapoint, label, label_index):
        pointcloud_index = pointcloud.addmetapoint(metapoint)
        Landmark.__init__(self, pointcloud, pointcloud_index,
                label, label_index)

    @property
    def metapoint_index(self):
        """ How far into the metapoints part of the array this metapoint is
        """
        return self.index - self.pc.n_points - 1

class LandmarkManager(object):
    """Class for storing and manipulating Landmarks associated with a shape.
    Landmarks index into the points and metapoints of the associated
    PointCloud. Landmarks which are expicitly given as coordinates would
    be entirely constructed from metapoints, whereas point indexed landmarks
    would be composed entirely of points. This class can handle any arbitrary
    mixture of the two.
    """
    def __init__(self, pointcloud):
        """ pointcloud - the shape whose these landmarks apply to
        landmark_dict - keys - landmark classes (e.g. 'mouth')
                        values - ordered list of landmark indices into
                        pointcloud.points_and_metapoints
        """
        self.pc = pointcloud
        self.all_landmarks = []
        self.labels = {}
        # indexes are the indexes into the points and metapoints of self.pc.
        # note that the labels are always sorted when stored.

    def add_reference_landmarks(self, landmark_dict):
        #self.indexes = OrderedDict(sorted(landmarks_dict.iteritems()))
        for k, v in landmark_dict.iteritems():
            k_lms = []
            for i, index in enumerate(v):
                lm = ReferenceLandmark(self.pc, index, k, i)
                self.all_landmarks.append(lm)
                k_lms.append(lm)
            self.labels[k] = k_lms


    def all(self, labels=False, indexes=False, numbered=False):
        """return all the landmark indexes. The order is always guaranteed to
        be the same for a given landmark configuration - specifically, the
        points will be returned by sorted label, and always in the order that
        each point the landmark was construted in. THIS IS OUT OF DATE.
        """
        v = self.reference_landmarks
        if indexes:
            all_lm = [x.asindex() for x in v]
        else:
            all_lm = [list(x.aspoint()) for x in v]
        newlabels = [x.label for x in v]
        if numbered:
            newlabels = [x.numbered_label for x in v]
        lmlabels = newlabels
        if labels:
           return np.array(all_lm), lmlabels
        return np.array(all_lm)

    @property
    def reference_landmarks(self):
        return [x for x in self.all_landmarks if isinstance(x, ReferenceLandmark)]

    @property
    def meta_landmarks(self):
        return [x for x in self.all_landmarks if isinstance(x, MetaLandmark)]

    def with_label(self, label):
        return [x for x in self.all_landmarks if x.label == label]

    #def __getitem__(self, label):
    #    return self.pc.points_and_metapoints[self.indexes[label]]

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
        """A frozen set specifying all the landmarks numbered labels
        """
        #"""A nested tuple specifying the precise nature of the landmarks
        #(labels, and n_points per label). Allows for comparison of Landmarks
        #to see if they are likely describing the same shape.
        #"""
        #return tuple((k,len(v)) for k,v in self.labels.iteritems())
        return frozenset(x.numbered_label for x in self.all_landmarks)


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
        self.data = list(spatialdataiter)

    def add_spatialdata(self, spatialdata):
        """ Adds an instance of spatialdata to the collection
        """
        if not isinstance(spatialdata, SpatialData):
            raise SpatialDataCollectionError('Can only add SpatialData '\
                    + ' instances')
        else:
            self.data.append(spatialdata)


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

