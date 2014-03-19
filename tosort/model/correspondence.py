from menpo.shape.collection import TriMeshShapeClass


class CorrespondingTriMeshPair(TriMeshShapeClass):

    def __init__(self, trimeshiter):
        TriMeshShapeClass.__init__(self, trimeshiter)
        if len(self.data) != 2:
            raise Exception("A Corresponding trimesh pair requires two"\
                    + " TriMesh instances")
        self.f0 = self.data[0]
        self.f1 = self.data[1]
        if (self.f0.pointfields.get(self.f1) == None or
                self.f1.pointfields.get(self.f0) == None):
            raise Exception("Need to have a correspondence pointfield set on "\
                    + "each trimesh for the opposite trimesh")
