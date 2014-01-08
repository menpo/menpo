

class FieldError(Exception):
    """
    Base class for errors to do with setting fields on PointClouds and their
    subclasses.
    """
    pass


class PointFieldError(FieldError):
    """
    Raised when setting point fields on PointClouds.
    """
    pass
