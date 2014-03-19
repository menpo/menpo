from decorator import decorator


@decorator
def const(f, *args, **kwargs):
    """
    Decorator to maintain constness of parameters.

    This should be used on any function that takes ndarrays as parameters.
    This helps protect against potential subtle bugs caused by the pass by
    reference nature of python arguments.

    ``copy`` is assumed to a shallow copy.

    Parameters
    ----------
    f : func
        Function to wrap
    args : list
        Function ``f`` s args
    kwargs : dictionary
        Function ``f`` s kwargs
    copy : {True, False}, optional
        If ``False`` then don't copy the parameters - use with caution

        Default : True
    """
    copy = kwargs.pop('copy', True)
    if copy:
        args = [x.copy() for x in args]
        kwargs = dict((k, v.copy()) for k, v in kwargs.items())
    return f(*args, **kwargs)
