import wrapt

# tests currently expect that all features automatically constrain landmarks
# small wrapper which does this. Note that this decorator only works when
# called with menpo Image instances.
@wrapt.decorator
def constrain_landmarks(wrapped, instance, args, kwargs):

    def _execute(image, *args, **kwargs):
        feature = wrapped(image, *args, **kwargs)
        # after calculation, constrain the landmarks to the bounds
        feature.constrain_landmarks_to_bounds()
        return feature

    return _execute(*args, **kwargs)


def check_features(features, n_levels, pyramid_on_features):
    r"""
    Checks the feature type per level.
    If pyramid_on_features is False, it must be a function or a list of
    those containing 1 or {n_levels} elements.
    If pyramid_on_features is True, it must be a function or a list of 1
    of those.

    Parameters
    ----------
    n_levels: int
        The number of pyramid levels.
    pyramid_on_features: boolean
        If True, the pyramid will be applied to the feature image, so
        the user needs to define a single features.
        If False, the pyramid will be applied to the intensities image and
        features will be extracted at each level, so the user can define
        a features per level.

    Returns
    -------
    feature_list: list
        A list of feature function.
        If pyramid_on_features is True, the list will have length 1.
        If pyramid_on_features is False, the list will have length
        {n_levels}.
    """
    # Firstly, make sure we have a list of callables of the right length
    if not pyramid_on_features:
        try:
            all_callables = check_list_callables(features, n_levels)
        except ValueError:
            raise ValueError("features must be a callable or a list of "
                             "{} callables".format(n_levels))
    else:
        if not callable(features):
            raise ValueError("pyramid_on_features is enabled so features "
                             "must be a single callable")
        all_callables = check_list_callables(features, n_levels)

    # constrain each feature to the bounds
    all_callable_constrained = []
    for ft in all_callables:
        all_callable_constrained.append(constrain_landmarks(ft))
    return all_callable_constrained


def check_list_callables(callables, n_callables):
    if not isinstance(callables, list):
        return [callables] * n_callables
    for c in callables:
        if not callable(c):
            raise ValueError("All items must be callables")
    if len(callables) != n_callables:
        raise ValueError("List of callables must be {} "
                         "long".format(n_callables))
    return callables


def check_n_levels(n_levels):
    r"""
    Checks the number of pyramid levels - must be int > 0.
    """
    if not isinstance(n_levels, int) or n_levels < 1:
        raise ValueError("n_levels must be int > 0")


def check_downscale(downscale):
    r"""
    Checks the downscale factor of the pyramid that must be >= 1.
    """
    if downscale < 1:
        raise ValueError("downscale must be >= 1")


def check_normalization_diagonal(normalization_diagonal):
    r"""
    Checks the diagonal length used to normalize the images' size that
    must be >= 20.
    """
    if normalization_diagonal is not None and normalization_diagonal < 20:
        raise ValueError("normalization_diagonal must be >= 20")


def check_boundary(boundary):
    r"""
    Checks the boundary added around the reference shape that must be
    int >= 0.
    """
    if not isinstance(boundary, int) or boundary < 0:
        raise ValueError("boundary must be >= 0")


def check_max_components(max_components, n_levels, var_name):
    r"""
    Checks the maximum number of components per level either of the shape
    or the appearance model. It must be None or int or float or a list of
    those containing 1 or {n_levels} elements.
    """
    str_error = ("{} must be None or an int > 0 or a 0 <= float <= 1 or "
                 "a list of those containing 1 or {} elements").format(
        var_name, n_levels)
    if not isinstance(max_components, list):
        max_components_list = [max_components] * n_levels
    elif len(max_components) == 1:
        max_components_list = [max_components[0]] * n_levels
    elif len(max_components) == n_levels:
        max_components_list = max_components
    else:
        raise ValueError(str_error)
    for comp in max_components_list:
        if comp is not None:
            if not isinstance(comp, int):
                if not isinstance(comp, float):
                    raise ValueError(str_error)
    return max_components_list
