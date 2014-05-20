def extract_parametric_features(appearance_model, warped_image,
                                rergession_features):
    r"""
    Extracts a particular parametric feature given an appearance model and
    a warped image.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    rergession_features: function/closure
        Defines the function from which the parametric features will be
        extracted.

        Non-default regression feature options and new experimental features
        can be used by defining a closure. In this case, the closure must
        define a function that receives as an input an appearance model and
        a warped masked image and returns a particular parametric feature
        representation. For example:

    Returns
    -------
    features: np.array
        The resulting parametric features.
    """
    if rergession_features is None:
        features = weights(appearance_model, warped_image)
    elif hasattr(rergession_features, '__call__'):
        features = rergession_features(appearance_model, warped_image)
    else:
        raise ValueError("rergession_features can only be: (1) None "
                         "or (2) a closure defining a non-standard "
                         "feature computation (see `menpo.fit.regression."
                         "parametricfeatures`")
    return features


def weights(appearance_model, warped_image):
    r"""
    Returns the resulting weights after projecting the warped image to the
    appearance PCA model.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    return appearance_model.project(warped_image)


def whitened_weights(appearance_model, warped_image):
    r"""
    Returns the sheared weights after projecting the warped image to the
    appearance PCA model.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    return appearance_model.project_whitened(warped_image)


def appearance(appearance_model, warped_image):
    r"""
    Projects the warped image onto the appearance model and rebuilds from the
    weights found.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    return appearance_model.reconstruct(warped_image).as_vector()


def difference(appearance_model, warped_image):
    r"""
    Returns the difference between the warped image and the image constructed
    by projecting the warped image onto the appearance model and rebuilding it
    from the weights found.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    return (warped_image.as_vector() -
            appearance(appearance_model, warped_image))


def warped_image(appearance_model, warped_image):
    r"""
    Returns the difference between the warped image and the image constructed
    by projecting the warped image onto the appearance model and rebuilding it
    from the weights found.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    return warped_image.as_vector()


def project_out(appearance_model, warped_image):
    r"""
    Returns a version of the whitened warped image where all the basis of the
    model have been projected out and which has been scaled by the inverse of
    the appearance model's noise_variance.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    diff = warped_image.as_vector() - appearance_model.mean.as_vector()
    return appearance_model.distance_to_subspace_vector(diff).ravel()


def probabilistic(appearance_model, warped_image):
    r"""
    Returns the sum of the projected-out reconstructed warped image and the
    sheared (non-orthogonal) reconstruction of the image.

    Parameters
    ----------
    appearance_model: :class:`menpo.model.pca`
        The appearance model based on which the parametric features will be
        computed.
    warped_image: :class:`menpo.image.masked`
        The warped image.
    """
    diff = warped_image.as_vector() - appearance_model.mean.as_vector()
    po = appearance_model.distance_to_subspace_vector(diff).ravel()
    return po + appearance_model.project_whitened_vector(diff).ravel()


# TODO: complete me
def quadratic_weights(appearance_model, warped_image):
    r"""
    """
    return appearance_model.project(warped_image)
