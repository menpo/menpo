def compute_features(image, feature_type):
    r"""
    Computes a particular feature representation of the given images.

    Parameters
    ----------
    image: :class:`menpo.image.MaskedNDImage`
        The original image from which the features will be computed.
    feature_type: string or closure
        If None, no feature representation will be computed from the
        original image.
        If string or closure, the feature representation will be computed
        in the following way:
            If string, the feature representation will be extracted by
            executing:

                feature_image = eval('img.features.' + feature_type + '()')

            For this to work properly feature_type needs to be one of
            Menpo's standard image feature methods. Note that, in this case,
            the feature computation will be carried out using its default
            options.

            Non-default feature options and new experimental feature can be
            used by defining a closure. In this case, the closure must define a
            function that receives as an input an image and returns a
            particular feature representation of that image. For example:

                def igo_double_from_std_normalized_intensities(image)
                    image = deepcopy(image)
                    image.normalize_std_inplace()
                    return image.feature_type.igo(double_angles=True)

            See `menpo.image.feature.py` for details more details on
            Menpo's standard image features and feature options.

    Returns
    -------
    feature_image: :class:`menpo.image.MaskedNDImage`
        The resulting feature image.
    """
    if feature_type is not None:
        if type(feature_type) is str:
            image = eval('image.features.' + feature_type + '()')
        elif hasattr(feature_type, '__call__'):
            image = feature_type(image)
        else:
            raise ValueError("feature_type can only be: (1) None, "
                             "(2) a string defining one of Menpo's standard "
                             "image feature_type ('hog', 'igo', etc) "
                             "or (3) a closure defining a non-standard "
                             "feature computation")
    return image


def sparse_hog(image):
    return image.features.hog(mode='sparse')
