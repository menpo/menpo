# TODO: document me
def weights(appearance_model, warped_image):
    r"""
    """
    return appearance_model.project(warped_image)


# TODO: document me
def whiten_weights(appearance_model, warped_image):
    r"""
    """
    return appearance_model.project_whiten(warped_image)


# TODO: document me
def quadratic_weights(appearance_model, warped_image):
    r"""
    """
    return appearance_model.project(warped_image)


# TODO: document me
def appearance(appearance_model, warped_image):
    r"""
    """
    return appearance_model.reconstruct(warped_image).as_vector()


def difference(appearance_model, warped_image):
    r"""
    """
    return (warped_image.as_vector() -
            appearance(appearance_model, warped_image))


# TODO: document me
def project_out(appearance_model, warped_image):
    r"""
    """
    diff = warped_image.as_vector() - appearance_model.mean.as_vector()
    return appearance_model.distance_to_subspace_vector(diff).ravel()


# TODO: document me
def probabilistic(appearance_model, warped_image):
    r"""
    """
    diff = warped_image.as_vector() - appearance_model.mean.as_vector()
    po = appearance_model.distance_to_subspace_vector(diff).ravel()
    return po + appearance_model.project_whitened_vector(diff).ravel()

