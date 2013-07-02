import numpy as np
import scipy.ndimage as ndimage


def map_coordinates_interpolator(image, coords, order, type):
    # Swap x and y for images
    coords[:2] = coords[1::-1]
    concat_coords = [c[np.newaxis, ...] for c in coords]
    concat_coords = np.concatenate(concat_coords, axis=0)
    return ndimage.map_coordinates(image, concat_coords, order=order,
                                   mode=type)


def matlab_interpolator(image, coords, order, type):
    from pybug.matlab.wrapper import mlab
    # Swap x and y back due to syntax of interpn
    coords[:2] = coords[1::-1]
    return mlab.interpn(image, *coords)


def warp(image, template_shape, transform,
         interpolator=matlab_interpolator,
         type='constant', order=1):
    # Swap x and y for images
    dims = list(template_shape)
    dims[:2] = dims[1::-1]
    ranges = [np.arange(dim) for dim in dims]

    # Generate regular grid and flatten for every dimension
    grids = np.meshgrid(*ranges)
    grids = [g.reshape(-1, 1) for g in grids]
    grids = np.hstack(grids)


    # Warp grids
    uv = transform.apply(grids).T
    uv = [np.reshape(x, template_shape) for x in uv]

    # Interpolate according to transformed grid
    warped_image = interpolator(image, uv, order, type)
    warped_image = np.nan_to_num(warped_image)

    return warped_image