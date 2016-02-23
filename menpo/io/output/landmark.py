import json
import itertools
import numpy as np


def LJSONExporter(landmark_group, file_handle, **kwargs):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.

    Writes out the LJSON format which is a verbose format that closely
    resembles the landmark group format. It describes semantic labels and
    connectivity between labels. The first axis of the format represents
    the image y-axis and is consistent with ordering within Menpo.

    Parameters
    ----------
    landmark_group : map:`LandmarkGroup`
        The landmark group to write out.
    file_handle : `file`-like object
        The file to write in to
    """
    lg_json = landmark_group.tojson()
    # Add version string
    lg_json['version'] = 2

    # Convert nan values to None so that json correctly maps them to 'null'
    points = lg_json['landmarks']['points']
    # Flatten list
    try:
        ndim = len(points[0])
    except IndexError:
        ndim = 0
    filtered_points = [None if np.isnan(x) else x
                       for x in itertools.chain(*points)]
    # Recreate tuples
    if ndim == 2:
        lg_json['landmarks']['points'] = list(zip(filtered_points[::2],
                                              filtered_points[1::2]))
    elif ndim == 3:
        lg_json['landmarks']['points'] = list(zip(filtered_points[::3],
                                              filtered_points[1::3],
                                              filtered_points[2::3]))
    else:
        lg_json['landmarks']['points'] = []

    return json.dump(lg_json, file_handle, indent=4, separators=(',', ': '),
                     sort_keys=True, allow_nan=False)


def PTSExporter(landmark_group, file_handle, **kwargs):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.

    Writes out the PTS format which is a very simple format that does not
    contain any semantic labels. We assume that the PTS format has been created
    using Matlab and so use 1-based indexing and put the image x-axis as the
    first coordinate (which is the second axis within Menpo).

    Parameters
    ----------
    landmark_group : map:`LandmarkGroup`
        The landmark group to write out.
    file_handle : `file`-like object
        The file to write in to
    """
    pts = landmark_group.lms.points
    # Swap the x and y axis and add 1 to undo our processing
    # We are assuming (as on import) that the landmark file was created using
    # Matlab which is 1 based
    pts = pts[:, [1, 0]] + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(file_handle, pts, delimiter=' ', header=header, footer='}',
               fmt='%.3f', comments='')
