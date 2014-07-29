import json
import numpy as np


def LJSONExporter(file_handle, landmark_group):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.

    Writes out the LJSON format which is a verbose format that closely
    resembles the landmark group format. It describes semantic labels and
    connectivity between labels. The first axis of the format represents
    the image y-axis and is consistent with ordering within Menpo.

    Parameters
    ----------
    file_handle : `file`-like object
        The file to write in to
    landmark_group : map:`LandmarkGroup`
        The landmark group to write out.
    """
    lg_json = landmark_group.tojson()
    # Add version string
    lg_json['version'] = 1
    return json.dump(lg_json, file_handle)


def PTSExporter(file_handle, landmark_group):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.

    Writes out the PTS format which is a very simple format that does not
    contain any semantic labels. We assume that the PTS format has been created
    using Matlab and so use 1-based indexing and put the image x-axis as the
    first coordinate (which is the second axis within Menpo).

    Parameters
    ----------
    file_handle : `file`-like object
        The file to write in to
    landmark_group : map:`LandmarkGroup`
        The landmark group to write out.
    """
    pts = landmark_group.lms.points
    # Swap the x and y axis and add 1 to undo our processing
    # We are assuming (as on import) that the landmark file was created using
    # Matlab which is 1 based
    pts = pts[:, [1, 0]] + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(file_handle, pts, delimiter=' ', header=header, footer='}',
               fmt='%.3f', comments='')
