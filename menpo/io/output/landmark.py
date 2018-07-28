import json
import itertools
import numpy as np


class _UTF8Encoder(json.JSONEncoder):
    def iterencode(self, obj, **kwargs):
        for chunk in json.JSONEncoder.iterencode(self, obj, **kwargs):
            yield chunk.encode('utf8')


def ljson_exporter(landmarks_object, file_handle, **kwargs):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.

    Writes out the LJSON format which is a verbose format that closely
    resembles the labelled point graph format. It describes semantic
    labels and connectivity between labels. The first axis of the format
    represents the image y-axis and is consistent with ordering within Menpo.

    Parameters
    ----------
    landmarks_object : dict or :map:`LandmarkManager`  or
        :map:`PointCloud` or subclass of :map:`PointCloud`
        The shape to write out.
    file_handle : `file`-like object
        The file to write in to
    """
    try:
        _ = landmarks_object.n_points
        landmark_dict = {'LJSON': landmarks_object}
    except AttributeError:
        # This should be a dict or a Landmark Manager.
        landmark_dict = landmarks_object

    # Add version string
    ljson = {'version': 3}
    groups = {}
    for key, pointcloud in landmark_dict.items():
        lg_json = pointcloud.tojson()

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
        # append to the final ljson dict.
        groups[key] = lg_json

    ljson['groups'] = groups
    return json.dump(ljson, file_handle, indent=4, separators=(',', ': '),
                     sort_keys=True, allow_nan=False, cls=_UTF8Encoder)


def pts_exporter(pointcloud, file_handle, **kwargs):
    r"""
    Given a file handle to write in to (which should act like a Python `file`
    object), write out the landmark data. No value is returned.

    Writes out the PTS format which is a very simple format that does not
    contain any semantic labels. We assume that the PTS format has been created
    using Matlab and so use 1-based indexing and put the image x-axis as the
    first coordinate (which is the second axis within Menpo).

    Note that the PTS file format is only powerful enough to represent a
    basic pointcloud. Any further specialization is lost.

    Parameters
    ----------
    pointcloud : :map:`PointCloud` or subclass
        The pointcloud to write out.
    file_handle : `file`-like object
        The file to write in to
    """
    pts = pointcloud.points
    # Swap the x and y axis and add 1 to undo our processing
    # We are assuming (as on import) that the landmark file was created using
    # Matlab which is 1 based
    pts = pts[:, [1, 0]] + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(file_handle, pts, delimiter=' ', header=header, footer='}',
               fmt='%.3f', comments='')
