import json
import numpy as np


def LJSONExporter(landmark_group, file_handle):
    lg_json = landmark_group.tojson()
    # Add version string
    lg_json['version'] = 1
    return json.dump(lg_json, file_handle)


def PTSExporter(landmark_group, file_handle):
    pts = landmark_group.lms.points
    # Swap the x and y axis and add 1 to undo our processing
    # We are assuming (as on import) that the landmark file was created using
    # Matlab which is 1 based
    pts = pts[:, [1, 0]] + 1

    header = 'version: 1\nn_points: {}\n{{'.format(pts.shape[0])
    np.savetxt(file_handle, pts, delimiter=' ', header=header, footer='}',
               fmt='%.3f', comments='')
