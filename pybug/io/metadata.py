import scipy.io
import numpy as np
import json

# TODO json landmark format

# TODO Import landmarks (subclass of Importer)

def msc_bu4d_landmarks(path):
    """ A function to import BU4D landmarks in the format used for my MSc 
            thesis. Returns a nested set of dictionaries of format
            [modelName][emotionName][frameName][landmarkType], where landmarkType
        can be either 'auto' (unedited computed landmarks) or 'human' (manually
      improved landmarks). Example:
          landmarks['F001']['Angry']['000']['auto']
      if no landmarks exist the entry is guaranteed to be of type 'None'
    """
    lmfile = scipy.io.loadmat(path)
    matlabLandmarks = lmfile['landmarkDB']
    landmarks = {}
    for subject in matlabLandmarks:
        subjectName = str(subject[0][0])
        subjectData = subject[1]
        subjectDict = {}
        for emotion in subjectData[1]:
            # emotion is repeated - just grab top one
            emotionName = str(emotion[0][1][0])
            emotionDict = {}
            for frame in emotion:
                frameNo = str(frame[2][0])
                frameData = frame[3]
                landmarkDict = {}
                if frameData[0][0] == 0:
                    landmarkDict['auto'] = None
                    landmarkDict['human'] = None
                else:
                    landmarkDict['auto'] = frameData[0][1]
                    landmarkDict['human'] = frameData[1][1]
                emotionDict[frameNo] = landmarkDict
            subjectDict[emotionName] = emotionDict
        landmarks[subjectName] = subjectDict
        return landmarks


class MissingLandmarksError(Exception):
    pass


def medical_landmarks(path):
    try:
        with open(path, 'rb') as f:
            return np.fromfile(
                f, dtype=np.float32)[3:].reshape([-1, 3]).astype(np.double)
    except IOError:
        raise MissingLandmarksError


def json_pybug_landmarks(filepath):
    """ Loads locally stored .jsonlandmarks files that are generated from
    the picker landmarker.
    """
    path_to_lm = filepath + '.jsonlandmarks'
    try:
        with open(path_to_lm, 'rb') as f:
            print 'found landmarks! Importing them'
            return json.load(f)
    except IOError:
        raise MissingLandmarksError
