import scipy.io

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
          landmarkDict['auto']  = None
          landmarkDict['human'] = None
        else:
          landmarkDict['auto']  = frameData[0][1]
          landmarkDict['human'] = frameData[1][1]
        emotionDict[frameNo] = landmarkDict
      subjectDict[emotionName] = emotionDict
    landmarks[subjectName] = subjectDict
    return landmarks

