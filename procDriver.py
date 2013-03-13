from ibugMM.alignment.rigid import Procrustes
import numpy as np

source1 = np.array([[ 0, 0],[ 0, 1],[ 1, 1],[ 1, 0]],dtype=np.float64)
source2 = np.array([[-1,-1],[-1,10],[10,10],[10,-1]],dtype=np.float64)
source3 = np.array([[-2, -2],[-2, 3],[ 3, 3],[ 3, -2]],dtype=np.float64)
proc = Procrustes([source1,source2,source3])

