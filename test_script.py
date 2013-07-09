__author__ = 'ja310'

import numpy as np
from pybug.align import TPS

# landmarks used in Principal Warps paper
src_landmarks = np.array([[3.6929, 10.3819],
                          [6.5827,  8.8386],
                          [6.7756, 12.0866],
                          [4.8189, 11.2047],
                          [5.6969, 10.0748]])

tgt_landmarks = np.array([[3.9724, 6.5354],
                          [6.6969, 4.1181],
                          [6.5394, 7.2362],
                          [5.4016, 6.4528],
                          [5.7756, 5.1142]])

tps = TPS(src_landmarks, tgt_landmarks)
tps.transform.jacobian_source(np.concatenate((src_landmarks,tgt_landmarks),
                                             axis=0))