__author__ = 'ja310'

import numpy as np
from pybug.align import TPS

src_landmarks = np.array([[0, 1.0],
                          [-1, 0.0],
                          [0, -1.0],
                          [1, 0.0]])

tgt_landmarks = np.array([[0, 0.75],
                          [-1, 0.25],
                          [0, -1.25],
                          [1, 0.25]])

tps = TPS(src_landmarks, tgt_landmarks)
np.allclose(tps.transform.apply(src_landmarks), tgt_landmarks)
tps.view()

tps.transform.jacobian_source(np.concatenate((src_landmarks,tgt_landmarks),
                                             axis=0))