__author__ = 'ja310'

import numpy as np
from pybug.align import TPS
import matplotlib.pyplot as plt

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

x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)
xx, yy = np.meshgrid(x, y)
points = np.array([xx.flatten(1), yy.flatten(1)]).T

dW_dx = tps.transform.jacobian_source(points)

plt.imshow(dW_dx[:,1].reshape((40,40)))