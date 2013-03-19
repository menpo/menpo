# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from ibugMM.alignment.rigid import Procrustes
import numpy as np

# <codecell>

src_1 = np.array([ 0.0, 0.0,
                   1.0, 0.0,
                   1.0, 1.0,
                   0.0, 1.0]).reshape([-1,2])
src_2 = np.array([ 0.0, 0.0,
                   2.0, 0.0,
                   2.0, 2.0,
                   0.0, 2.0]).reshape([-1,2])
src_3 = np.array([-1.0, 0.0,
                   0.0, 0.0,
                   0.0, 1.0,
                  -1.0, 1.0]).reshape([-1,2])
tgt   = np.array([ 1.0, 0.0,
                   1.0, 1.0,
                   0.0, 1.0,
                   0.0, 0.0]).reshape([-1,2])

proc = Procrustes([src_1], target=tgt)
proc.general_alignment()


