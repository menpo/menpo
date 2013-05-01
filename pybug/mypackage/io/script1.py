__author__ = 'ja310'

import os
import matplotlib.pyplot as plt
from pybug.mypackage.io import functions as fn


flr0 = '/data'
flr1 = 'db'
flr2 = 'lfpw'
flr3 = 'train'
path = os.path.join(flr0, flr1, flr2, flr3)

im_ext = ['.png', '.jpg']
im_list = [plt.imread(os.path.join(path, f)) for f in os.listdir(path) if os
.path.splitext(f)[-1] in im_ext]

la_ext = ['.pts', 'txt']
la_list = [fn.laread(os.path.join(path, f)) for f in os.listdir(path) if os
.path.splitext(f)[-1] in la_ext]


fig = plt.figure()
plt.ion()
plt.show()
plt.imshow(im_list[0])