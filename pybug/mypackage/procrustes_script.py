__author__ = 'ja310'

import numpy as np
from pybug.align import GeneralizedProcrustesAnalysis
import os
from pybug.mypackage.io import functions as myfn


flr0 = '/data'
flr1 = 'db'
flr2 = 'lfpw'
flr3 = 'train'
path = os.path.join(flr0, flr1, flr2, flr3, '')

lan_ext = ['.pts', '.txt']

lan_list = [myfn.laread(os.path.join(path,f)) for f in os.listdir(path) if
            os.path.splitext(f)[-1] in lan_ext]

gpa = GeneralizedProcrustesAnalysis(lan_list)

aligned_lan_list = [trans.apply(lan) for trans, lan in zip(gpa.transforms,
                                                           gpa.sources)]