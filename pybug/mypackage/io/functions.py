__author__ = 'ja310'

import numpy as np
import csv


def laread(filename):
    ext = str.rsplit(filename, '.', 1)[-1]
    if ext == 'pts':
        strdata = laread_pts(filename)
    elif ext == 'txt':
        strdata = laread_txt(filename)
    else:
        strdata = []
    numdata = np.array(strdata, dtype=np.float)
    return numdata


def laread_pts(filename):
    f = open(filename, 'r')
    for line in f:
        if line == '{\n':
            break
    data = []
    for line in csv.reader(f, delimiter=' '):
        if line != ['}']:
            data.append([line[0], line[1]])
            # data.append(line) more neat but requires .pts files to be
            # properly written !!!
    return data


def laread_txt(filename):
    f = open(filename, 'r')
    data = []
    for line in csv.reader(f, delimiter=','):
        data.append(line)
    return data
