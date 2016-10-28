# -*- coding:utf8 -*-
# File   : cnn.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/27/16 20:54
#
# This file is part of Kaleido
# (c) 2016 vccy.xyz

from kaleido import opr
from kaleido.graph import CompGraph
import numpy as np
import numpy.random as npr

if __name__ == '__main__':
    x = opr.placeholder('x')
    k = opr.placeholder('kernel')
    y = opr.conv2d(x, k)
    y = opr.pooling2d(y, kernel=2)

    func = CompGraph().compile([y])

    xv = npr.normal(size=(1, 5, 4, 4)).astype('float32')
    kv = npr.normal(size=(1, 5, 3, 3)).astype('float32')
    res = func(x=xv, kernel=kv)
    print(res[0].reshape(-1))
    print([np.sum(xv[:, :, 0:3, 0:3] * kv), np.sum(xv[:, :, 0:3, 1:4] * kv),
           np.sum(xv[:, :, 1:4, 0:3] * kv), np.sum(xv[:, :, 1:4, 1:4] * kv)])

    # from IPython import embed; embed()
