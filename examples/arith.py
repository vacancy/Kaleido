# -*- coding:utf8 -*-
# File   : arith.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/16 15:49
# 
# This file is part of Hitomi
# (c) 2016 vccy.xyz

from hitomi import opr
from hitomi.graph import CompGraph
import numpy as np

if __name__ == '__main__':
    a = opr.placeholder('a')
    b = opr.placeholder('b')
    c = opr.add(a, b, name='a+b')
    d = opr.add(c, a)
    sum_d = opr.sum(d)
    e = opr.grad(sum_d, b)

    func = CompGraph().compile([c, e])

    print(func(a=np.ones([5]) * 1, b=2))
    print(func(a=5, b=4))

    a = opr.placeholder('a')
    b = opr.placeholder('b')
    c = opr.mul(a, b)
    c = opr.add(c, 1)
    c = opr.mul(c, [10, 10])
    c = opr.max(c, 50)
    d = c
    c = opr.sum(c)
    func = CompGraph().compile([d, c, opr.grad(c, a), opr.grad(c, b)])
    print(func(a=[1, 2], b=3))

    # from IPython import embed; embed()
