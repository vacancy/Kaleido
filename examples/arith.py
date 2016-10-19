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

if __name__ == '__main__':
    a = opr.placeholder('a')
    b = opr.placeholder('b')
    c = opr.add(a, b)

    func = CompGraph().compile([c])

    cv,  = func(a=1, b=2)
    print('{}+{}={}'.format(1, 2, cv))
    cv,  = func(a=5, b=4)
    print('{}+{}={}'.format(5, 4, cv))
