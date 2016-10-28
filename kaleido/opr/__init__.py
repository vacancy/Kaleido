# -*- coding:utf8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/16 15:40
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz


from .arith import add, sub, neg, mul, div, pow, exp, log, tanh, sum, max, min, matmul, ge, gt, eq
from .arith import shapeof, shapeidx, update
from .cnn import conv2d, pooling2d
from .index import index_onehot, flatten2
from .reduce import reduce_max, reduce_min, reduce_sum
from .netsrc import placeholder, parameter, Parameter
from .grad import grad
