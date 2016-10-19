# -*- coding:utf8 -*-
# File   : arith.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/16 15:44
# 
# This file is part of Hitomi
# (c) 2016 vccy.xyz

from .base import SISOOprNodeBase, SingleOutputOprNodeBase
from ..graph.node import as_opr_func

import numpy as np


class ElemwiseOprNodeBase(object):
    pass


def get_not_one_axis(shape):
    return tuple(filter(lambda x: x > 1, shape))


class AutoBroadcaster(object):
    def __init__(self, var, tshape):
        self._a_axises = []
        self._b_axises = []
        self._oshape = var.shape
        self._value = self.broadcast(var, tshape)

    @property
    def value(self):
        return self._value

    def broadcast(self, var, tshape):
        shape1 = get_not_one_axis(var.shape)
        shape2 = tshape

        i, j = 0, 0
        n, m = len(shape1), len(shape2)

        dshape = []
        while i < n and j < m:
            if shape1[i] == shape2[j]:
                dshape.append(shape1[i])
                self._a_axises.append(j)
                i += 1
                j += 1
            else:
                dshape.append(1)
                self._b_axises.append(j)
                j += 1

        for k in range(j, m):
            dshape.append(1)
            self._b_axises.append(k)

        assert i == n, 'can not perform auto broadcast from {} to {}'.format(var.shape, tshape)
        return var.reshape(dshape)

    def inv_broadcast(self, var):
        var = var.transpose(self._a_axises + self._b_axises)
        var = var.reshape(var.shape[:len(self._a_axises)] + (-1,))
        var = var.sum(axis=len(var.shape) - 1)
        var = var.reshape(self._oshape)
        return var


class UnaryElemwiseOprNodeBase(ElemwiseOprNodeBase, SISOOprNodeBase):
    def _do_fprop(self, env):
        x = self.inputs[0].get_value()
        y = self._do_unary_fprop(env, x)
        self.outputs[0].set_value(y)

    def _do_unary_fprop(self, env, x):
        raise NotImplementedError()

    def _do_bprop(self, env, idx):
        assert idx == 0
        gy = self.outputs[0].get_grad()
        gx = self._do_unary_bprop(env, gy)
        return gx

    def _do_unary_bprop(self, env, g):
        raise NotImplementedError()


class BinaryElemwiseOprNodeBase(ElemwiseOprNodeBase, SingleOutputOprNodeBase):
    __nr_inputs__ = 2
    _broadcaster = None
    _broadcaster_idx = -1
    _var_a = None
    _var_b = None

    def _clear_broadcast_state(self):
        self._broadcaster = None
        self._broadcaster_idx = -1
        self._var_a = self._var_b = None

    def _broadcast(self, a, b):
        n, m = len(get_not_one_axis(a.shape)), len(get_not_one_axis(b.shape))
        if n > m:
            self._broadcaster_idx = 1
            self._broadcaster = AutoBroadcaster(b, a.shape)
            b = self._broadcaster.value
        else:
            self._broadcaster_idx = 0
            self._broadcaster = AutoBroadcaster(a, b.shape)
            a = self._broadcaster.value
        return a, b

    def _inv_broadcast(self, var, axis):
        assert self._broadcaster_idx in (0, 1)
        if self._broadcaster_idx == axis:
            var = self._broadcaster.inv_broadcast(var)
        return var

    def _do_fprop(self, env):
        a = self.inputs[0].get_value()
        b = self.inputs[1].get_value()
        self._clear_broadcast_state()
        a, b = self._broadcast(a, b)
        self._var_a, self._var_b = a, b
        c = self._do_binary_fprop(env, a, b)
        self.outputs[0].set_value(c)

    def _do_binary_fprop(self, env, a, b):
        raise NotImplementedError()

    def _do_bprop(self, env, idx):
        g = self.outputs[0].get_grad()
        g = self._do_binary_bprop(env, idx, self._var_a, self._var_b, g)
        g = self._inv_broadcast(g, idx)
        return g

    def _do_binary_bprop(self, env, idx, a, b, g):
        raise NotImplementedError()


class Add(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        return a + b

    def _do_binary_bprop(self, env, idx, a, b, g):
        return g


class Sub(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        return a - b

    def _do_binary_bprop(self, env, idx, a, b, g):
        if idx == 0:
            return g
        else:
            return -g


class Neg(UnaryElemwiseOprNodeBase):
    def _do_unary_fprop(self, env, x):
        return -x

    def _do_unary_bprop(self, env, g):
        return -g


class Mul(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        return a * b

    def _do_binary_bprop(self, env, idx, a, b, g):
        if idx == 0:
            return g * b
        else:
            return g * a


class Div(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        return a / b

    def _do_binary_bprop(self, env, idx, a, b, g):
        if idx == 0:
            return g / self.inputs[1].get_value()
        else:
            return -g * a / (b ** 2)


class Pow(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        return a ** b

    def _do_binary_bprop(self, env, idx, a, b, g):
        if idx == 0:
            return b * (a ** b - 1)
        else:
            return a ** b * np.log(a)


class Sum(UnaryElemwiseOprNodeBase):
    def _do_unary_fprop(self, env, x):
        return x.sum()

    def _do_unary_bprop(self, env, g):
        return np.ones_like(self.inputs[0].get_value()) * g


class Max(BinaryElemwiseOprNodeBase):
    _res_mask = -1

    def _do_binary_fprop(self, env, a, b):
        self._res_mask = m = b > a
        return m * b + (1 - m) * a

    def _do_binary_bprop(self, env, idx, a, b, g):
        if idx == 1:
            return self._res_mask * g
        else:
            return (1 - self._res_mask) * g


class Min(BinaryElemwiseOprNodeBase):
    _res_mask = -1

    def _do_binary_fprop(self, env, a, b):
        self._res_mask = m = b < a
        return m * b + (1 - m) * a

    def _do_binary_bprop(self, env, idx, a, b, g):
        if idx == 1:
            return self._res_mask * g
        else:
            return (1 - self._res_mask) * g


class GreaterEqual(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        self._res_idx = a >= b
        return self._res_idx.astype(np.int32)

    def _do_binary_bprop(self, env, idx, a, b, g):
        # print('warning: zero grad opr {}'.format(self.name))
        return 0


class GreaterThan(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        self._res_idx = a > b
        return self._res_idx.astype(np.int32)

    def _do_binary_bprop(self, env, idx, a, b, g):
        # print('warning: zero grad opr {}'.format(self.name))
        return 0


class Equal(BinaryElemwiseOprNodeBase):
    def _do_binary_fprop(self, env, a, b):
        self._res_idx = a == b
        return self._res_idx.astype(np.int32)

    def _do_binary_bprop(self, env, idx, a, b, g):
        # print('warning: zero grad opr {}'.format(self.name))
        return 0


class ShapeOf(SISOOprNodeBase):
    def _do_fprop(self, env):
        self.outputs[0].set_value(self.inputs[0].get_value().shape)

    def _do_bprop(self, env, idx):
        return 0


class ShapeIdx(SingleOutputOprNodeBase):
    __nr_inputs__ = 2

    def _do_fprop(self, env):
        idx = self.inputs[1].get_value()
        self.outputs[0].set_value(self.inputs[0].get_value().shape[idx[0]])

    def _do_bprop(self, env, idx):
        # print('warning: zero grad opr {}'.format(self.name))
        return 0


class MatrixMul(SingleOutputOprNodeBase):
    __nr_inputs__ = 2

    def _do_fprop(self, env):
        a, b = map(lambda x: x.get_value(), self.inputs)
        assert len(a.shape) == 2 and len(b.shape) == 2 and a.shape[1] == b.shape[0]
        self.outputs[0].set_value(np.matmul(a, b))

    def _do_bprop(self, env, idx):
        a, b = map(lambda x: x.get_value(), self.inputs)
        g = self.outputs[0].get_grad()
        if idx == 0:
            return np.matmul(g, b.T)
        else:
            return np.matmul(a.T, g)


class Update(BinaryElemwiseOprNodeBase):
    def _init_outputs(self):
        from .netsrc import Parameter
        assert isinstance(self.inputs[0].owner_opr, Parameter), self.inputs[0].owner_opr
        super()._init_outputs()

    def _do_binary_fprop(self, env, a, b):
        self.inputs[0].owner_opr.set_value(a * 0 + b)
        return b

    def _do_binary_bprop(self, env, idx, a, b, g):
        if idx == 0:
            return 0
        else:
            return g

add = as_opr_func(Add)
sub = as_opr_func(Sub)
neg = as_opr_func(Neg)
mul = as_opr_func(Mul)
div = as_opr_func(Div)
pow = as_opr_func(Pow)
sum = as_opr_func(Sum)
max = as_opr_func(Max)
min = as_opr_func(Min)
ge = as_opr_func(GreaterEqual)
gt = as_opr_func(GreaterThan)
eq = as_opr_func(Equal)
matmul = as_opr_func(MatrixMul)
shapeidx = as_opr_func(ShapeIdx)
shapeof = as_opr_func(ShapeOf)
update = as_opr_func(Update)
