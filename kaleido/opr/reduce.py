# -*- coding:utf8 -*-
# File   : reduce.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/28/16 10:24
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz

import numpy as np

from itertools import chain

from .base import SISOOprNodeBase
from ..graph.node import as_opr_func


class ReduceOprNodeBase(SISOOprNodeBase):
    def __init__(self, src, axis, keepdims, name=None):
        super().__init__(src, name=name)
        self._axis = int(axis)
        self._keepdims = bool(keepdims)

    def _do_fprop(self, env):
        x = self.inputs[0].get_value()
        self.outputs[0].set_value(self._do_reduce_fprop(x, self._axis, self._keepdims, env))

    def _do_bprop(self, env, idx):
        g = self.outputs[0].get_grad()
        x = self.inputs[0].get_value()

        assert self._axis < len(x.shape), (x.shape, self._axis)

        xshape = x.shape
        axis, ndim, alen = self._axis, len(xshape), xshape[self._axis]

        g_hat = g.reshape(-1)
        r_hat = np.zeros((g_hat.shape[0], alen), dtype=x.dtype)
        r_hat = self._do_reduce_bprop(g_hat, x, axis, env)
        r = r_hat.reshape(xshape[:axis] + xshape[axis + 1:] + (alen,))
        r = np.moveaxis(r, -1, axis)
        return r

    def _do_reduce_fprop(self, x, axis, keepdims, env):
        raise NotImplementedError()

    def _do_reduce_bprop(self, g, x, r_hat, axis, env):
        raise NotImplementedError()


class ReduceMax(ReduceOprNodeBase):
    def _do_reduce_fprop(self, x, axis, keepdims, env):
        return np.max(x, axis=axis, keepdims=keepdims)

    def _do_reduce_bprop(self, g, x, r_hat, axis, env):
        n = r_hat.shape[0]
        r_hat[np.arange(n), np.argmax(x, axis=axis).reshape(-1)] = g


class ReduceMin(ReduceOprNodeBase):
    def _do_reduce_fprop(self, x, axis, keepdims, env):
        return np.min(x, axis=axis, keepdims=keepdims)

    def _do_reduce_bprop(self, g, x, r_hat, axis, env):
        n = r_hat.shape[0]
        r_hat[np.arange(n), np.argmin(x, axis=axis).reshape(-1)] = g


class ReduceSum(ReduceOprNodeBase):
    def _do_reduce_fprop(self, x, axis, keepdims, env):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def _do_reduce_bprop(self, g, x, r_hat, axis, env):
        n = r_hat.shape[0]
        r_hat[np.arange(n), :] = g

reduce_max = as_opr_func(ReduceMax)
reduce_min = as_opr_func(ReduceMin)
reduce_sum = as_opr_func(ReduceSum)
