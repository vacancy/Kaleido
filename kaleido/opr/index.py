# -*- coding:utf8 -*-
# File   : index.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/28/16 10:00
#
# This file is part of Kaleido
# (c) 2016 vccy.xyz


import numpy as np

from .base import SISOOprNodeBase, SingleOutputOprNodeBase
from ..graph.node import as_opr_func


class IndexOnehot(SingleOutputOprNodeBase):
    __nr_inputs__ = 2

    def __init__(self, src, index, axis, name=None):
        super().__init__(src, index, name=name)
        self._axis = int(axis)

    def _do_fprop(self, env):
        x = self.inputs[0].get_value()
        i = self.inputs[1].get_value()
        assert len(x.shape) == len(i.shape) + 1 and self._axis <= len(x.shape)

        xshape = x.shape
        axis, ndim, alen = self._axis, len(xshape), xshape[self._axis]

        x_hat = np.moveaxis(x, axis, -1).reshape(-1, alen)
        i_hat = i.reshpae(-1)
        y_hat = x_hat[np.arange(x_hat.shape[0]), i_hat]
        y = y_hat.reshape(xshape[:axis] + xshape[axis+1:])

        self.outputs[0].set_value(y)

    def _do_bprop(self, env, idx):
        if idx == 0:
            x = self.inputs[0].get_value()
            i = self.inputs[1].get_value()

            g = self.outputs[0].get_grad()

            xshape = x.shape
            axis, ndim, alen = self._axis, len(xshape), xshape[self._axis]

            i_hat = i.reshpae(-1)
            g_hat = g.reshape(-1)
            r_hat = np.zeros((i.shape[0], alen), dtype=x.dtype)
            r_hat[np.arange(r_hat.shape[0]), i_hat] = g_hat

            r = r_hat.reshape(xshape[:axis] + xshape[axis+1:] + (alen, ))
            r = np.moveaxis(r, -1, axis)
            return r
        else:
            return 0


class Flatten2(SISOOprNodeBase):
    def _do_fprop(self, env):
        x = self.inputs[0].get_value()
        y = x.reshape(x.shape[0], -1)
        self.outputs[0].set_value(y)

    def _do_bprop(self, env, idx):
        x = self.inputs[0].get_value()
        g = self.outputs[0].get_grad()
        return g.reshape(x.shape)


index_onehot = as_opr_func(IndexOnehot)
flatten2 = as_opr_func(Flatten2)
