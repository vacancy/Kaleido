# -*- coding:utf8 -*-
# File   : cnn.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/27/16 20:40
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz


from .base import SISOOprNodeBase, SingleOutputOprNodeBase, get_2dshape
from ..graph.node import as_opr_func
from ..opr_kernel import cnn as cnn_kernel


class Conv2D(SingleOutputOprNodeBase):
    __nr_inputs__ = 2

    def __init__(self, src, kernel, padding=0, stride=1, name=None):
        super().__init__(src, kernel, name=name)

        self._padding = get_2dshape(padding)
        self._stride = get_2dshape(stride)

    def _do_fprop(self, env):
        x = self.inputs[0].get_value()
        k = self.inputs[1].get_value()

        ph, pw = self._padding
        sh, sw = self._stride
        y = cnn_kernel.conv2d_forward(x, k, ph, pw, sh, sw)

        self.outputs[0].set_value(y)

    def _do_bprop(self, env, idx):
        x = self.inputs[0].get_value()
        k = self.inputs[1].get_value()

        ph, pw = self._padding
        sh, sw = self._stride

        g = self.outputs[0].get_grad()

        if idx == 0:
            return cnn_kernel.conv2d_backward_data(g, x, k, ph, pw, sh, sw)
        else:
            return cnn_kernel.conv2d_backward_kernel(g, x, k, ph, pw, sh, sw)


class Pooling2D(SISOOprNodeBase):
    def __init__(self, src, kernel, padding=0, stride=None, method='MAX', name=None):
        super().__init__(src, name=name)

        if stride is None:
            stride = kernel

        method = method.upper()
        assert method in ('MAX', 'AVG')

        self._kernel = get_2dshape(kernel)
        self._padding = get_2dshape(padding)
        self._stride = get_2dshape(stride)
        self._method = method

    def _do_fprop(self, env):
        x = self.inputs[0].get_value()

        kh, kw = self._kernel
        ph, pw = self._padding
        sh, sw = self._stride
        method = self._method
        y = cnn_kernel.pooling2d_forward(x, kh, kw, ph, pw, sh, sw, method)

        self.outputs[0].set_value(y)

    def _do_bprop(self, env, idx):
        x = self.inputs[0].get_value()

        kh, kw = self._kernel
        ph, pw = self._padding
        sh, sw = self._stride
        method = self._method

        g = self.outputs[0].get_grad()

        return cnn_kernel.pooling2d_backward(g, x, kh, kw, ph, pw, sh, sw, method)


conv2d = as_opr_func(Conv2D)
pooling2d = as_opr_func(Pooling2D)
