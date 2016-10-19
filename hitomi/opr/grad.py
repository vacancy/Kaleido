# -*- coding:utf8 -*-
# File   : grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/19/16 16:50
# 
# This file is part of Hitomi
# (c) 2016 vccy.xyz

from .base import SingleOutputOprNodeBase
from ..graph.node import as_opr_func


class Gradient(SingleOutputOprNodeBase):
    __nr_inputs__ = 2

    def _auto_name(self, *inputs):
        return 'grad({}, {})'.format(inputs[0].name, inputs[1].name)

    def _do_fprop(self, env):
        self.outputs[0].set_value(self.inputs[1].get_grad())

    def _do_bprop(self, env, idx):
        return 0


grad = as_opr_func(Gradient)
