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


class ElemwiseOprNodeBase(object):
    pass


class UnaryElemwiseOprNodeBase(ElemwiseOprNodeBase, SISOOprNodeBase):
    pass


class BinaryElemwiseOprNodeBase(ElemwiseOprNodeBase, SingleOutputOprNodeBase):
    __nr_inputs__ = 2
    pass


class Add(BinaryElemwiseOprNodeBase):
    def _do_fprop(self, env):
        a = self.inputs[0].get_value()
        b = self.inputs[1].get_value()
        self.outputs[0].set_value(a+b)

    def _do_bprop(self, env, idx):
        g = self.outputs[0].get_grad()
        return g


class Neg(UnaryElemwiseOprNodeBase):
    def _do_fprop(self, env):
        a = self.inputs[0].get_value()
        self.outputs[0].set_value(-a)

    def _do_bprop(self, env, idx):
        g = self.outputs[0].get_grad()
        return -g


add = as_opr_func(Add)
neg = as_opr_func(Neg)
