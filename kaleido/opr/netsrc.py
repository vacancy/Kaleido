# -*- coding:utf8 -*-
# File   : netsrc.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/16 15:40
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz

from .base import SingleOutputOprNodeBase
from ..graph.node import as_opr_func


class NetSrcOprNodeBase(SingleOutputOprNodeBase):
    __nr_inputs__ = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = []


class PlaceHolder(NetSrcOprNodeBase):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self._value = None

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def _do_fprop(self, env):
        self.outputs[0].set_value(self._value)

    def _do_bprop(self, env, idx):
        return 0


class Parameter(NetSrcOprNodeBase):
    def __init__(self, value, name=None):
        super().__init__(name=name)
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def _do_fprop(self, env):
        self.outputs[0].set_value(self._value)

    def _do_bprop(self, env, idx):
        return 0


class Immutable(NetSrcOprNodeBase):
    def __init__(self, value, name=None):
        super().__init__(name=name)
        self._value = value

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def _do_fprop(self, env):
        self.outputs[0].set_value(self._value)

    def _do_bprop(self, env, idx):
        return 0



placeholder = as_opr_func(PlaceHolder)
parameter = as_opr_func(Parameter)
immutable = as_opr_func(Immutable)
