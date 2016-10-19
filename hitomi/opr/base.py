# -*- coding:utf8 -*-
# File   : base.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/18/16 15:40
# 
# This file is part of Hitomi
# (c) 2016 vccy.xyz

from ..graph.node import OprNodeBase, VarNode


class OutputNumFixedOprNodeBase(OprNodeBase):
    def _init_outputs(self):
        nr_outputs = type(self).__nr_outputs__
        res = []
        for i in range(nr_outputs):
            res.append(VarNode(self, i))
        self._set_outputs(res)


class SingleOutputOprNodeBase(OutputNumFixedOprNodeBase):
    __nr_outputs__ = 1

    def _init_outputs(self):
        assert type(self).__nr_outputs__ == 1
        self._set_outputs([VarNode(self, 0)])


class SISOOprNodeBase(SingleOutputOprNodeBase):
    __nr_inputs__ = 1
