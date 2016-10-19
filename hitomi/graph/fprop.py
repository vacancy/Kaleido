# -*- coding:utf8 -*-
# File   : fprop.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/17/16 13:09
# 
# This file is part of Hitomi
# (c) 2016 vccy.xyz

from collections import deque


class Function(object):
    def __init__(self, outputs, all_oprs, env):
        self._outputs = outputs
        self._all_oprs = all_oprs
        self._env = env

    def __call__(self, *args, **kwargs):
        from ..opr.netsrc import PlaceHolder

        for opr in self._all_oprs:
            for o in opr.outputs:
                o.clear_state()
            if isinstance(opr, PlaceHolder):
                assert opr.name in kwargs, 'missing input value for {}'.format(opr.name)
                opr.set_value(kwargs[opr.name])

        for opr in self._all_oprs:
            opr.fprop(self._env)
        for opr in self._all_oprs:
            opr.bprop(self._env)
        return [o.get_value() for o in self._outputs]


class CompGraph(object):
    def __init__(self):
        self._env = None

    def compile(self, outputs):
        all_oprs = TopoSorter(outputs).sort()
        return Function(outputs, all_oprs, self._env)


class TopoSorter(object):
    def __init__(self, targets, initial_visited=None):
        self._targets = targets
        self._initial_visited = initial_visited or []
        self._visited = set()
        self._all_related_oprs = list()
        self._sorted = None

    def _gen_all_related_oprs(self, dest_var):
        if dest_var in self._visited:
            return

        queue = deque()
        queue.append(dest_var.owner_opr)
        while len(queue) != 0:
            wrt = queue.popleft()
            self._all_related_oprs.append(wrt)
            for i in wrt.inputs:
                opr = i.owner_opr
                if opr not in self._visited:
                    self._visited.add(opr)
                    queue.append(opr)

    def _topo_sort(self):
        oprs = self._all_related_oprs
        degrees = {o: len(o.inputs) for o in oprs}
        out_edges = {o: [] for o in oprs}
        for o in oprs:
            for i in o.inputs:
                out_edges[i.owner_opr].append(o)

        queue = deque()
        for o in oprs:
            if degrees[o] == 0:
                queue.append(o)
        while len(queue) != 0:
            i = queue.popleft()
            self._sorted.append(i)
            for o in out_edges[i]:
                degrees[o] -= 1
                if degrees[o] == 0:
                    queue.append(o)

    def sort(self):
        if self._sorted is not None:
            return self._sorted

        self._sorted = list()

        for i in self._initial_visited:
            if i not in self._visited:
                self._visited.add(i)
                self._sorted.append(i.owner_opr)

        for o in self._targets:
            self._gen_all_related_oprs(o)
        self._topo_sort()

        return self._sorted
