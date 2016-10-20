# -*- coding:utf8 -*-
# File   : fprop.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/17/16 13:09
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz

from collections import deque


class Function(object):
    def __init__(self, outputs, comp_graph):
        self._outputs = outputs
        self._cg = comp_graph

    def __call__(self, *args, **kwargs):
        cg = self._cg

        from ..opr.netsrc import PlaceHolder

        for opr in cg.all_oprs:
            for o in opr.outputs:
                o.clear_state()
            if isinstance(opr, PlaceHolder):
                assert opr.name in kwargs, 'missing input value for {}'.format(opr.name)
                opr.set_value(kwargs[opr.name])

        for opr in cg.oprs_pre_grad:
            opr.fprop(cg.env)
        for loss, grad_oprs in cg.grad_wrts.items():
            rel_oprs = cg.label_wrts(loss)

            loss.set_or_accumulate_grad(1)
            for opr in reversed(rel_oprs):
                opr.bprop(cg.env)
            for gopr in grad_oprs:
                gopr.fprop(cg.env)
        for opr in cg.oprs_post_grad:
            opr.fprop(cg.env)

        return [o.get_value() for o in self._outputs]


class CompGraph(object):
    def __init__(self, env=None):
        self._env = None

        self._outputs = None
        self._all_oprs = None
        self._oprs_pre_grad = None
        self._oprs_post_grad = None
        self._grad_oprs = []
        self._grad_wrts = {}
        self._out_edges = None

        self._compiled = False

    @property
    def env(self):
        return self._env

    @property
    def outputs(self):
        return self._outputs

    @property
    def all_oprs(self):
        return self._all_oprs

    @property
    def grad_wrts(self):
        return self._grad_wrts

    @property
    def oprs_pre_grad(self):
        return self._oprs_pre_grad

    @property
    def grad_oprs(self):
        return self._grad_oprs

    @property
    def oprs_post_grad(self):
        return self._oprs_post_grad

    @property
    def compiled(self):
        return self._compiled

    def compile(self, outputs):
        assert not self.compiled, 'can not compile a comp_graph twice'

        self._outputs = outputs
        self._all_oprs, self._out_edges = TopoSorter(outputs).sort()
        self._check_grad_dependency()
        self._find_grad_oprs()
        self._split_oprs()

        return Function(outputs, self)

    def find_opr(self, output, name):
        all_oprs, _ = TopoSorter(output).sort()
        for o in all_oprs:
            if o.name == name:
                return o
        return None

    def find_all_oprs(self, outputs):
        all_oprs, _ = TopoSorter(outputs).sort()
        return all_oprs

    def _check_grad_dependency(self):
        from ..opr.grad import Gradient

        visited = set()
        for opr in self._all_oprs:
            if isinstance(opr, Gradient):
                assert opr not in visited, 'do not support order-2 gradient computation'
                visited.add(opr)

            if opr in visited:
                for o in opr.outputs:
                    if o in self._out_edges:
                        for kpr in self._out_edges[o]:
                            visited.add(kpr)

    def _find_grad_oprs(self):
        from ..opr.grad import Gradient

        for o in self._all_oprs:
            if isinstance(o, Gradient):
                self._grad_wrts.setdefault(o.inputs[0], set())
                self._grad_wrts[o.inputs[0]].add(o)
                self._grad_oprs.append(o)

    def _split_oprs(self):
        self._oprs_pre_grad, _ = TopoSorter(list(self._grad_wrts)).sort()
        for loss in self._grad_wrts:
            wrts = [opr.inputs[1].owner_opr for opr in self._grad_wrts[loss]]
            all_deps, _ = TopoSorter([loss]).sort()
            all_deps = {opr for opr in all_deps}
            for wrt in wrts:
                assert wrt in all_deps, 'loss {} does not depend on w.r.t. value {}'.format(loss, wrt)
        self._oprs_post_grad, _ = TopoSorter(self._outputs, self._oprs_pre_grad + self._grad_oprs).sort()

    def label_wrts(self, loss):
        assert len(loss.get_value().shape) == 1 and loss.get_value().shape[0] == 1, \
            'only support compute grad of a scalar, got {}'.format(loss.get_value().shape)

        for opr in self._all_oprs:
            for o in opr.outputs:
                o.clear_grad()

        wrts = [opr.inputs[1] for opr in self._grad_wrts[loss]]
        all_oprs, out_edges = TopoSorter([loss], [wrt.owner_opr for wrt in wrts]).sort()

        visited = set()
        queue = deque()
        for v in wrts:
            v.need_grad = True
            visited.add(v)
            queue.append(v)

        while len(queue) != 0:
            v = queue.popleft()
            if v in out_edges:
                for opr in out_edges[v]:
                    assert opr in all_oprs
                    for o in opr.outputs:
                        o.need_grad = True
                        if o not in visited:
                            visited.add(o)
                            queue.append(o)

        for o in loss.owner_opr.outputs:
            if o != loss:
                o.need_grad = False

        return all_oprs


class TopoSorter(object):
    def __init__(self, targets, initial_visited=None):
        self._targets = targets
        self._initial_visited = initial_visited or []
        self._visited = set()
        self._all_related_oprs = list()
        self._sorted = None
        self._out_edges = None

    def _gen_all_related_oprs(self, dest_var):
        if dest_var.owner_opr in self._visited:
            return

        queue = deque()
        self._visited.add(dest_var.owner_opr)
        queue.append(dest_var.owner_opr)
        while len(queue) != 0:
            wrt = queue.popleft()
            self._all_related_oprs.append(wrt)
            for i in wrt.inputs:
                opr = i.owner_opr
                if opr not in self._visited:
                    self._visited.add(opr)
                    queue.append(opr)

    def _topo_sort(self, oprs):
        degrees = {o: len(o.inputs) for o in oprs}

        out_edges = {}
        for opr in oprs:
            for i in opr.inputs:
                out_edges.setdefault(i, [])
                out_edges[i].append(opr)

        for ipr in self._initial_visited:
            for o in ipr.outputs:
                if o in out_edges:
                    for opr in out_edges[o]:
                        if opr in degrees:
                            degrees[opr] -= 1

        queue = deque()
        for opr in oprs:
            if degrees[opr] == 0:
                queue.append(opr)

        while len(queue) != 0:
            opr = queue.popleft()
            self._sorted.append(opr)
            for o in opr.outputs:
                if o in out_edges:
                    for ipr in out_edges[o]:
                        degrees[ipr] -= 1
                        if degrees[ipr] == 0:
                            queue.append(ipr)
        self._out_edges = out_edges

        for opr in degrees:
            assert degrees[opr] == 0, 'topo sort failed for oprnode {}'.format(opr)

    def sort(self):
        if self._sorted is not None:
            return self._sorted, self._out_edges

        self._sorted = list()

        for ipr in self._initial_visited:
            if ipr not in self._visited:
                self._visited.add(ipr)
        for o in self._targets:
            self._gen_all_related_oprs(o)

        self._topo_sort(self._all_related_oprs)

        return self._sorted, self._out_edges
