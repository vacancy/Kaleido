# -*- coding:utf8 -*-
# File   : node.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/17/16 13:03
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz

import numpy as np


def as_numpy_array(var):
    if type(var) in (float, int):
        return np.array((var, ))
    var = np.array(var)
    if len(var.shape) == 0:
        return np.array((var, ))
    return var


def as_varnode(var):
    if isinstance(var, VarNode):
        return var
    assert type(var) in (float, int, list, tuple, np.ndarray)

    from ..opr.netsrc import immutable
    return immutable(var)


class VarNodeOprMixin:
    def __unary_op(self, name):
        from .. import opr
        return getattr(opr, name)(self)

    def __lbinary_op(self, other, name):
        from .. import opr
        return getattr(opr, name)(self, other)

    def __rbinary_op(self, other, name):
        from .. import opr
        return getattr(opr, name)(other, self)

    def __add__(self, other):
        return self.__lbinary_op(other, 'add')

    def __radd__(self, other):
        return self.__rbinary_op(other, 'add')

    def __sub__(self, other):
        return self.__lbinary_op(other, 'sub')

    def __rsub__(self, other):
        return self.__rbinary_op(other, 'sub')

    def __mul__(self, other):
        return self.__lbinary_op(other, 'mul')

    def __rmul__(self, other):
        return self.__rbinary_op(other, 'mul')

    def __truediv__(self, other):
        return self.__lbinary_op(other, 'div')

    def __rtruediv__(self, other):
        return self.__rbinary_op(other, 'div')

    def __neg__(self):
        return self.__unary_op('neg')

    def __pow__(self, power, modulo=None):
        return self.__lbinary_op(power, 'pow')

    def __ge__(self, other):
        return self.__lbinary_op(other, 'ge')

    def __gt__(self, other):
        return self.__lbinary_op(other, 'gt')

    def __le__(self, other):
        return self.__rbinary_op(other, 'ge')

    def __lt__(self, other):
        return self.__rbinary_op(other, 'gt')

    def eq(self, other):
        return self.__lbinary_op(other, 'eq')

    @property
    def shape(self):
        return self.__unary_op('shapeof')

    def shapeidx(self, idx):
        return self.__lbinary_op(idx, 'shapeidx')

    def sum(self):
        return self.__unary_op('sum')


class VarNode(VarNodeOprMixin):
    def __init__(self, owner_opr, owner_opr_idx, name=None):
        if name is None:
            name = '{}:{}'.format(owner_opr.name, owner_opr_idx)
        self.__name = name
        self.__name_reset = False
        self.__owner_opr = owner_opr
        self.__owner_opr_idx = owner_opr_idx
        self.__value = None
        self.__grad = None
        self.__need_grad = False

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name_reset = True
        self.__name = name

    @property
    def owner_opr(self):
        return self.__owner_opr

    @property
    def owner_opr_idx(self):
        return self.__owner_opr_idx

    @property
    def need_grad(self):
        return self.__need_grad

    @need_grad.setter
    def need_grad(self, x):
        self.__need_grad = bool(x)

    def get_value(self):
        assert self.__value is not None, 'invalid value {}'.format(str(self))
        return self.__value

    def set_value(self, value):
        self.__value = as_numpy_array(value)

    @property
    def is_value_set(self):
        return self.__value is not None

    def get_grad(self):
        assert self.__grad is not None, 'invalid grad {}'.format(str(self))
        return self.__grad

    def set_or_accumulate_grad(self, grad):
        if type(grad) in (float, int) and grad == 0:
            if self.__grad is None:
                self.__grad = np.zeros_like(self.get_value())
            return

        grad = as_numpy_array(grad)
        assert grad.shape == self.get_value().shape, \
            'invalid grad shape gshape={}, vshape={}'.format(grad.shape, self.get_value().shape)
        if self.__grad is None:
            self.__grad = grad
        else:
            self.__grad += grad

    @property
    def is_grad_set(self):
        return self.__grad is not None

    def clear_state(self):
        self.clear_value()
        self.clear_grad()

    def clear_value(self):
        self.__value = None

    def clear_grad(self):
        self.__grad = None
        self.__need_grad = False

    def __str__(self):
        if not self.__name_reset:
            return self.name
        raw_name = '{}:{}'.format(self.owner_opr.name, self.owner_opr_idx)
        return self.name + '{' + raw_name + '}'

    def __repr__(self):
        return str(self)


class OprNodeBase(object):
    __nr_inputs__ = None
    __nr_outputs__ = None

    def __init__(self, *inputs, name=None):
        self.__auto_name = self._auto_name(*inputs)
        if name is None:
            name = self.__auto_name
            self.__name_reset = False
        else:
            self.__name_reset = True

        self.__name = name
        self.__inputs = []
        self.__outputs = []
        self.__inputs_set = False
        self.__outputs_set = False

        if len(inputs) > 0:
            self.inputs = inputs

    def _auto_name(self, *inputs):
        return '{}@{}'.format(type(self).__name__, hex(id(self)))

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name_reset = True
        self.__name = name

    @property
    def inputs(self):
        assert self.__inputs_set
        return self.__inputs

    @inputs.setter
    def inputs(self, inputs):
        if type(self).__nr_inputs__ is not None:
            assert type(self).__nr_inputs__ == len(inputs)
        self.__inputs_set = True
        self.__inputs = list(map(as_varnode, inputs))

    @property
    def outputs(self):
        if not self.__outputs_set:
            self._init_outputs()
            self.__outputs_set = True
        return self.__outputs

    def _set_outputs(self, outputs):
        self.__outputs = outputs

    def fprop(self, env):
        print('fproping', str(self))
        for i in self.__inputs:
            assert i.is_value_set, 'Got invalid input at opr={}, input={}'.format(str(self), str(i))
        self._do_fprop(env)
        for o in self.__outputs:
            o.get_value()  # try to get value

    def bprop(self, env):
        print('bproping', str(self))
        for idx, i in enumerate(self.__inputs):
            if i.need_grad:
                self.inputs[idx].set_or_accumulate_grad(self._do_bprop(env, idx))

    def _init_outputs(self):
        raise NotImplementedError()

    def _do_fprop(self, env):
        raise NotImplementedError()

    def _do_bprop(self, env, idx):
        raise NotImplementedError()

    def __str__(self):
        if self.__name_reset:
            raw_name = self.__auto_name
            return self.name + '{' + raw_name + '}'
        return self.name

    def __repr__(self):
        return str(self)


def as_opr_func(OprCls):
    def func(*args, **kwargs):
        from kaleido.opr.base import SingleOutputOprNodeBase
        opr = OprCls(*args, **kwargs)
        if issubclass(OprCls, SingleOutputOprNodeBase):
            return opr.outputs[0]
        return opr.outputs
    return func
