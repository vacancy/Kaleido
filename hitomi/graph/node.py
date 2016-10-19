# -*- coding:utf8 -*-
# File   : node.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/17/16 13:03
# 
# This file is part of Hitomi
# (c) 2016 vccy.xyz


class VarNode(object):
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
        assert self.__value is not None, 'Getting invalid value {}'.format(str(self))
        return self.__value

    def set_value(self, value):
        self.__value = value

    @property
    def is_value_set(self):
        return self.__value is not None

    def get_grad(self):
        assert self.__grad is not None, 'Getting invalid grad {}'.format(str(self))
        return self.__grad

    def set_grad(self, grad):
        self.__grad = grad

    @property
    def is_grad_set(self):
        return self.__grad is not None

    def clear_state(self):
        self.__value = None
        self.__grad = None
        self.__need_grad = False

    def __str__(self):
        if not self.__name_reset:
            return 'Var:' + self.name
        raw_name = '{}:{}'.format(self.owner_opr.name, self.owner_opr_idx)
        return 'Var:' + self.name + '{' + raw_name + '}'


class OprNodeBase(object):
    __nr_inputs__ = None
    __nr_outputs__ = None

    def __init__(self, *inputs, name=None):
        if name is None:
            name = '{}@{}'.format(type(self).__name__, hex(id(self)))

        self.__name = name
        self.__inputs = []
        self.__outputs = []
        self.__inputs_set = False
        self.__outputs_set = False

        if len(inputs) > 0:
            self.inputs = inputs

    @property
    def name(self):
        return self.__name

    @property
    def inputs(self):
        assert self.__inputs_set
        return self.__inputs

    @inputs.setter
    def inputs(self, inputs):
        if type(self).__nr_inputs__ is not None:
            assert type(self).__nr_inputs__ == len(inputs)
        self.__inputs_set = True
        self.__inputs = inputs

    @property
    def outputs(self):
        if not self.__outputs_set:
            self._init_outputs()
            self.__outputs_set = True
        return self.__outputs

    def _set_outputs(self, outputs):
        self.__outputs = outputs

    def fprop(self, env):
        for i in self.__inputs:
            assert i.is_value_set, 'Got invalid input at opr={}, input={}'.format(str(self), str(i))
        self._do_fprop(env)
        for o in self.__outputs:
            o.get_value()  # try to get value

    def bprop(self, env):
        for idx, i in enumerate(self.__inputs):
            if i.need_grad:
                self._do_bprop(env, idx)
                assert i.is_grad_set, 'Got invalid grad at opr={}, input={}'.format(str(self), str(i))

    def _init_outputs(self):
        raise NotImplementedError()

    def _do_fprop(self, env):
        raise NotImplementedError()

    def _do_bprop(self, env, idx):
        raise NotImplementedError()

    def __str__(self):
        raw_name = '{}@{}'.format(type(self).__name__, hex(id(self)))
        return 'Opr:' + self.name + '{' + raw_name + '}'


def as_opr_func(OprCls):
    def func(*args, **kwargs):
        from hitomi.opr.base import SingleOutputOprNodeBase
        opr = OprCls(*args, **kwargs)
        if issubclass(OprCls, SingleOutputOprNodeBase):
            return opr.outputs[0]
        return opr.outputs
    return func
