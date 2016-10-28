# -*- coding:utf8 -*-
# File   : mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/28/16 01:27
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz

import pickle
import joblib
import os
import numpy as np
import numpy.random as npr

from kaleido import opr
from kaleido.graph import CompGraph


def load_data(fname):
    obj = joblib.load(os.path.join('data', fname))
    return obj


def make_data(name, shuffle=True):
    x = load_data(name + '.npy').astype('float32')
    y = load_data(name + '_anno.npy').astype('int32')
    if shuffle:
        l = x.shape[0]
        index = np.arange(l)
        np.random.shuffle(index)
        x, y = (x[index], y[index])
    return {'img': x, 'label': y}


def conv(name, src, cout, cin, k, p=0, s=1, wstd=0.01):
    W = opr.parameter(npr.normal(scale=wstd, size=(cout, cin, k, k)).astype('float32'), name=name + ':W')
    b = opr.parameter(np.zeros(shape=(1, cout, 1, 1), dtype='float32'), name=name + ':b')
    y = opr.conv2d(src, W, padding=p, stride=s, name=name) + b
    return opr.tanh(y)


def pool(src, k, s=None, name=None):
    return opr.pooling2d(src, k, stride=s, name=name)


def view(src):
    return opr.flatten2(src)


def fc(name, src, cout, cin, wstd=0.01, nonlin=True):
    W = opr.parameter(npr.normal(scale=wstd, size=(cin, cout)).astype('float32'), name=name + ':W')
    b = opr.parameter(np.zeros(shape=(1, cout), dtype='float32'), name=name + ':b')
    y = opr.matmul(src, W) + b
    return opr.tanh(y) if nonlin else y


def cross_entropy(pred, label):
    y = -opr.log(pred)
    y_index = opr.index_onehot(y, label, axis=2)
    return y_index.sum()


def softmax(src, axis):
    exp_src = opr.exp(src)
    exp_sum = opr.reduce_sum(exp_src, axis=axis, keepdims=True)
    pred = exp_src / exp_sum
    return pred


def make_net():
    img = opr.placeholder('img')
    label = opr.placeholder('label')

    _ = img
    _ = conv('conv1', _, cout=16, cin=1, k=5)
    _ = pool(_, k=2)
    _ = conv('conv2', _, cout=32, cin=16, k=3)
    _ = pool(_, k=2)
    _ = view(_)
    _ = fc('fc1', _, cout=64, cin=800)
    _ = fc('softmax', _, cout=10, cin=64, nonlin=False)
    pred = softmax(_, 1)
    loss = cross_entropy(pred, label)
    return pred, loss


def make_func(pred, loss, is_train, lr=0.1):
    cg = CompGraph()

    if is_train:
        lr = opr.parameter(lr)

        optimizable = []
        for o in cg.find_all_oprs([loss]):
            if isinstance(o, opr.Parameter):
                optimizable.append(o.outputs[0])
        updates = []
        outputs = []
        for o in optimizable:
            updates.append(opr.update(o, o - lr * opr.grad(loss, o)))

        func = cg.compile([loss] + outputs + updates)
        return func, lr.owner_opr
    else:
        func = cg.compile([pred])
        return func


def main():
    pred, loss = make_net()
    print('net constructed')
    func_train = make_func(pred, loss, True)
    func_test = make_func(pred, loss, False)
    print('func constructed')

    data = make_data('train')
    testdata = make_data('test')
    nr_minibatch = len(data['img']) // 128
    nr_test_minibatch = len(testdata['img']) // 128
    print('data constructed')

    def do_train():
        quick_loss = 0
        for minibatch in range(nr_minibatch):
            print('minibatch', minibatch)
            batch_data = dict()
            for k, v in data.items():
                batch_data[k] = np.ascontiguousarray(data[k][minibatch*128:minibatch*128+128], dtype='float32')
            quick_loss += func_train(**batch_data)[0]
        return quick_loss

    def do_test():
        nr_total, nr_correct = 0, 0
        for minibatch in range(nr_test_minibatch):
            print('minibatch', minibatch)
            batch_data = dict()
            for k, v in testdata.items():
                batch_data[k] = np.ascontiguousarray(testdata[k][minibatch*128:minibatch*128+128], dtype='float32')
            pred = func_test(img=batch_data['img'])[0].argmax(axis=1)
            nr_correct += (pred == batch_data['label']).sum()
            nr_total += len(pred)
        return nr_total, nr_correct

    # env.network.set_param_values_dict(io.load('epoch2.neuartist.pkl'))

    print('begin training')
    for epoch in range(100):
        if epoch != 0:
            quick_loss = do_train()
        else:
            quick_loss = 0
        nr_total, nr_correct = do_test()

        print('epoch={}, loss={}, accuracy={}'.format(epoch, quick_loss / nr_minibatch, nr_correct / nr_total))
    print('end training')


if __name__ == '__main__':
    main()