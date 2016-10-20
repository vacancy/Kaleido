# -*- coding:utf8 -*-
# File   : svm.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 10/19/16 20:29
# 
# This file is part of Kaleido
# (c) 2016 vccy.xyz

from kaleido import opr
from kaleido.graph import CompGraph
import numpy as np


def make_data():
    import pickle

    with open('data.pkl', 'rb') as f:
        a = pickle.load(f, encoding='latin1')
    x = np.array([i[0] for i in a])
    y = np.array([i[1] for i in a])
    x[np.where(y == 1)] += [0.27, -0.025]
    return x, y


def eval_accuracy(pred, label):
    pred = pred.reshape(-1)
    label = label.copy()
    label[np.where(label == -1)] = 0
    raw = pred != label
    return min(raw.sum(), (1 - raw).sum())


def demo_graph(xs, pred, label, W, b):
    assert xs.shape[1] == 2

    from matplotlib import pyplot as plt
    plt.plot(xs[np.where(label == 1), 0], xs[np.where(label == 1), 1], 'ro')
    plt.plot(xs[np.where(label == -1), 0], xs[np.where(label == -1), 1], 'bo')

    lxs = np.arange(-2, 2, 0.01)
    lys = (- b - lxs * W[0]) / W[1]
    plt.plot(lxs, lys)
    plt.xlim(0.3, 1.2)
    plt.ylim(0.65, 1.2)
    plt.show()


def make_svm():
    W = opr.parameter(np.random.normal(size=[2, 1]), name='W')
    b = opr.parameter(0, name='b')
    x = opr.placeholder(name='x')

    y = opr.matmul(x, W) + b
    pred = y >= 0

    label = opr.placeholder(name='label')
    hinge_loss = opr.max(1 - label * y, 0)
    hinge_loss = hinge_loss.sum() / hinge_loss.shapeidx(0)

    decay = opr.mul(W, W) / 2
    decay = decay.sum()

    loss = 0.01 * decay + hinge_loss

    return pred, loss


def make_func(pred, loss, is_train, lr=1):
    cg = CompGraph()

    if is_train:
        optimizable = []
        for o in cg.find_all_oprs([loss]):
            if isinstance(o, opr.Parameter):
                optimizable.append(o.outputs[0])
        updates = []
        for o in optimizable:
            updates.append(opr.update(o, o - lr * opr.grad(loss, o)))

        func = cg.compile([loss] + updates)
        return func
    else:
        func = cg.compile([pred])
        return func, cg.find_opr([loss], 'W'), cg.find_opr([loss], 'b')


def main():
    pred, loss = make_svm()

    xs, ys = make_data()
    lr = 1

    train_func = make_func(pred, loss, True, lr=lr)
    for i in range(1000):
        res = train_func(x=xs, label=ys)
        print('iter {}, loss={}'.format(i, res[0]))

    test_func, W, b = make_func(pred, loss, False)
    pred, = test_func(x=xs)

    print('Accuracy:', (len(ys) - eval_accuracy(pred, ys)) / len(ys) * 100, '%')
    demo_graph(xs, pred, ys, W.get_value(), b.get_value())


if __name__ == '__main__':
    main()
